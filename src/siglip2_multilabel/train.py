import argparse
import math
import os

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .augmentations import TrainImageTransform, ValImageTransform
from .data import Collator, MultiLabelImageDataset
from .losses import MultiLabelAsymmetricLoss
from .metrics import multilabel_map_from_logits
from .modeling import build_model_and_processor, build_param_groups
from .utils import ensure_dir, load_labels, save_json, set_seed


class EMA:
    def __init__(self, decay: float = 0.9997):
        self.decay = float(decay)
        self.shadow = {}
        self._inited = False

    @torch.no_grad()
    def init(self, model):
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().cpu().clone()
        self._inited = True

    @torch.no_grad()
    def update(self, model):
        if not self._inited:
            self.init(model)
            return
        for name, p in model.named_parameters():
            if p.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = p.detach().float().cpu().clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(p.detach().float().cpu(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.device, dtype=p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])


def build_parser():
    parser = argparse.ArgumentParser(description="Train SigLIP2 multi-label classifier with ML-Decoder.")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--model_root", type=str, default=None, help="Optional local root; composed with size/patch/resolution.")
    parser.add_argument("--model_size", type=str, default="base", choices=["tiny", "small", "base", "large", "giant", "so400m"])
    parser.add_argument("--model_patch", type=int, default=16, help="Patch size, e.g. 16 or 32.")
    parser.add_argument("--image_size", type=int, default=256, help="Input resolution and model variant resolution, e.g. 224/256/384.")
    parser.add_argument("--train_txt", type=str, required=True)
    parser.add_argument("--val_txt", type=str, required=True)
    parser.add_argument("--labels_txt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9997)
    parser.add_argument("--query_num", type=int, default=80)
    parser.add_argument("--embed_dim", type=int, default=0)
    parser.add_argument("--mldecoder_layers", type=int, default=1)
    parser.add_argument("--mldecoder_heads", type=int, default=8)
    parser.add_argument("--mldecoder_ff", type=int, default=0)
    parser.add_argument("--mldecoder_act", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--keep_self_attn", action="store_true")
    parser.add_argument("--train_query_embed", action="store_true")
    parser.add_argument("--token_stride", type=int, default=2)
    parser.add_argument("--keep_cls", action="store_true")
    parser.add_argument("--pos_drop", type=float, default=0.0)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--cutout_length", type=int, default=224)
    parser.add_argument("--no_randaugment", action="store_true")
    parser.add_argument("--save_dir", type=str, default="outputs/default_run")
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--print_classwise_ap", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=args.grad_accum,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    labels = load_labels(args.labels_txt)
    num_labels = len(labels)
    model, processor, model_id, feat_dim = build_model_and_processor(args, num_labels=num_labels, device=str(device))

    if args.freeze_vision:
        for p in model.vision.parameters():
            p.requires_grad = False

    train_ds = MultiLabelImageDataset(args.train_txt, num_labels, transform=TrainImageTransform(args.image_size, args.cutout_length, not args.no_randaugment), dummy_size=(args.image_size, args.image_size))
    val_ds = MultiLabelImageDataset(args.val_txt, num_labels, transform=ValImageTransform(args.image_size), dummy_size=(args.image_size, args.image_size))
    collate_fn = Collator(processor, disable_resize_ops=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, persistent_workers=args.num_workers > 0)

    optimizer = torch.optim.AdamW(build_param_groups(model, args.lr_backbone, args.lr_head, args.weight_decay), betas=(0.9, 0.999), eps=1e-8)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    asl_loss = MultiLabelAsymmetricLoss(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg, clip=args.asl_clip)
    ema = EMA(decay=args.ema_decay) if args.use_ema else None
    best_map = -1.0
    global_step = 0

    if accelerator.is_main_process:
        save_json(vars(args) | {"resolved_model_id": model_id, "feat_dim": feat_dim, "num_labels": num_labels}, os.path.join(args.save_dir, "run_config.json"))

    for epoch in range(args.epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", disable=not accelerator.is_main_process)
        for batch in train_bar:
            with accelerator.accumulate(model):
                logits = model(pixel_values=batch["pixel_values"])["logits"]
                loss_raw = asl_loss(logits, batch["labels"])
                valid = batch["valid"].unsqueeze(-1)
                valid_sum = valid.sum()
                loss = logits.sum() * 0.0 if valid_sum.item() < 0.5 else (loss_raw * valid).sum() / valid_sum
                accelerator.backward(loss)
                if args.max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None and accelerator.sync_gradients and accelerator.is_main_process:
                    ema.update(accelerator.unwrap_model(model))
            global_step += 1
            if accelerator.is_main_process:
                train_bar.set_postfix(loss=float(loss.item()), lr=float(scheduler.get_last_lr()[0]))
            if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0 and accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.save_dir, f"step_{global_step}.pt"))

        model.eval()
        all_logits, all_labels = [], []
        backup = None
        if ema is not None and accelerator.is_main_process:
            backup = ema.apply_to(accelerator.unwrap_model(model))

        with torch.no_grad():
            for batch in val_loader:
                logits = model(pixel_values=batch["pixel_values"])["logits"]
                logits_g = accelerator.gather_for_metrics(logits)
                labels_g = accelerator.gather_for_metrics(batch["labels"])
                valid_g = accelerator.gather_for_metrics(batch["valid"])
                mask = valid_g > 0.5
                if mask.any():
                    all_logits.append(logits_g[mask].float().cpu())
                    all_labels.append(labels_g[mask].float().cpu())

        if ema is not None and accelerator.is_main_process and backup is not None:
            ema.restore(accelerator.unwrap_model(model), backup)

        if all_logits and accelerator.is_main_process:
            mAP, ap_per_class = multilabel_map_from_logits(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
            print(f"[Val] Epoch {epoch + 1}: mAP={mAP:.6f}")
            if args.print_classwise_ap:
                vals, inds = torch.topk(ap_per_class, k=min(10, len(ap_per_class)), largest=True)
                for v, i in zip(vals.tolist(), inds.tolist()):
                    print(f"  [{i:03d}] {labels[i]}: {v:.6f}")
            if mAP > best_map:
                best_map = mAP
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.save_dir, "best_model.pt"))
                print(f"Saved best model: {os.path.join(args.save_dir, 'best_model.pt')} (best_mAP={best_map:.6f})")


if __name__ == "__main__":
    main()
