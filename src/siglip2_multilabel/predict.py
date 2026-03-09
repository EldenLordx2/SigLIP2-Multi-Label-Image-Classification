import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Collator, ImageOnlyDataset
from .modeling import build_model_and_processor
from .utils import load_labels, strip_prefix_if_present


def build_parser():
    ap = argparse.ArgumentParser(description="Predict with SigLIP2 multi-label ML-Decoder model.")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--model_root", type=str, default=None)
    ap.add_argument("--model_size", type=str, default="base", choices=["tiny", "small", "base", "large", "giant", "so400m"])
    ap.add_argument("--model_patch", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--labels_txt", required=True)
    ap.add_argument("--input_txt", required=True)
    ap.add_argument("--output_txt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--query_num", type=int, default=0)
    ap.add_argument("--embed_dim", type=int, default=0)
    ap.add_argument("--mldecoder_layers", type=int, default=1)
    ap.add_argument("--mldecoder_heads", type=int, default=8)
    ap.add_argument("--mldecoder_ff", type=int, default=0)
    ap.add_argument("--mldecoder_act", type=str, default="relu", choices=["relu", "gelu"])
    ap.add_argument("--keep_self_attn", action="store_true")
    ap.add_argument("--train_query_embed", action="store_true")
    ap.add_argument("--token_stride", type=int, default=2)
    ap.add_argument("--keep_cls", action="store_true")
    ap.add_argument("--pos_drop", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=4)
    return ap


@torch.no_grad()
def main():
    args = build_parser().parse_args()
    labels = load_labels(args.labels_txt)
    model, processor, _, _ = build_model_and_processor(args, num_labels=len(labels), device=args.device)
    model = model.to(args.device).eval()

    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint format not understood. Expect a state_dict dict.")
    model.load_state_dict(strip_prefix_if_present(sd, "module."), strict=True)

    ds = ImageOnlyDataset(args.input_txt, dummy_size=(args.image_size, args.image_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"), collate_fn=Collator(processor, disable_resize_ops=False))

    with open(args.output_txt, "w", encoding="utf-8") as fout:
        for batch in tqdm(dl, total=len(dl), desc="Predicting", ncols=100):
            probs = torch.sigmoid(model(pixel_values=batch["pixel_values"].to(args.device, non_blocking=True))["logits"])
            preds = probs >= args.threshold
            for i, path in enumerate(batch["paths"]):
                if batch["read_errors"][i]:
                    fout.write(f"{path}\n")
                    continue
                pred_ids = preds[i].nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(pred_ids, int):
                    pred_ids = [pred_ids]
                pred_labels = [labels[j] for j in pred_ids]
                fout.write(f"{path}\t{';'.join(pred_labels) if pred_labels else ''}\n")

    print(f"Results saved to: {args.output_txt}")


if __name__ == "__main__":
    main()
