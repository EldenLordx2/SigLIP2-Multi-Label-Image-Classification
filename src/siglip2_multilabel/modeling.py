import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .utils import resolve_model_id


class CrossAttnOnlyDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str = "relu", norm_first: bool = True, remove_self_attn: bool = True):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = None if remove_self_attn else nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def _sa_block(self, x):
        if self.self_attn is None:
            return x
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        return self.dropout(attn_out)

    def _ca_block(self, x, mem):
        attn_out, _ = self.cross_attn(x, mem, mem, need_weights=False)
        return self.dropout(attn_out)

    def _ff_block(self, x):
        x = self.linear2(self.dropout_ff(self.act(self.linear1(x))))
        return self.dropout(x)

    def forward(self, tgt, memory):
        if self.norm_first:
            x = tgt + self._sa_block(self.norm1(tgt))
            x = x + self._ca_block(self.norm2(x), memory)
            x = x + self._ff_block(self.norm3(x))
            return x
        x = self.norm1(tgt + self._sa_block(tgt))
        x = self.norm2(x + self._ca_block(x, memory))
        x = self.norm3(x + self._ff_block(x))
        return x


class MLDecoderPaddleStyle(nn.Module):
    def __init__(self, num_labels: int, in_dim: int, query_num: Optional[int] = None, embed_dim: Optional[int] = None, depth: int = 1, num_heads: int = 8, mlp_hidden_dim: Optional[int] = None, dropout: float = 0.1, activation: str = "relu", freeze_query_embed: bool = True, remove_self_attn: bool = True, norm_first: bool = True):
        super().__init__()
        self.num_labels = int(num_labels)
        if query_num is None:
            query_num = self.num_labels
        query_num = max(1, min(int(query_num), self.num_labels))
        self.query_num = query_num
        if embed_dim is None:
            embed_dim = in_dim
        self.embed_dim = int(embed_dim)
        if mlp_hidden_dim is None:
            mlp_hidden_dim = max(2048, self.embed_dim * 4)

        self.input_proj = nn.Linear(in_dim, self.embed_dim)
        self.query_embed = nn.Embedding(self.query_num, self.embed_dim)
        if freeze_query_embed:
            self.query_embed.weight.requires_grad_(False)

        self.layers = nn.ModuleList([
            CrossAttnOnlyDecoderLayer(
                d_model=self.embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_hidden_dim,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                remove_self_attn=remove_self_attn,
            )
            for _ in range(depth)
        ])

        self.group_factor = int(math.ceil(self.num_labels / self.query_num))
        self.group_conv = nn.Conv1d(
            in_channels=self.query_num * self.embed_dim,
            out_channels=self.query_num * self.group_factor,
            kernel_size=1,
            groups=self.query_num,
            bias=True,
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.xavier_normal_(self.group_conv.weight)
        nn.init.constant_(self.group_conv.bias, 0.0)

    def group_fc_pool(self, x: torch.Tensor) -> torch.Tensor:
        bsz, q, c = x.shape
        x = x.reshape(bsz, q * c).unsqueeze(-1)
        x = self.group_conv(x).squeeze(-1)
        return x[:, : self.num_labels]

    def forward(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        bsz = memory_tokens.shape[0]
        mem = F.relu(self.input_proj(memory_tokens)).transpose(0, 1)
        tgt = self.query_embed.weight.unsqueeze(1).expand(self.query_num, bsz, self.embed_dim)
        for layer in self.layers:
            tgt = layer(tgt=tgt, memory=mem)
        return self.group_fc_pool(tgt.transpose(0, 1))


class Siglip2ForMultiLabelMLDecoder(nn.Module):
    def __init__(self, backbone, num_labels: int, feat_dim: int, query_num: Optional[int] = None, embed_dim: Optional[int] = None, mldecoder_layers: int = 1, mldecoder_heads: int = 8, mldecoder_ff: Optional[int] = None, remove_self_attn: bool = True, freeze_query_embed: bool = True, activation: str = "relu", token_stride: int = 2, drop_cls: bool = True, pos_drop: float = 0.0):
        super().__init__()
        self.vision = backbone.vision_model if hasattr(backbone, "vision_model") else backbone
        self.drop_cls = drop_cls
        self.token_stride = max(1, token_stride)
        self.pos_drop = nn.Dropout(pos_drop) if pos_drop > 0 else nn.Identity()
        self.mldecoder = MLDecoderPaddleStyle(
            num_labels=num_labels,
            in_dim=feat_dim,
            query_num=query_num,
            embed_dim=embed_dim,
            depth=mldecoder_layers,
            num_heads=mldecoder_heads,
            mlp_hidden_dim=mldecoder_ff,
            dropout=0.1,
            activation=activation,
            freeze_query_embed=freeze_query_embed,
            remove_self_attn=remove_self_attn,
            norm_first=True,
        )

    def forward(self, pixel_values):
        out = self.vision(pixel_values=pixel_values)
        if not hasattr(out, "last_hidden_state") or out.last_hidden_state is None:
            raise RuntimeError("Vision output has no last_hidden_state; cannot use ML-Decoder.")
        tokens = out.last_hidden_state
        if self.drop_cls and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        if self.token_stride > 1 and tokens.shape[1] > self.token_stride:
            tokens = tokens[:, :: self.token_stride, :]
        tokens = self.pos_drop(tokens)
        return {"logits": self.mldecoder(tokens)}


@torch.no_grad()
def infer_feat_dim(backbone, processor, device, image_size=256) -> int:
    backbone.eval()
    dummy_img = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    try:
        inputs = processor(images=[dummy_img], return_tensors="pt", do_resize=False, do_rescale=True, do_normalize=True)
    except TypeError:
        inputs = processor(images=[dummy_img], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    vision = backbone.vision_model if hasattr(backbone, "vision_model") else backbone
    out = vision(pixel_values=pixel_values)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        feat_dim = out.last_hidden_state.shape[-1]
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat_dim = out.pooler_output.shape[-1]
    else:
        raise RuntimeError("Cannot infer feature dimension from vision output.")
    backbone.train()
    return int(feat_dim)


def build_model_and_processor(args, num_labels: int, device: Optional[str] = None):
    model_id = resolve_model_id(args.model_id, args.model_root, args.model_size, args.model_patch, args.image_size)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=args.local_files_only)
    backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True, local_files_only=args.local_files_only)
    infer_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(infer_device)
    feat_dim = infer_feat_dim(backbone, processor, infer_device, image_size=min(int(args.image_size), 256))
    backbone = backbone.to("cpu")

    query_num = None if getattr(args, "query_num", 0) in (None, 0) else int(args.query_num)
    embed_dim = None if getattr(args, "embed_dim", 0) in (None, 0) else int(args.embed_dim)
    mldecoder_ff = None if getattr(args, "mldecoder_ff", 0) in (None, 0) else int(args.mldecoder_ff)

    model = Siglip2ForMultiLabelMLDecoder(
        backbone=backbone,
        num_labels=num_labels,
        feat_dim=feat_dim,
        query_num=query_num,
        embed_dim=embed_dim,
        mldecoder_layers=args.mldecoder_layers,
        mldecoder_heads=args.mldecoder_heads,
        mldecoder_ff=mldecoder_ff,
        remove_self_attn=not args.keep_self_attn,
        freeze_query_embed=not args.train_query_embed,
        activation=args.mldecoder_act,
        token_stride=args.token_stride,
        drop_cls=not args.keep_cls,
        pos_drop=args.pos_drop,
    )
    return model, processor, model_id, feat_dim


def is_no_decay(name: str, p: torch.nn.Parameter) -> bool:
    if p.ndim == 1:
        return True
    ln = name.lower()
    return ln.endswith(".bias") or "layernorm" in ln or ".ln" in ln or "norm" in ln or "embedding" in ln or "embed" in ln


def build_param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict]:
    groups = []
    def add_groups(named_params, lr):
        decay, no_decay = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            (no_decay if is_no_decay(n, p) else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    add_groups(list(model.vision.named_parameters()), lr_backbone)
    add_groups(list(model.mldecoder.named_parameters()), lr_head)
    return groups
