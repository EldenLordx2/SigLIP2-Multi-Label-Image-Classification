import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_labels(labels_txt: str) -> List[str]:
    with open(labels_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_model_id(
    model_id: Optional[str],
    model_root: Optional[str],
    model_size: Optional[str],
    model_patch: Optional[int],
    image_size: Optional[int],
) -> str:
    if model_id:
        return model_id

    if not model_size or model_patch is None or image_size is None:
        raise ValueError(
            "Either --model_id or the trio (--model_size, --model_patch, --image_size) must be provided."
        )

    name = f"siglip2-{model_size}-patch{int(model_patch)}-{int(image_size)}"
    if model_root:
        return str(Path(model_root) / name)
    return f"google/{name}"


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out
