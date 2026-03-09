from typing import Tuple

import torch


@torch.no_grad()
def average_precision_per_class(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.float()
    pos = targets.sum()
    if pos.item() < 1.0:
        return scores.new_tensor(0.0)

    order = torch.argsort(scores, descending=True)
    t = targets[order]
    tp = torch.cumsum(t, dim=0)
    fp = torch.cumsum(1.0 - t, dim=0)

    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / pos

    recall = torch.cat([recall.new_tensor([0.0]), recall, recall.new_tensor([1.0])])
    precision = torch.cat([precision.new_tensor([1.0]), precision, precision.new_tensor([0.0])])
    return torch.trapz(precision, recall)


@torch.no_grad()
def multilabel_map_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, torch.Tensor]:
    probs = torch.sigmoid(logits).float()
    labels = labels.float()
    aps = [average_precision_per_class(probs[:, c], labels[:, c]) for c in range(probs.shape[1])]
    ap_per_class = torch.stack(aps, dim=0)
    return ap_per_class.mean().item(), ap_per_class
