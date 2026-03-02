from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


@dataclass
class AUScore:
    per_au_f1: torch.Tensor
    mean_f1: float
    per_au_auc: Optional[torch.Tensor]
    mean_auc: Optional[float]


def compute_f1_per_au(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    # logits, targets: [N, num_aus]
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(targets.dtype)
    targets = targets.to(preds.dtype)

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    return f1


def compute_auc_per_au(logits: torch.Tensor, targets: torch.Tensor) -> Optional[torch.Tensor]:
    if roc_auc_score is None:
        return None

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    t = targets.detach().cpu().numpy()
    aucs = []
    for i in range(t.shape[1]):
        y_true = t[:, i]
        if y_true.max() == y_true.min():
            aucs.append(float("nan"))
            continue
        aucs.append(float(roc_auc_score(y_true, probs[:, i])))
    return torch.tensor(aucs)


class AUDetectionMeter:
    def __init__(self) -> None:
        self.logits = []
        self.targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.logits.append(logits.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float | torch.Tensor | None]:
        if not self.logits:
            raise RuntimeError("No samples accumulated")
        logits = torch.cat(self.logits, dim=0)
        targets = torch.cat(self.targets, dim=0)
        per_au_f1 = compute_f1_per_au(logits, targets)
        auc = compute_auc_per_au(logits, targets)
        out: Dict[str, float | torch.Tensor | None] = {
            "per_au_f1": per_au_f1,
            "mean_f1": float(torch.nanmean(per_au_f1).item()),
            "per_au_auc": auc,
            "mean_auc": float(torch.nanmean(auc).item()) if auc is not None else None,
        }
        return out

    def reset(self) -> None:
        self.logits.clear()
        self.targets.clear()
