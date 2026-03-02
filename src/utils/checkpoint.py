import pathlib
from typing import Any, Dict, Optional

import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


class CheckpointManager:
    def __init__(self, out_dir: str):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: Optional[float] = None

    def save_latest(self, state: Dict[str, Any]) -> str:
        path = self.out_dir / "latest.ckpt"
        torch.save(state, path)
        return str(path)

    def save_best(self, state: Dict[str, Any], metric: float, mode: str = "min") -> Optional[str]:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        improved = (
            self.best_metric is None
            or (metric < self.best_metric if mode == "min" else metric > self.best_metric)
        )
        if not improved:
            return None

        self.best_metric = metric
        path = self.out_dir / "best.ckpt"
        state = dict(state)
        state["best_metric"] = metric
        torch.save(state, path)
        return str(path)
