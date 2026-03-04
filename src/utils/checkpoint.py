import pathlib
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_p = p.with_suffix(".ckpt.tmp")
    torch.save(state, temp_p)
    temp_p.replace(p)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


class CheckpointManager:
    def __init__(self, out_dir: str):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: Optional[float] = None

    def load_best_metric(self, metric: Optional[float]) -> None:
        self.best_metric = metric

    def save_latest(self, state: Dict[str, Any]) -> str:
        path = self.out_dir / "latest.ckpt"
        temp_path = self.out_dir / "latest.ckpt.tmp"
        torch.save(state, temp_path)
        temp_path.replace(path)
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
        temp_path = self.out_dir / "best.ckpt.tmp"
        state = dict(state)
        state["best_metric"] = metric
        torch.save(state, temp_path)
        temp_path.replace(path)
        return str(path)
