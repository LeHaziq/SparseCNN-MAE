from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from typing import Any

import torch

from src.data.disfa import DISFADataset
from src.models.au_head import VideoAUModel
from src.models.mae import VideoMAE
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.metrics import AUDetectionMeter
from src.utils.version import warn_if_not_torch_210


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def _build_model(cfg: dict[str, Any]) -> VideoAUModel:
    m = cfg["model"]
    backbone = VideoMAE(
        in_channels=int(m.get("in_channels", 3)),
        input_size=tuple(m.get("input_size", [32, 112, 112])),
        patch_size=tuple(m.get("patch_size", [2, 16, 16])),
        embed_dim=int(m.get("embed_dim", 192)),
        mask_ratio=float(m.get("mask_ratio", 0.9)),
        encoder_channels=tuple(m.get("encoder_channels", [192, 256])),
        encoder_blocks=tuple(m.get("encoder_blocks", [2, 2])),
        downsample_stages=tuple(m.get("downsample_stages", [1])),
        decoder_channels=int(m.get("decoder_channels", 192)),
        prefer_spconv=bool(m.get("prefer_spconv", True)),
        loss_type=str(m.get("loss_type", "mse")),
        visible_loss_weight=float(m.get("visible_loss_weight", 0.0)),
    )
    model = VideoAUModel(
        backbone=backbone,
        num_aus=len(cfg["data"]["au_list"]),
        out_frames=int(cfg["data"].get("clip_len", 32)),
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DISFA AU detector")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    warn_if_not_torch_210()

    ckpt_path = args.ckpt or cfg.get("ckpt")
    if not ckpt_path:
        raise ValueError("Checkpoint path is required (use --ckpt or ckpt=/path/to.ckpt)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    data_cfg = cfg["data"]
    ds = DISFADataset(
        manifest_path=data_cfg["manifest"],
        split=args.split,
        au_list=data_cfg["au_list"],
        clip_len=int(data_cfg.get("clip_len", 32)),
        stride=int(data_cfg.get("stride", 1)),
        train=False,
        label_threshold=float(data_cfg.get("label_threshold", 0.0)),
        mean=data_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=data_cfg.get("std", [0.229, 0.224, 0.225]),
        backend=data_cfg.get("video_backend", "auto"),
        align_mode=data_cfg.get("align_mode", "off"),
        align_cache_dir=data_cfg.get("align_cache_dir"),
        color_jitter=False,
    )

    train_cfg = cfg["train"]
    num_workers = int(train_cfg.get("num_workers", 4))
    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        prefetch_factor = train_cfg.get("prefetch_factor", 2)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **loader_kwargs,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    meter = AUDetectionMeter()
    total_loss = 0.0
    n = 0
    amp_enabled = bool(train_cfg.get("amp", True))

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled):
                logits = model(video)
                loss = criterion(logits, labels)

            meter.update(
                logits.permute(0, 2, 1).reshape(-1, logits.shape[1]),
                labels.permute(0, 2, 1).reshape(-1, labels.shape[1]),
            )
            total_loss += float(loss.item())
            n += 1

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(1, n)

    out = {
        "split": args.split,
        "checkpoint": ckpt_path,
        "loss": metrics["loss"],
        "mean_f1": metrics["mean_f1"],
        "mean_auc": metrics["mean_auc"],
        "per_au_f1": [float(x) for x in metrics["per_au_f1"].tolist()],
        "per_au_auc": None
        if metrics.get("per_au_auc") is None
        else [float(x) for x in metrics["per_au_auc"].tolist()],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
