from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
import math
import numpy as np
import pathlib
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.voxceleb2 import VoxCeleb2MAEDataset
from src.models.mae import VideoMAE
from src.utils.checkpoint import CheckpointManager, load_checkpoint
from src.utils.config import load_config
from src.utils.logging import create_tb_writer, get_logger
from src.utils.seed import set_seed, worker_init_fn
from src.utils.version import warn_if_not_torch_210


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def _make_grad_scaler(device: torch.device, amp_enabled: bool):
    enabled = amp_enabled and device.type == "cuda"
    # torch.cuda.amp.GradScaler is deprecated in newer PyTorch versions.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _build_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    data_cfg = cfg["data"]
    train = split == "train"
    manifest = data_cfg["train_manifest"] if train else data_cfg["val_manifest"]

    ds = VoxCeleb2MAEDataset(
        manifest_path=manifest,
        clip_len=int(data_cfg.get("clip_len", 32)),
        stride=int(data_cfg.get("stride", 2)),
        train=train,
        mean=data_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=data_cfg.get("std", [0.229, 0.224, 0.225]),
        backend=data_cfg.get("video_backend", "auto"),
        align_mode=data_cfg.get("align_mode", "off"),
        align_cache_dir=data_cfg.get("align_cache_dir"),
        color_jitter=bool(data_cfg.get("color_jitter", True if train else False)),
    )

    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=train,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=train,
        worker_init_fn=worker_init_fn,
    )
    return loader


def _build_model(cfg: dict[str, Any]) -> VideoMAE:
    m = cfg["model"]
    model = VideoMAE(
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
    return model


def _build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def validate(
    model: VideoMAE,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_masked = 0.0
    total_visible = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        video = batch["video"].to(device, non_blocking=True)
        with _autocast_context(device, amp_enabled):
            out = model(video)
        total_loss += float(out.loss.item())
        total_masked += float(out.masked_loss.item())
        total_visible += float(out.visible_loss.item())
        n += 1

    if n == 0:
        return {"loss": float("nan"), "masked_loss": float("nan"), "visible_loss": float("nan")}

    return {
        "loss": total_loss / n,
        "masked_loss": total_masked / n,
        "visible_loss": total_visible / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain fully-convolutional Video-MAE on VoxCeleb2")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny 50-step smoke run")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    warn_if_not_torch_210()

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    out_dir = pathlib.Path(cfg.get("output_dir", "outputs/pretrain_voxceleb2"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("pretrain", str(out_dir / "train.log"))
    writer = create_tb_writer(str(out_dir / "tb"))

    train_loader = _build_loader(cfg, split="train")
    val_loader = _build_loader(cfg, split="val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)

    train_cfg = cfg["train"]
    amp_enabled = bool(train_cfg.get("amp", True))
    accum_steps = int(train_cfg.get("accum_steps", 4))
    epochs = int(train_cfg.get("epochs", 30))
    max_steps = int(train_cfg.get("max_steps", -1))
    smoke_steps = int(train_cfg.get("smoke_steps", 50))
    smoke = args.smoke or bool(train_cfg.get("smoke", False))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )

    updates_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    total_updates = smoke_steps if smoke else (max_steps if max_steps > 0 else epochs * updates_per_epoch)
    warmup_steps = int(train_cfg.get("warmup_steps", 1000))
    scheduler = _build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_updates)
    scaler = _make_grad_scaler(device=device, amp_enabled=amp_enabled)

    start_epoch = 0
    global_update_step = 0
    ckpt_mgr = CheckpointManager(str(out_dir / "checkpoints"))

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_update_step = int(ckpt.get("global_update_step", 0))
        if "best_metric" in ckpt:
            ckpt_mgr.load_best_metric(ckpt["best_metric"])
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
        if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        logger.info("Resumed from %s at epoch=%d update_step=%d", args.resume, start_epoch, global_update_step)

    log_interval = int(train_cfg.get("log_interval", 10))
    val_interval = int(train_cfg.get("val_interval", 200))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    model.train()
    done = False

    for epoch in range(start_epoch, epochs if not smoke else 999999):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        num_batches = len(train_loader)
        for step, batch in enumerate(pbar):
            video = batch["video"].to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled):
                out = model(video)
                loss = out.loss / accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_update_step = ((step + 1) % accum_steps == 0) or ((step + 1) == num_batches)
            if is_update_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_update_step += 1

                writer.add_scalar("train/loss", float(out.loss.item()), global_update_step)
                writer.add_scalar("train/masked_loss", float(out.masked_loss.item()), global_update_step)
                writer.add_scalar("train/visible_loss", float(out.visible_loss.item()), global_update_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_update_step)
                writer.add_scalar("train/used_spconv", float(out.used_spconv), global_update_step)

                if global_update_step % log_interval == 0:
                    logger.info(
                        "epoch=%d update=%d loss=%.5f masked=%.5f visible=%.5f lr=%.3e",
                        epoch,
                        global_update_step,
                        out.loss.item(),
                        out.masked_loss.item(),
                        out.visible_loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )

                if global_update_step % val_interval == 0 or global_update_step >= total_updates:
                    metrics = validate(
                        model,
                        val_loader,
                        device=device,
                        amp_enabled=amp_enabled,
                        max_steps=10 if smoke else None,
                    )
                    writer.add_scalar("val/loss", metrics["loss"], global_update_step)
                    writer.add_scalar("val/masked_loss", metrics["masked_loss"], global_update_step)
                    writer.add_scalar("val/visible_loss", metrics["visible_loss"], global_update_step)

                    state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                        "epoch": epoch,
                        "global_update_step": global_update_step,
                        "best_metric": ckpt_mgr.best_metric,
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                        "config": cfg,
                    }
                    ckpt_mgr.save_latest(state)
                    best_path = ckpt_mgr.save_best(state, metric=float(metrics["loss"]), mode="min")
                    if best_path:
                        logger.info("New best checkpoint: %s (val_loss=%.5f)", best_path, metrics["loss"])

                if global_update_step >= total_updates:
                    done = True
                    break

        if done:
            break

    logger.info("Training complete. total_update_steps=%d", global_update_step)
    writer.close()


if __name__ == "__main__":
    main()
