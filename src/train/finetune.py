from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import pathlib
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.disfa import DISFADataset
from src.models.au_head import VideoAUModel
from src.models.mae import VideoMAE
from src.utils.checkpoint import CheckpointManager, load_checkpoint
from src.utils.config import load_config
from src.utils.logging import create_tb_writer, get_logger
from src.utils.metrics import AUDetectionMeter
from src.utils.seed import set_seed, worker_init_fn
from src.utils.version import warn_if_not_torch_210


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def _build_backbone(model_cfg: dict[str, Any]) -> VideoMAE:
    return VideoMAE(
        in_channels=int(model_cfg.get("in_channels", 3)),
        input_size=tuple(model_cfg.get("input_size", [32, 112, 112])),
        patch_size=tuple(model_cfg.get("patch_size", [2, 16, 16])),
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        mask_ratio=float(model_cfg.get("mask_ratio", 0.9)),
        encoder_channels=tuple(model_cfg.get("encoder_channels", [192, 256])),
        encoder_blocks=tuple(model_cfg.get("encoder_blocks", [2, 2])),
        downsample_stages=tuple(model_cfg.get("downsample_stages", [1])),
        decoder_channels=int(model_cfg.get("decoder_channels", 192)),
        prefer_spconv=bool(model_cfg.get("prefer_spconv", True)),
        loss_type=str(model_cfg.get("loss_type", "mse")),
        visible_loss_weight=float(model_cfg.get("visible_loss_weight", 0.0)),
    )


def _build_loader(cfg: dict[str, Any], split: str, smoke: bool = False) -> DataLoader:
    data_cfg = cfg["data"]
    ds = DISFADataset(
        manifest_path=data_cfg["manifest"],
        split=split,
        au_list=data_cfg["au_list"],
        clip_len=int(data_cfg.get("clip_len", 32)),
        stride=int(data_cfg.get("stride", 1)),
        train=(split == "train"),
        label_threshold=float(data_cfg.get("label_threshold", 0.0)),
        mean=data_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=data_cfg.get("std", [0.229, 0.224, 0.225]),
        backend=data_cfg.get("video_backend", "auto"),
        align_mode=data_cfg.get("align_mode", "off"),
        align_cache_dir=data_cfg.get("align_cache_dir"),
        color_jitter=bool(data_cfg.get("color_jitter", split == "train")),
    )
    if smoke:
        # Keep memory/time small while still exercising the full path.
        ds.rows = ds.rows[: max(4, min(32, len(ds.rows)))]

    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=(split == "train"),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=(split == "train"),
        worker_init_fn=worker_init_fn,
    )
    return loader


def _build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(
    model: VideoAUModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    max_steps: int | None = None,
) -> dict[str, Any]:
    model.eval()
    meter = AUDetectionMeter()
    total_loss = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        video = batch["video"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with _autocast_context(device, amp_enabled):
            logits = model(video)
            loss = criterion(logits, labels)

        # Flatten frames into sample axis for metrics.
        meter.update(
            logits.permute(0, 2, 1).reshape(-1, logits.shape[1]),
            labels.permute(0, 2, 1).reshape(-1, labels.shape[1]),
        )
        total_loss += float(loss.item())
        n += 1

    score = meter.compute()
    score["loss"] = total_loss / max(1, n)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AU detector on DISFA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained MAE checkpoint")
    parser.add_argument("--smoke", action="store_true")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    warn_if_not_torch_210()

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    out_dir = pathlib.Path(cfg.get("output_dir", "outputs/finetune_disfa"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("finetune", str(out_dir / "train.log"))
    writer = create_tb_writer(str(out_dir / "tb"))

    smoke = args.smoke or bool(cfg["train"].get("smoke", False))

    train_loader = _build_loader(cfg, split="train", smoke=smoke)
    val_loader = _build_loader(cfg, split="val", smoke=smoke)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = _build_backbone(cfg["model"])

    if args.pretrained:
        pt = load_checkpoint(args.pretrained, map_location="cpu")
        missing, unexpected = backbone.load_state_dict(pt.get("model", pt), strict=False)
        logger.info(
            "Loaded pretrained backbone from %s (missing=%d, unexpected=%d)",
            args.pretrained,
            len(missing),
            len(unexpected),
        )

    model = VideoAUModel(
        backbone=backbone,
        num_aus=len(cfg["data"]["au_list"]),
        out_frames=int(cfg["data"].get("clip_len", 32)),
    ).to(device)

    if bool(cfg["train"].get("freeze_backbone", False)):
        for p in model.backbone.parameters():
            p.requires_grad = False

    train_cfg = cfg["train"]
    amp_enabled = bool(train_cfg.get("amp", True))
    accum_steps = int(train_cfg.get("accum_steps", 4))
    epochs = int(train_cfg.get("epochs", 20))
    max_steps = int(train_cfg.get("max_steps", -1))
    smoke_steps = int(train_cfg.get("smoke_steps", 50))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )

    updates_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    total_updates = smoke_steps if smoke else (max_steps if max_steps > 0 else epochs * updates_per_epoch)
    scheduler = _build_scheduler(
        optimizer,
        warmup_steps=int(train_cfg.get("warmup_steps", 500)),
        total_steps=total_updates,
    )

    pos_weight = train_cfg.get("pos_weight")
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")

    ckpt_mgr = CheckpointManager(str(out_dir / "checkpoints"))
    start_epoch = 0
    global_update_step = 0

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler.is_enabled() and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_update_step = int(ckpt.get("global_update_step", 0))

    log_interval = int(train_cfg.get("log_interval", 10))
    val_interval = int(train_cfg.get("val_interval", 200))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    done = False
    for epoch in range(start_epoch, epochs if not smoke else 999999):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)

        num_batches = len(train_loader)
        for step, batch in enumerate(pbar):
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with _autocast_context(device, amp_enabled):
                logits = model(video)
                loss = criterion(logits, labels) / accum_steps

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

                writer.add_scalar("train/loss", float(loss.item() * accum_steps), global_update_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_update_step)

                if global_update_step % log_interval == 0:
                    logger.info(
                        "epoch=%d update=%d loss=%.5f lr=%.3e",
                        epoch,
                        global_update_step,
                        loss.item() * accum_steps,
                        optimizer.param_groups[0]["lr"],
                    )

                if global_update_step % val_interval == 0 or global_update_step >= total_updates:
                    val_metrics = evaluate(
                        model=model,
                        loader=val_loader,
                        criterion=criterion,
                        device=device,
                        amp_enabled=amp_enabled,
                        max_steps=10 if smoke else None,
                    )

                    writer.add_scalar("val/loss", float(val_metrics["loss"]), global_update_step)
                    writer.add_scalar("val/mean_f1", float(val_metrics["mean_f1"]), global_update_step)
                    if val_metrics.get("mean_auc") is not None:
                        writer.add_scalar("val/mean_auc", float(val_metrics["mean_auc"]), global_update_step)

                    logger.info(
                        "val update=%d loss=%.5f mean_f1=%.4f mean_auc=%s",
                        global_update_step,
                        float(val_metrics["loss"]),
                        float(val_metrics["mean_f1"]),
                        "nan" if val_metrics.get("mean_auc") is None else f"{float(val_metrics['mean_auc']):.4f}",
                    )

                    state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                        "epoch": epoch,
                        "global_update_step": global_update_step,
                        "config": cfg,
                    }
                    ckpt_mgr.save_latest(state)
                    best_path = ckpt_mgr.save_best(state, metric=float(val_metrics["mean_f1"]), mode="max")
                    if best_path:
                        logger.info("New best checkpoint: %s (mean_f1=%.4f)", best_path, float(val_metrics["mean_f1"]))

                if global_update_step >= total_updates:
                    done = True
                    break

        if done:
            break

    logger.info("Fine-tuning complete. total_update_steps=%d", global_update_step)
    writer.close()


if __name__ == "__main__":
    main()
