from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
import math
import numpy as np
import pathlib
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.disfa import DISFADataset
from src.models.au_head import VideoAUModel
from src.models.mae import VideoMAE
from src.utils.checkpoint import CheckpointManager, load_checkpoint
from src.utils.config import load_config
from src.utils.distributed import (
    DistributedContext,
    cleanup_distributed,
    count_unsynced_batchnorm_layers,
    init_distributed_mode,
    make_eval_sampler,
    make_train_sampler,
    unwrap_model,
)
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


def _make_grad_scaler(device: torch.device, amp_enabled: bool):
    enabled = amp_enabled and device.type == "cuda"
    # torch.cuda.amp.GradScaler is deprecated in newer PyTorch versions.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _configure_cuda_speed_flags(train_cfg: dict[str, Any], device: torch.device, logger) -> None:
    if device.type != "cuda":
        return

    cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    allow_tf32_matmul = bool(train_cfg.get("allow_tf32_matmul", True))
    allow_tf32_cudnn = bool(train_cfg.get("allow_tf32_cudnn", True))
    matmul_precision = train_cfg.get("float32_matmul_precision", "high")

    torch.backends.cudnn.benchmark = cudnn_benchmark
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32_matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32_cudnn

    if matmul_precision is not None:
        try:
            torch.set_float32_matmul_precision(str(matmul_precision))
        except Exception as e:
            logger.warning("Failed to set float32 matmul precision (%s): %s", matmul_precision, e)

    logger.info(
        "CUDA speed flags: cudnn_benchmark=%s, allow_tf32_matmul=%s, allow_tf32_cudnn=%s, float32_matmul_precision=%s",
        cudnn_benchmark,
        allow_tf32_matmul,
        allow_tf32_cudnn,
        str(matmul_precision),
    )


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


def _build_loader(
    cfg: dict[str, Any],
    split: str,
    dist_ctx: DistributedContext,
    smoke: bool = False,
) -> tuple[DataLoader, torch.utils.data.Sampler | None]:
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
    sampler = (
        make_train_sampler(ds, dist_ctx=dist_ctx, seed=int(cfg.get("seed", 42)), drop_last=True)
        if split == "train"
        else make_eval_sampler(ds, dist_ctx=dist_ctx)
    )

    num_workers = int(cfg["train"].get("num_workers", 4))
    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["train"].get("persistent_workers", True))
        prefetch_factor = cfg["train"].get("prefetch_factor", 2)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=(split == "train") and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        worker_init_fn=worker_init_fn,
        **loader_kwargs,
    )
    return loader, sampler


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
):
    min_lr = min(max(0.0, float(min_lr)), float(base_lr))
    min_factor = 0.0 if base_lr <= 0 else (min_lr / base_lr)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _resolve_learning_rate(
    train_cfg: dict[str, Any],
    batch_size: int,
    accum_steps: int,
    default_blr: float,
) -> tuple[float, float, int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = int(torch.distributed.get_world_size())
    else:
        world_size = 1
    eff_batch_size = batch_size * accum_steps * world_size
    if train_cfg.get("lr") is not None:
        return float(train_cfg["lr"]), float(train_cfg.get("blr", default_blr)), eff_batch_size, world_size
    base_lr = float(train_cfg.get("blr", default_blr))
    lr = base_lr * eff_batch_size / 256.0
    return lr, base_lr, eff_batch_size, world_size


def _resolve_warmup_steps(
    train_cfg: dict[str, Any],
    updates_per_epoch: int,
    default_warmup_steps: int,
) -> int:
    warmup_steps_cfg = train_cfg.get("warmup_steps")
    if warmup_steps_cfg is not None:
        return max(0, int(warmup_steps_cfg))
    warmup_epochs_cfg = train_cfg.get("warmup_epochs")
    if warmup_epochs_cfg is not None:
        warmup_epochs = max(0.0, float(warmup_epochs_cfg))
        return int(round(warmup_epochs * updates_per_epoch))
    return max(0, int(default_warmup_steps))


def _maybe_convert_sync_batchnorm(model: torch.nn.Module, dist_ctx: DistributedContext, logger) -> torch.nn.Module:
    bn_layers = count_unsynced_batchnorm_layers(model)
    if not dist_ctx.is_distributed or bn_layers == 0:
        return model
    if dist_ctx.device.type != "cuda":
        logger.warning(
            "Distributed training detected %d BatchNorm layer(s), but SyncBatchNorm conversion is only enabled on CUDA.",
            bn_layers,
        )
        return model

    logger.info("Converting %d BatchNorm layer(s) to SyncBatchNorm for DDP", bn_layers)
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


def _require_nonempty_train_loader(
    loader: DataLoader,
    *,
    batch_size: int,
    dist_ctx: DistributedContext,
    smoke: bool,
) -> int:
    num_batches = len(loader)
    if num_batches > 0:
        return num_batches

    sampler = getattr(loader, "sampler", None)
    shard_samples = len(sampler) if sampler is not None and hasattr(sampler, "__len__") else len(loader.dataset)
    scope = "per-rank training shard" if dist_ctx.is_distributed else "training dataset"
    smoke_hint = " Disable `--smoke`," if smoke else ""
    raise RuntimeError(
        f"Empty train loader on rank {dist_ctx.rank}: the {scope} has {shard_samples} sample(s), "
        f"so DataLoader(batch_size={batch_size}, drop_last=True) yields zero batches.{smoke_hint} "
        "reduce `world_size` or `batch_size`, or provide more training samples."
    )


@torch.no_grad()
def evaluate(
    model: VideoAUModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    dist_ctx: DistributedContext,
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

    logits = torch.cat(meter.logits, dim=0) if meter.logits else torch.empty((0, 0), dtype=torch.float32)
    targets = torch.cat(meter.targets, dim=0) if meter.targets else torch.empty((0, 0), dtype=torch.float32)
    if dist_ctx.is_distributed:
        gathered: list[dict[str, torch.Tensor] | None] = [None for _ in range(dist_ctx.world_size)]
        dist.all_gather_object(gathered, {"logits": logits, "targets": targets})
        gathered_logits = [item["logits"] for item in gathered if item is not None and item["logits"].numel() > 0]
        gathered_targets = [item["targets"] for item in gathered if item is not None and item["targets"].numel() > 0]
        logits = torch.cat(gathered_logits, dim=0) if gathered_logits else logits
        targets = torch.cat(gathered_targets, dim=0) if gathered_targets else targets
        loss_counts = torch.tensor([total_loss, float(n)], dtype=torch.float64, device=device)
        dist.all_reduce(loss_counts, op=dist.ReduceOp.SUM)
        total_loss = float(loss_counts[0].item())
        n = int(loss_counts[1].item())

    meter = AUDetectionMeter()
    if logits.numel() > 0:
        meter.update(logits, targets)
    score = meter.compute()
    score["loss"] = total_loss / max(1, n)
    return score


def main() -> None:
    writer = None
    dist_ctx = init_distributed_mode()
    parser = argparse.ArgumentParser(description="Fine-tune AU detector on DISFA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained MAE checkpoint")
    parser.add_argument("--smoke", action="store_true")
    args, overrides = parser.parse_known_args()

    try:
        cfg = load_config(args.config, overrides)
        warn_if_not_torch_210()

        seed = int(cfg.get("seed", 42))
        set_seed(seed + dist_ctx.rank)

        out_dir = pathlib.Path(cfg.get("output_dir", "outputs/finetune_disfa"))
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = get_logger(
            "finetune",
            str(out_dir / "train.log") if dist_ctx.is_main_process else None,
            enabled=dist_ctx.is_main_process,
        )
        writer = create_tb_writer(str(out_dir / "tb") if dist_ctx.is_main_process else None)

        train_cfg = cfg["train"]
        device = dist_ctx.device
        _configure_cuda_speed_flags(train_cfg=train_cfg, device=device, logger=logger)
        if dist_ctx.is_main_process and dist_ctx.is_distributed:
            logger.info(
                "Initialized DDP with world_size=%d rank=%d local_rank=%d",
                dist_ctx.world_size,
                dist_ctx.rank,
                dist_ctx.local_rank,
            )

        smoke = args.smoke or bool(cfg["train"].get("smoke", False))

        train_loader, train_sampler = _build_loader(cfg, split="train", dist_ctx=dist_ctx, smoke=smoke)
        val_loader, _ = _build_loader(cfg, split="val", dist_ctx=dist_ctx, smoke=smoke)

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

        model = _maybe_convert_sync_batchnorm(model, dist_ctx=dist_ctx, logger=logger)
        if dist_ctx.is_distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
            )

        batch_size = int(train_cfg.get("batch_size", 2))
        amp_enabled = bool(train_cfg.get("amp", True))
        accum_steps = int(train_cfg.get("accum_steps", 4))
        epochs = int(train_cfg.get("epochs", 20))
        max_steps = int(train_cfg.get("max_steps", -1))
        smoke_steps = int(train_cfg.get("smoke_steps", 50))
        lr, blr, eff_batch_size, world_size = _resolve_learning_rate(
            train_cfg=train_cfg,
            batch_size=batch_size,
            accum_steps=accum_steps,
            default_blr=5e-4,
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        )
        logger.info(
            "LR setup: lr=%.3e, blr=%.3e, eff_batch=%d (batch=%d, accum=%d, world_size=%d)",
            lr,
            blr,
            eff_batch_size,
            batch_size,
            accum_steps,
            world_size,
        )

        num_batches = _require_nonempty_train_loader(
            train_loader,
            batch_size=batch_size,
            dist_ctx=dist_ctx,
            smoke=smoke,
        )
        updates_per_epoch = math.ceil(num_batches / accum_steps)
        total_updates = smoke_steps if smoke else (max_steps if max_steps > 0 else epochs * updates_per_epoch)
        warmup_steps = _resolve_warmup_steps(
            train_cfg=train_cfg,
            updates_per_epoch=updates_per_epoch,
            default_warmup_steps=500,
        )
        min_lr = float(train_cfg.get("min_lr", 0.0))
        scheduler = _build_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_updates,
            base_lr=lr,
            min_lr=min_lr,
        )
        logger.info(
            "Scheduler setup: warmup_steps=%d, min_lr=%.3e",
            warmup_steps,
            min_lr,
        )

        pos_weight = train_cfg.get("pos_weight")
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        scaler = _make_grad_scaler(device=device, amp_enabled=amp_enabled)

        ckpt_mgr = CheckpointManager(str(out_dir / "checkpoints"))
        start_epoch = 0
        global_update_step = 0

        if args.resume:
            ckpt = load_checkpoint(args.resume, map_location=device)
            unwrap_model(model).load_state_dict(ckpt["model"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and scaler.is_enabled() and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_update_step = int(ckpt.get("global_update_step", 0))
            if "best_metric" in ckpt:
                ckpt_mgr.load_best_metric(ckpt["best_metric"])
            if not dist_ctx.is_distributed:
                if "rng_state" in ckpt:
                    torch.set_rng_state(ckpt["rng_state"])
                if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
                if "numpy_rng_state" in ckpt:
                    np.random.set_state(ckpt["numpy_rng_state"])
                if "python_rng_state" in ckpt:
                    random.setstate(ckpt["python_rng_state"])

        log_interval = int(train_cfg.get("log_interval", 10))
        val_interval = int(train_cfg.get("val_interval", 200))
        grad_clip = float(train_cfg.get("grad_clip", 1.0))

        done = False
        for epoch in range(start_epoch, epochs if not smoke else 999999):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            iterator = tqdm(train_loader, desc=f"epoch {epoch}", leave=False) if dist_ctx.is_main_process else train_loader

            for step, batch in enumerate(iterator):
                is_update_step = ((step + 1) % accum_steps == 0) or ((step + 1) == num_batches)
                sync_context = model.no_sync if dist_ctx.is_distributed and not is_update_step else nullcontext

                video = batch["video"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with sync_context():
                    with _autocast_context(device, amp_enabled):
                        logits = model(video)
                        loss = criterion(logits, labels) / accum_steps

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

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
                            dist_ctx=dist_ctx,
                            max_steps=10 if smoke else None,
                        )

                        writer.add_scalar("val/loss", float(val_metrics["loss"]), global_update_step)
                        writer.add_scalar("val/mean_f1", float(val_metrics["mean_f1"]), global_update_step)
                        if val_metrics.get("mean_auc") is not None:
                            writer.add_scalar("val/mean_auc", float(val_metrics["mean_auc"]), global_update_step)
                        model.train()

                        logger.info(
                            "val update=%d loss=%.5f mean_f1=%.4f mean_auc=%s",
                            global_update_step,
                            float(val_metrics["loss"]),
                            float(val_metrics["mean_f1"]),
                            "nan" if val_metrics.get("mean_auc") is None else f"{float(val_metrics['mean_auc']):.4f}",
                        )

                        if dist_ctx.is_main_process:
                            state = {
                                "model": unwrap_model(model).state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                                "epoch": epoch,
                                "global_update_step": global_update_step,
                                "best_metric": ckpt_mgr.best_metric,
                                "config": cfg,
                            }
                            if not dist_ctx.is_distributed:
                                state.update(
                                    {
                                        "rng_state": torch.get_rng_state(),
                                        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                                        "numpy_rng_state": np.random.get_state(),
                                        "python_rng_state": random.getstate(),
                                    }
                                )
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
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
