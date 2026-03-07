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

from src.data.voxceleb2 import VoxCeleb2MAEDataset
from src.models.mae import VideoMAE
from src.utils.checkpoint import CheckpointManager, load_checkpoint
from src.utils.config import load_config
from src.utils.distributed import (
    DistributedContext,
    cleanup_distributed,
    count_unsynced_batchnorm_layers,
    init_distributed_mode,
    is_distributed_ready,
    make_eval_sampler,
    make_train_sampler,
    unwrap_model,
)
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


def _build_loader(
    cfg: dict[str, Any],
    split: str,
    dist_ctx: DistributedContext,
) -> tuple[DataLoader, torch.utils.data.Sampler | None]:
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
    sampler = (
        make_train_sampler(ds, dist_ctx=dist_ctx, seed=int(cfg.get("seed", 42)), drop_last=True)
        if train
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
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        worker_init_fn=worker_init_fn,
        **loader_kwargs,
    )
    return loader, sampler


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
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    dist_ctx: DistributedContext,
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
        metrics = torch.zeros(4, device=device)
    else:
        metrics = torch.tensor([total_loss, total_masked, total_visible, float(n)], device=device)

    if dist_ctx.is_distributed and is_distributed_ready():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    if int(metrics[3].item()) == 0:
        return {"loss": float("nan"), "masked_loss": float("nan"), "visible_loss": float("nan")}

    denom = float(metrics[3].item())
    return {
        "loss": float(metrics[0].item() / denom),
        "masked_loss": float(metrics[1].item() / denom),
        "visible_loss": float(metrics[2].item() / denom),
    }


def main() -> None:
    writer = None
    dist_ctx = init_distributed_mode()
    parser = argparse.ArgumentParser(description="Pretrain fully-convolutional Video-MAE on VoxCeleb2")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny 50-step smoke run")
    args, overrides = parser.parse_known_args()

    try:
        cfg = load_config(args.config, overrides)
        warn_if_not_torch_210()

        seed = int(cfg.get("seed", 42))
        set_seed(seed + dist_ctx.rank)

        out_dir = pathlib.Path(cfg.get("output_dir", "outputs/pretrain_voxceleb2"))
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = get_logger(
            "pretrain",
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

        train_loader, train_sampler = _build_loader(cfg, split="train", dist_ctx=dist_ctx)
        val_loader, _ = _build_loader(cfg, split="val", dist_ctx=dist_ctx)

        model = _build_model(cfg).to(device)
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
        epochs = int(train_cfg.get("epochs", 30))
        max_steps = int(train_cfg.get("max_steps", -1))
        smoke_steps = int(train_cfg.get("smoke_steps", 50))
        smoke = args.smoke or bool(train_cfg.get("smoke", False))
        lr, blr, eff_batch_size, world_size = _resolve_learning_rate(
            train_cfg=train_cfg,
            batch_size=batch_size,
            accum_steps=accum_steps,
            default_blr=1.5e-4,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
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
            default_warmup_steps=1000,
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
        scaler = _make_grad_scaler(device=device, amp_enabled=amp_enabled)

        start_epoch = 0
        global_update_step = 0
        ckpt_mgr = CheckpointManager(str(out_dir / "checkpoints"))

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
            logger.info("Resumed from %s at epoch=%d update_step=%d", args.resume, start_epoch, global_update_step)

        log_interval = int(train_cfg.get("log_interval", 10))
        val_interval = int(train_cfg.get("val_interval", 200))
        grad_clip = float(train_cfg.get("grad_clip", 1.0))

        model.train()
        done = False

        for epoch in range(start_epoch, epochs if not smoke else 999999):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            iterator = tqdm(train_loader, desc=f"epoch {epoch}", leave=False) if dist_ctx.is_main_process else train_loader
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(iterator):
                is_update_step = ((step + 1) % accum_steps == 0) or ((step + 1) == num_batches)
                sync_context = model.no_sync if dist_ctx.is_distributed and not is_update_step else nullcontext

                video = batch["video"].to(device, non_blocking=True)
                with sync_context():
                    with _autocast_context(device, amp_enabled):
                        out = model(video)
                        loss = out.loss / accum_steps

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
                            dist_ctx=dist_ctx,
                            max_steps=10 if smoke else None,
                        )
                        writer.add_scalar("val/loss", metrics["loss"], global_update_step)
                        writer.add_scalar("val/masked_loss", metrics["masked_loss"], global_update_step)
                        writer.add_scalar("val/visible_loss", metrics["visible_loss"], global_update_step)
                        model.train()

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
                            best_path = ckpt_mgr.save_best(state, metric=float(metrics["loss"]), mode="min")
                            if best_path:
                                logger.info("New best checkpoint: %s (val_loss=%.5f)", best_path, metrics["loss"])

                    if global_update_step >= total_updates:
                        done = True
                        break

            if done:
                break

        logger.info("Training complete. total_update_steps=%d", global_update_step)
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
