from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.models.mae import VideoMAE
from src.utils.distributed import cleanup_distributed, init_distributed_mode, unwrap_model


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_grad_scaler(device: torch.device):
    enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> None:
    dist_ctx = init_distributed_mode()
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    try:
        device = dist_ctx.device
        model = VideoMAE(
            in_channels=3,
            input_size=(8, 80, 80),
            patch_size=(2, 16, 16),
            embed_dim=32,
            mask_ratio=0.75,
            encoder_channels=(32, 48),
            encoder_blocks=(1, 1),
            downsample_stages=(1,),
            decoder_channels=32,
            prefer_spconv=False,
            loss_type="mse",
            visible_loss_weight=0.0,
        ).to(device)
        model.train()

        if dist_ctx.is_distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
            )

        scaler = _make_grad_scaler(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        optimizer.zero_grad(set_to_none=True)
        last_out = None
        micro_batches = 2
        for micro_idx in range(micro_batches):
            sync_context = model.no_sync if dist_ctx.is_distributed and micro_idx < (micro_batches - 1) else nullcontext
            x = torch.randn(1, 3, 8, 80, 80, device=device) + (0.1 * dist_ctx.rank)

            with sync_context():
                with _autocast_context(device):
                    out = model(x)
                    loss = out.loss / micro_batches

                assert out.recon.shape == x.shape, f"recon shape mismatch: {out.recon.shape} vs {x.shape}"
                assert out.mask.shape == (1, 4, 5, 5), f"mask shape mismatch: {out.mask.shape}"
                assert torch.isfinite(loss).item(), "Loss is not finite"

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            last_out = out

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        checksum = torch.tensor(
            float(sum(p.detach().float().sum().item() for p in unwrap_model(model).parameters())),
            device=device,
        )
        if dist_ctx.is_distributed:
            checksum_min = checksum.clone()
            checksum_max = checksum.clone()
            dist.all_reduce(checksum_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(checksum_max, op=dist.ReduceOp.MAX)
            assert torch.allclose(checksum_min, checksum_max, atol=1e-5, rtol=1e-5), "Parameter sync mismatch across ranks"

        loss_tensor = torch.tensor(float(last_out.loss.item()), device=device)
        if dist_ctx.is_distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= dist_ctx.world_size

        if dist_ctx.is_main_process:
            print("Distributed smoke test passed")
            print(
                {
                    "world_size": dist_ctx.world_size,
                    "rank": dist_ctx.rank,
                    "loss_mean": float(loss_tensor.item()),
                    "masked_loss": float(last_out.masked_loss.item()),
                    "visible_loss": float(last_out.visible_loss.item()),
                    "used_spconv": last_out.used_spconv,
                    "param_checksum": float(checksum.item()),
                }
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
