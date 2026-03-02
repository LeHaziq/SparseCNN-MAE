from __future__ import annotations

import torch

from src.models.mae import VideoMAE


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoMAE(
        in_channels=3,
        input_size=(32, 112, 112),
        patch_size=(2, 16, 16),
        embed_dim=192,
        mask_ratio=0.9,
        encoder_channels=(192, 256),
        encoder_blocks=(1, 1),
        downsample_stages=(1,),
        decoder_channels=192,
        prefer_spconv=True,
        loss_type="mse",
        visible_loss_weight=0.0,
    ).to(device)
    model.train()

    x = torch.randn(2, 3, 32, 112, 112, device=device)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    opt.zero_grad(set_to_none=True)
    with _autocast_context(device):
        out = model(x)
        loss = out.loss

    assert out.recon.shape == x.shape, f"recon shape mismatch: {out.recon.shape} vs {x.shape}"
    assert out.mask.shape == (2, 16, 7, 7), f"mask shape mismatch: {out.mask.shape}"
    assert torch.isfinite(loss).item(), "Loss is not finite"

    # Verify masked-loss path computes expected values when per-pixel error is constant.
    with torch.no_grad():
        pred = torch.zeros_like(x)
        target = torch.ones_like(x)
        patch_mask = torch.zeros((2, 16, 7, 7), dtype=torch.bool, device=device)
        patch_mask[:, :8] = True
        total, masked, visible = model._recon_loss(pred, target, patch_mask)
        # For MSE with constant difference=1, masked and visible loss should both be ~1.
        assert abs(masked.item() - 1.0) < 1e-4, f"masked loss expected ~1, got {masked.item()}"
        assert abs(visible.item() - 1.0) < 1e-4, f"visible loss expected ~1, got {visible.item()}"
        assert abs(total.item() - 1.0) < 1e-4, f"total loss expected ~1, got {total.item()}"

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()

    print("Smoke test passed")
    print(
        {
            "loss": float(out.loss.item()),
            "masked_loss": float(out.masked_loss.item()),
            "visible_loss": float(out.visible_loss.item()),
            "used_spconv": out.used_spconv,
        }
    )


if __name__ == "__main__":
    main()
