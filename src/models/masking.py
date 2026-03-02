from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MaskOutput:
    mask: torch.Tensor  # [B, Tg, Hg, Wg], True means masked
    visible: torch.Tensor  # [B, Tg, Hg, Wg], True means visible


class RandomPatchMasker3D:
    def __init__(self, mask_ratio: float = 0.9) -> None:
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")
        self.mask_ratio = float(mask_ratio)

    def __call__(
        self,
        batch_size: int,
        grid_size: tuple[int, int, int],
        device: torch.device,
        mask_ratio: Optional[float] = None,
    ) -> MaskOutput:
        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        if not (0.0 < ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")

        tg, hg, wg = grid_size
        total = tg * hg * wg
        # Keep at least one visible and one masked token for stable loss.
        num_visible = int(round(total * (1.0 - ratio)))
        num_visible = max(1, min(total - 1, num_visible))

        noise = torch.rand(batch_size, total, device=device)
        ids = torch.argsort(noise, dim=1)
        visible_flat = torch.zeros(batch_size, total, dtype=torch.bool, device=device)
        visible_flat.scatter_(1, ids[:, :num_visible], True)

        visible = visible_flat.view(batch_size, tg, hg, wg)
        mask = ~visible
        return MaskOutput(mask=mask, visible=visible)


def patch_mask_to_pixel_mask(
    patch_mask: torch.Tensor,
    patch_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
) -> torch.Tensor:
    """Expand patch mask [B,Tg,Hg,Wg] to pixel mask [B,1,T,H,W]."""
    pt, ph, pw = patch_size
    pixel_mask = patch_mask
    pixel_mask = pixel_mask.repeat_interleave(pt, dim=1)
    pixel_mask = pixel_mask.repeat_interleave(ph, dim=2)
    pixel_mask = pixel_mask.repeat_interleave(pw, dim=3)
    pixel_mask = pixel_mask[:, None, : output_size[0], : output_size[1], : output_size[2]]
    return pixel_mask
