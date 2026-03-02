from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class PatchGrid:
    t: int
    h: int
    w: int

    @property
    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.t, self.h, self.w)


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 192,
        patch_size: tuple[int, int, int] = (2, 16, 16),
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got shape={tuple(x.shape)}")
        return self.proj(x)

    def get_grid_size(self, x: torch.Tensor) -> PatchGrid:
        pt, ph, pw = self.patch_size
        _, _, t, h, w = x.shape
        return PatchGrid(t=t // pt, h=h // ph, w=w // pw)
