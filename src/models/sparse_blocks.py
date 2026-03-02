from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv

    HAS_SPCONV = True
except Exception:
    spconv = None
    HAS_SPCONV = False


def is_spconv_available() -> bool:
    return HAS_SPCONV


def _replace_feature(x, feat: torch.Tensor):
    return x.replace_feature(feat)


@dataclass
class SparseBatch:
    features: torch.Tensor
    indices: torch.Tensor
    spatial_shape: list[int]
    batch_size: int


def build_sparse_batch(tokens: torch.Tensor, visible_mask: torch.Tensor) -> SparseBatch:
    """
    Convert dense patch tokens to sparse COO-style representation for spconv.

    tokens: [B,C,Tg,Hg,Wg]
    visible_mask: [B,Tg,Hg,Wg] (True means visible)
    """
    if tokens.dim() != 5:
        raise ValueError(f"Expected tokens [B,C,T,H,W], got {tokens.shape}")
    if visible_mask.dim() != 4:
        raise ValueError(f"Expected visible_mask [B,T,H,W], got {visible_mask.shape}")

    b, c, tg, hg, wg = tokens.shape
    if visible_mask.shape != (b, tg, hg, wg):
        raise ValueError(
            f"visible_mask shape mismatch, expected {(b, tg, hg, wg)} got {tuple(visible_mask.shape)}"
        )

    indices = visible_mask.nonzero(as_tuple=False).to(torch.int32)  # [N,4] -> [b,z,y,x]
    # Gather feature vectors in the same order as nonzero coordinates.
    dense = tokens.permute(0, 2, 3, 4, 1).contiguous()  # [B,T,H,W,C]
    features = dense[visible_mask].contiguous()  # [N,C]
    return SparseBatch(features=features, indices=indices, spatial_shape=[tg, hg, wg], batch_size=b)


class DenseResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.act(x)
        return x


class DenseDownsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DenseStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if downsample or in_channels != out_channels:
            if downsample:
                layers.append(DenseDownsample3D(in_channels, out_channels))
            else:
                layers.extend(
                    [
                        nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
        for _ in range(num_blocks):
            layers.append(DenseResidualBlock3D(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if HAS_SPCONV:

    class SparseResidualBlock3D(nn.Module):
        def __init__(self, channels: int, indice_key: str) -> None:
            super().__init__()
            self.conv1 = spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=f"{indice_key}_subm",
            )
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv2 = spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=f"{indice_key}_subm",
            )
            self.bn2 = nn.BatchNorm1d(channels)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = _replace_feature(out, self.bn1(out.features))
            out = _replace_feature(out, self.act(out.features))
            out = self.conv2(out)
            out = _replace_feature(out, self.bn2(out.features))
            out = _replace_feature(out, out.features + identity.features)
            out = _replace_feature(out, self.act(out.features))
            return out


    class SparseDownsample3D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, indice_key: str) -> None:
            super().__init__()
            self.conv = spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
                indice_key=f"{indice_key}_down",
            )
            self.bn = nn.BatchNorm1d(out_channels)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = _replace_feature(x, self.bn(x.features))
            x = _replace_feature(x, self.act(x.features))
            return x


    class SparseStage(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            downsample: bool,
            stage_id: int,
        ) -> None:
            super().__init__()
            self.proj = None
            if downsample:
                self.proj = SparseDownsample3D(in_channels, out_channels, indice_key=f"stage{stage_id}")
            elif in_channels != out_channels:
                self.proj = spconv.SparseSequential(
                    spconv.SubMConv3d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        bias=False,
                        indice_key=f"stage{stage_id}_proj",
                    )
                )
                self.proj_bn = nn.BatchNorm1d(out_channels)
                self.proj_act = nn.ReLU(inplace=True)
            else:
                self.proj_bn = None
                self.proj_act = None

            self.blocks = nn.ModuleList(
                [SparseResidualBlock3D(out_channels, indice_key=f"stage{stage_id}_blk{i}") for i in range(num_blocks)]
            )

        def forward(self, x):
            if self.proj is not None:
                x = self.proj(x)
                if hasattr(self, "proj_bn") and self.proj_bn is not None:
                    x = _replace_feature(x, self.proj_bn(x.features))
                    x = _replace_feature(x, self.proj_act(x.features))
            for blk in self.blocks:
                x = blk(x)
            return x


def warn_dense_fallback_once() -> None:
    if not HAS_SPCONV:
        warnings.warn(
            "spconv v2 is not available. Falling back to dense Conv3d encoder; sparse speedups are disabled.",
            stacklevel=2,
        )


def validate_stage_config(channels: Sequence[int], blocks: Sequence[int]) -> None:
    if len(channels) != len(blocks):
        raise ValueError("channels and blocks must have the same length")
    if len(channels) == 0:
        raise ValueError("At least one stage is required")
