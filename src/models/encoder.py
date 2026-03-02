from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .sparse_blocks import (
    HAS_SPCONV,
    DenseStage,
    SparseStage,
    build_sparse_batch,
    validate_stage_config,
    warn_dense_fallback_once,
)

if HAS_SPCONV:
    import spconv.pytorch as spconv


@dataclass
class EncoderOutput:
    latent_dense: torch.Tensor
    used_spconv: bool


class SparseVideoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_channels: Sequence[int] = (192, 256),
        stage_blocks: Sequence[int] = (2, 2),
        downsample_stages: Sequence[int] = (1,),
        prefer_spconv: bool = True,
    ) -> None:
        super().__init__()
        validate_stage_config(stage_channels, stage_blocks)
        self.prefer_spconv = prefer_spconv
        self.use_spconv = prefer_spconv and HAS_SPCONV
        if prefer_spconv and not HAS_SPCONV:
            warn_dense_fallback_once()

        self.stage_channels = list(stage_channels)
        self.downsample_stages = set(downsample_stages)
        self.out_channels = self.stage_channels[-1]

        if self.use_spconv:
            self.sparse_stages = nn.ModuleList()
            prev_c = in_channels
            for i, (c, nblk) in enumerate(zip(stage_channels, stage_blocks)):
                self.sparse_stages.append(
                    SparseStage(
                        in_channels=prev_c,
                        out_channels=c,
                        num_blocks=nblk,
                        downsample=i in self.downsample_stages,
                        stage_id=i,
                    )
                )
                prev_c = c
            self.dense_stages = None
        else:
            self.dense_stages = nn.ModuleList()
            prev_c = in_channels
            for i, (c, nblk) in enumerate(zip(stage_channels, stage_blocks)):
                self.dense_stages.append(
                    DenseStage(
                        in_channels=prev_c,
                        out_channels=c,
                        num_blocks=nblk,
                        downsample=i in self.downsample_stages,
                    )
                )
                prev_c = c
            self.sparse_stages = None

    def forward(self, x: torch.Tensor, visible_mask: torch.Tensor) -> EncoderOutput:
        """
        x: [B,C,Tg,Hg,Wg]
        visible_mask: [B,Tg,Hg,Wg]
        """
        if self.use_spconv:
            sparse_batch = build_sparse_batch(x, visible_mask)
            if sparse_batch.features.numel() == 0:
                raise RuntimeError("No visible tokens available for sparse encoder")

            x_sparse = spconv.SparseConvTensor(
                features=sparse_batch.features,
                indices=sparse_batch.indices,
                spatial_shape=sparse_batch.spatial_shape,
                batch_size=sparse_batch.batch_size,
            )

            for stage in self.sparse_stages:
                x_sparse = stage(x_sparse)

            dense = x_sparse.dense()
            return EncoderOutput(latent_dense=dense, used_spconv=True)

        # Dense fallback: masked tokens are zeroed before the convolutions.
        x_dense = x * visible_mask.unsqueeze(1).to(dtype=x.dtype)
        for stage in self.dense_stages:
            x_dense = stage(x_dense)
        return EncoderOutput(latent_dense=x_dense, used_spconv=False)
