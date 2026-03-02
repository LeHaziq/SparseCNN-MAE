from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import DenseVideoDecoder
from .encoder import EncoderOutput, SparseVideoEncoder
from .masking import RandomPatchMasker3D, patch_mask_to_pixel_mask
from .patch_embed import PatchEmbed3D


@dataclass
class MAEOutput:
    loss: torch.Tensor
    recon: torch.Tensor
    mask: torch.Tensor
    masked_loss: torch.Tensor
    visible_loss: torch.Tensor
    used_spconv: bool


class VideoMAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        input_size: tuple[int, int, int] = (32, 112, 112),
        patch_size: tuple[int, int, int] = (2, 16, 16),
        embed_dim: int = 192,
        mask_ratio: float = 0.9,
        encoder_channels: tuple[int, ...] = (192, 256),
        encoder_blocks: tuple[int, ...] = (2, 2),
        downsample_stages: tuple[int, ...] = (1,),
        decoder_channels: int = 192,
        prefer_spconv: bool = True,
        loss_type: str = "mse",
        visible_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.loss_type = loss_type
        self.visible_loss_weight = visible_loss_weight

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.masker = RandomPatchMasker3D(mask_ratio=mask_ratio)
        self.encoder = SparseVideoEncoder(
            in_channels=embed_dim,
            stage_channels=encoder_channels,
            stage_blocks=encoder_blocks,
            downsample_stages=downsample_stages,
            prefer_spconv=prefer_spconv,
        )
        self.decoder = DenseVideoDecoder(
            in_channels=self.encoder.out_channels,
            hidden_channels=decoder_channels,
            patch_size=patch_size,
            encoder_downsample_spatial=(len(downsample_stages) > 0),
            out_channels=in_channels,
        )

    def _recon_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"Prediction/target mismatch: {pred.shape} vs {target.shape}")

        if self.loss_type == "mse":
            per_pixel = F.mse_loss(pred, target, reduction="none")
        elif self.loss_type == "huber":
            per_pixel = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        pixel_mask = patch_mask_to_pixel_mask(patch_mask, self.patch_size, target.shape[2:])
        masked_weight = pixel_mask.to(per_pixel.dtype)
        visible_weight = (~pixel_mask).to(per_pixel.dtype)

        masked_loss = (per_pixel * masked_weight).sum() / masked_weight.sum().clamp_min(1.0)
        visible_loss = (per_pixel * visible_weight).sum() / visible_weight.sum().clamp_min(1.0)
        loss = masked_loss + self.visible_loss_weight * visible_loss
        return loss, masked_loss, visible_loss

    def encode_visible(self, x: torch.Tensor) -> EncoderOutput:
        patches = self.patch_embed(x)
        b, _, tg, hg, wg = patches.shape
        visible = torch.ones((b, tg, hg, wg), dtype=torch.bool, device=x.device)
        return self.encoder(patches, visible)

    def forward(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> MAEOutput:
        patches = self.patch_embed(x)
        b, _, tg, hg, wg = patches.shape

        mask_out = self.masker(
            batch_size=b,
            grid_size=(tg, hg, wg),
            device=x.device,
            mask_ratio=mask_ratio,
        )

        encoded = self.encoder(patches, mask_out.visible)
        recon = self.decoder(encoded.latent_dense)
        loss, masked_loss, visible_loss = self._recon_loss(recon, x, mask_out.mask)

        return MAEOutput(
            loss=loss,
            recon=recon,
            mask=mask_out.mask,
            masked_loss=masked_loss,
            visible_loss=visible_loss,
            used_spconv=encoded.used_spconv,
        )
