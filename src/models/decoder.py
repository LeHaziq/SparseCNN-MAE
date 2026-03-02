from __future__ import annotations

import torch
import torch.nn as nn


class DenseVideoDecoder(nn.Module):
    """
    Sparse encoder output is densified before entering this decoder.
    Decoder is fully convolutional and reconstructs pixel clip directly.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 192,
        patch_size: tuple[int, int, int] = (2, 16, 16),
        encoder_downsample_spatial: bool = True,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.encoder_downsample_spatial = encoder_downsample_spatial

        if encoder_downsample_spatial:
            self.grid_upsample = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels,
                    hidden_channels,
                    kernel_size=(1, 3, 3),
                    stride=(1, 2, 2),
                    padding=(0, 1, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(hidden_channels),
                nn.GELU(),
            )
            dec_in = hidden_channels
        else:
            self.grid_upsample = nn.Identity()
            dec_in = in_channels

        self.refine = nn.Sequential(
            nn.Conv3d(dec_in, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
        )

        # Unpatchify to pixel space.
        self.unpatchify = nn.ConvTranspose3d(
            hidden_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grid_upsample(x)
        x = self.refine(x)
        x = self.unpatchify(x)
        return x
