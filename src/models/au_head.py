from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mae import VideoMAE


class AUHead(nn.Module):
    def __init__(self, in_channels: int, num_aus: int, out_frames: int = 32) -> None:
        super().__init__()
        self.out_frames = out_frames
        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm3d(in_channels)
        self.temporal = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.temporal_bn = nn.BatchNorm1d(in_channels)
        self.classifier = nn.Conv1d(in_channels, num_aus, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,C,Tg,Hg,Wg]
        x = self.proj(feat)
        x = self.proj_bn(x)
        x = self.act(x)
        x = x.mean(dim=(-1, -2))  # [B,C,Tg]
        x = F.interpolate(x, size=self.out_frames, mode="linear", align_corners=False)
        x = self.temporal(x)
        x = self.temporal_bn(x)
        x = self.act(x)
        logits = self.classifier(x)  # [B,num_aus,T]
        return logits


class VideoAUModel(nn.Module):
    def __init__(self, backbone: VideoMAE, num_aus: int, out_frames: int = 32) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = AUHead(in_channels=backbone.encoder.out_channels, num_aus=num_aus, out_frames=out_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.backbone.encode_visible(x)
        logits = self.head(enc.latent_dense)
        return logits
