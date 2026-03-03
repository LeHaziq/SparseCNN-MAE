from __future__ import annotations

import random
from typing import Sequence

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def frames_to_clip(frames: torch.Tensor) -> torch.Tensor:
    """Convert [T,H,W,C] or [T,C,H,W] uint8 to [C,T,H,W] float32 in [0,1]."""
    if frames.dim() != 4:
        raise ValueError(f"Expected 4D frames, got {frames.shape}")
    if frames.shape[-1] in (1, 3):
        thwc = frames
    elif frames.shape[1] in (1, 3):
        thwc = frames.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(
            "Unable to infer channel dimension for frames. "
            f"Expected [T,H,W,C] or [T,C,H,W], got {frames.shape}"
        )
    clip = thwc.permute(3, 0, 1, 2).contiguous().float() / 255.0
    return clip


def normalize_clip(
    clip: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=clip.dtype, device=clip.device).view(3, 1, 1, 1)
    std_t = torch.tensor(std, dtype=clip.dtype, device=clip.device).view(3, 1, 1, 1)
    return (clip - mean_t) / std_t


class VideoAugment:
    def __init__(
        self,
        output_size: int = 112,
        train: bool = True,
        color_jitter: bool = True,
        min_scale: float = 0.7,
        max_scale: float = 1.0,
        hflip_prob: float = 0.5,
    ) -> None:
        self.output_size = output_size
        self.train = train
        self.color_jitter = color_jitter
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hflip_prob = hflip_prob
        self._jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)

    def _apply_consistent_color_jitter(self, clip_tchw: torch.Tensor) -> torch.Tensor:
        # Apply identical jitter to all frames.
        if not self.color_jitter:
            return clip_tchw
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            self._jitter.brightness,
            self._jitter.contrast,
            self._jitter.saturation,
            self._jitter.hue,
        )
        out = clip_tchw
        for fn_id in fn_idx:
            if fn_id == 0 and b is not None:
                out = TF.adjust_brightness(out, b)
            elif fn_id == 1 and c is not None:
                out = TF.adjust_contrast(out, c)
            elif fn_id == 2 and s is not None:
                out = TF.adjust_saturation(out, s)
            elif fn_id == 3 and h is not None:
                out = TF.adjust_hue(out, h)
        return out

    def __call__(self, clip_cthw: torch.Tensor) -> torch.Tensor:
        # Convert to [T,C,H,W] for torchvision frame-wise ops.
        clip = clip_cthw.permute(1, 0, 2, 3).contiguous()
        t, c, h, w = clip.shape

        if self.train:
            i, j, th, tw = T.RandomResizedCrop.get_params(
                clip[0],
                scale=(self.min_scale, self.max_scale),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            )
            clip = TF.resized_crop(
                clip,
                top=i,
                left=j,
                height=th,
                width=tw,
                size=[self.output_size, self.output_size],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            if random.random() < self.hflip_prob:
                clip = TF.hflip(clip)
            clip = self._apply_consistent_color_jitter(clip)
        else:
            # Resize shortest side to output_size then center-crop.
            if h < w:
                new_h, new_w = self.output_size, int(round(w * self.output_size / h))
            else:
                new_h, new_w = int(round(h * self.output_size / w)), self.output_size
            clip = TF.resize(
                clip,
                [new_h, new_w],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            clip = TF.center_crop(clip, [self.output_size, self.output_size])

        clip = clip.clamp(0.0, 1.0)
        clip = clip.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
        return clip
