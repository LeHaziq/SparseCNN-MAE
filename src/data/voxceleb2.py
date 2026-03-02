from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from preprocess.face_align import FaceAligner

from .augment import VideoAugment, frames_to_clip, normalize_clip
from .video_reader import VideoClipReader


@dataclass
class VoxCeleb2Sample:
    video: torch.Tensor
    path: str
    start_frame: int


class VoxCeleb2MAEDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        clip_len: int = 32,
        stride: int = 2,
        train: bool = True,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        backend: str = "auto",
        align_mode: str = "off",
        align_cache_dir: Optional[str] = None,
        color_jitter: bool = True,
    ) -> None:
        self.manifest_path = str(manifest_path)
        self.clip_len = clip_len
        self.stride = stride
        self.train = train
        self.mean = tuple(mean)
        self.std = tuple(std)

        self.rows = self._load_manifest(self.manifest_path)
        if len(self.rows) == 0:
            raise RuntimeError(f"No videos found in manifest: {self.manifest_path}")

        self.reader = VideoClipReader(backend=backend)
        self.augment = VideoAugment(output_size=112, train=train, color_jitter=color_jitter)
        self.aligner = FaceAligner(mode=align_mode, cache_dir=align_cache_dir, output_size=112)

    @staticmethod
    def _load_manifest(path: str) -> list[dict[str, str]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row.get("video_path") or row.get("path")
                if not video_path:
                    continue
                if not Path(video_path).exists():
                    continue
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[idx]
        path = row.get("video_path") or row.get("path")

        sample = self.reader.read_clip(
            path=path,
            num_frames=self.clip_len,
            stride=self.stride,
            start_frame=None,
            random_start=self.train,
        )
        frames = sample.frames

        if self.aligner.enabled:
            frames = self.aligner.align_frames(
                frames,
                video_path=path,
                start_frame=sample.start_frame,
                stride=self.stride,
                clip_len=self.clip_len,
            )

        clip = frames_to_clip(frames)
        clip = self.augment(clip)
        clip = normalize_clip(clip, self.mean, self.std)

        out = VoxCeleb2Sample(video=clip, path=path, start_frame=sample.start_frame)
        return {"video": out.video, "path": out.path, "start_frame": out.start_frame}
