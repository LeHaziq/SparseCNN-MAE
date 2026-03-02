from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocess.face_align import FaceAligner

from .augment import VideoAugment, frames_to_clip, normalize_clip
from .video_reader import VideoClipReader


def _sorted_image_files(frame_dir: str) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in Path(frame_dir).iterdir() if p.suffix.lower() in exts]
    files.sort()
    return files


def _load_frames_from_dir(frame_dir: str, indices: Sequence[int]) -> torch.Tensor:
    files = _sorted_image_files(frame_dir)
    if len(files) == 0:
        raise RuntimeError(f"No image files found in frame_dir={frame_dir}")
    max_idx = len(files) - 1
    out = []
    for idx in indices:
        idx = int(max(0, min(idx, max_idx)))
        with Image.open(files[idx]).convert("RGB") as img:
            out.append(torch.from_numpy(np.array(img, dtype=np.uint8)))
    return torch.stack(out, dim=0)


def _parse_au_filename(path: Path) -> Optional[int]:
    m = re.search(r"au[_-]?(\d+)", path.stem.lower())
    if not m:
        m = re.search(r"(\d+)", path.stem)
    if not m:
        return None
    return int(m.group(1))


def _read_annotation_file(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()
            if len(parts) < 2:
                continue
            try:
                frame_idx = int(float(parts[0]))
                intensity = float(parts[1])
            except ValueError:
                continue
            rows.append((frame_idx, intensity))
    return rows


def _load_annotation_tensor(annotation_path: str, au_list: Sequence[int]) -> torch.Tensor:
    path = Path(annotation_path)
    au_list = [int(a) for a in au_list]

    if path.suffix.lower() == ".npy":
        import numpy as np

        arr = np.load(path)
        t = torch.from_numpy(arr).float()
        if t.dim() != 2:
            raise ValueError(f"Expected [num_frames, num_aus] in {path}")
        return t

    if path.suffix.lower() == ".npz":
        import numpy as np

        d = np.load(path)
        if "labels" in d:
            arr = d["labels"]
        else:
            arr = d[list(d.keys())[0]]
        t = torch.from_numpy(arr).float()
        if t.dim() != 2:
            raise ValueError(f"Expected [num_frames, num_aus] in {path}")
        return t

    if path.suffix.lower() in {".pt", ".pth"}:
        t = torch.load(path)
        if not isinstance(t, torch.Tensor) or t.dim() != 2:
            raise ValueError(f"Expected tensor [num_frames, num_aus] in {path}")
        return t.float()

    if path.is_dir():
        per_au: dict[int, dict[int, float]] = {au: {} for au in au_list}
        max_frame = 0
        for file in path.iterdir():
            if not file.is_file():
                continue
            au = _parse_au_filename(file)
            if au is None or au not in per_au:
                continue
            pairs = _read_annotation_file(file)
            for frame_idx, intensity in pairs:
                per_au[au][frame_idx] = intensity
                max_frame = max(max_frame, frame_idx)

        labels = torch.zeros((max_frame + 1, len(au_list)), dtype=torch.float32)
        for col, au in enumerate(au_list):
            for frame_idx, intensity in per_au[au].items():
                labels[frame_idx, col] = intensity
        return labels

    if path.suffix.lower() == ".csv":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if len(rows) == 0:
            return torch.zeros((0, len(au_list)), dtype=torch.float32)

        labels = torch.zeros((len(rows), len(au_list)), dtype=torch.float32)
        for i, row in enumerate(rows):
            for j, au in enumerate(au_list):
                candidates = [f"AU{au}", f"au{au}", str(au)]
                val = 0.0
                for key in candidates:
                    if key in row and row[key] != "":
                        val = float(row[key])
                        break
                labels[i, j] = val
        return labels

    raise ValueError(f"Unsupported annotation format: {annotation_path}")


class DISFADataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        au_list: Sequence[int],
        clip_len: int = 32,
        stride: int = 1,
        train: bool = True,
        label_threshold: float = 0.0,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        backend: str = "auto",
        align_mode: str = "off",
        align_cache_dir: Optional[str] = None,
        color_jitter: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.split = split
        self.au_list = [int(a) for a in au_list]
        self.clip_len = clip_len
        self.stride = stride
        self.train = train
        self.label_threshold = label_threshold
        self.mean = tuple(mean)
        self.std = tuple(std)

        self.rows = self._load_manifest(manifest_path, split)
        if len(self.rows) == 0:
            raise RuntimeError(f"No rows for split={split} in manifest={manifest_path}")

        self.reader = VideoClipReader(backend=backend)
        self.augment = VideoAugment(output_size=112, train=train, color_jitter=color_jitter)
        self.aligner = FaceAligner(mode=align_mode, cache_dir=align_cache_dir, output_size=112)
        self.label_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _load_manifest(path: str, split: str) -> list[dict[str, str]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = (row.get("split") or "train").lower()
                if row_split != split.lower():
                    continue
                has_video = row.get("video_path") and Path(row["video_path"]).exists()
                has_frames = row.get("frame_dir") and Path(row["frame_dir"]).exists()
                has_cache = row.get("cache_path") and Path(row["cache_path"]).exists()
                if not (has_video or has_frames or has_cache):
                    continue
                rows.append(row)
        return rows

    def _load_labels(self, annotation_path: str) -> torch.Tensor:
        if annotation_path not in self.label_cache:
            t = _load_annotation_tensor(annotation_path, self.au_list)
            self.label_cache[annotation_path] = t
        return self.label_cache[annotation_path]

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_indices(self, total_frames: int) -> tuple[int, torch.Tensor]:
        need = (self.clip_len - 1) * self.stride + 1
        max_start = max(0, total_frames - need)
        if self.train:
            start = torch.randint(low=0, high=max_start + 1, size=(1,)).item() if max_start > 0 else 0
        else:
            start = max_start // 2
        idx = start + torch.arange(self.clip_len, dtype=torch.long) * self.stride
        idx = torch.clamp(idx, 0, max(total_frames - 1, 0))
        return int(start), idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows[idx]
        annotation_path = row.get("annotation_path")
        if not annotation_path:
            raise ValueError("DISFA row must contain annotation_path")
        labels_full = self._load_labels(annotation_path)

        cache_path = row.get("cache_path")
        if cache_path and Path(cache_path).exists():
            cached = torch.load(cache_path)
            if isinstance(cached, dict) and "frames" in cached:
                frames = cached["frames"]
                start = int(cached.get("start_frame", 0))
                idxs = start + torch.arange(self.clip_len, dtype=torch.long) * self.stride
            else:
                frames = cached
                start = 0
                idxs = torch.arange(self.clip_len, dtype=torch.long)
            if frames.dtype != torch.uint8:
                frames = (frames.clamp(0, 1) * 255.0).to(torch.uint8)
            if frames.dim() == 4 and frames.shape[-1] != 3:
                # [C,T,H,W] -> [T,H,W,C]
                frames = frames.permute(1, 2, 3, 0).contiguous()
        else:
            if row.get("video_path") and Path(row["video_path"]).exists():
                n_total = int(row.get("num_frames", 0)) or self.reader.get_num_frames(row["video_path"])
                start, idxs = self._sample_indices(n_total)
                sample = self.reader.read_clip(
                    path=row["video_path"],
                    num_frames=self.clip_len,
                    stride=self.stride,
                    start_frame=start,
                    random_start=False,
                )
                frames = sample.frames
                start = sample.start_frame
                idxs = sample.indices
            elif row.get("frame_dir") and Path(row["frame_dir"]).exists():
                files = _sorted_image_files(row["frame_dir"])
                n_total = len(files)
                start, idxs = self._sample_indices(n_total)
                frames = _load_frames_from_dir(row["frame_dir"], idxs.tolist())
            else:
                raise RuntimeError("Row has neither video_path nor frame_dir nor cache_path")

            if self.aligner.enabled:
                ref = row.get("video_path") or row.get("frame_dir")
                frames = self.aligner.align_frames(
                    frames,
                    video_path=ref,
                    start_frame=start,
                    stride=self.stride,
                    clip_len=self.clip_len,
                )

        if labels_full.shape[0] == 0:
            labels = torch.zeros((self.clip_len, len(self.au_list)), dtype=torch.float32)
        else:
            idxs = torch.clamp(idxs, 0, labels_full.shape[0] - 1)
            labels = labels_full[idxs]  # [T, num_aus]
        labels = (labels > self.label_threshold).float()
        labels = labels.transpose(0, 1).contiguous()  # [num_aus, T]

        clip = frames_to_clip(frames)
        clip = self.augment(clip)
        clip = normalize_clip(clip, self.mean, self.std)

        subject_id = row.get("subject_id", "unknown")
        return {
            "video": clip,
            "labels": labels,
            "subject_id": subject_id,
            "start_frame": int(idxs[0].item()) if len(idxs) else 0,
        }
