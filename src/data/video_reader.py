from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch


@dataclass
class ClipSample:
    frames: torch.Tensor  # [T,H,W,C], uint8
    start_frame: int
    indices: torch.Tensor


class _TorchvisionBackend:
    name = "torchvision.io.VideoReader"

    def __init__(self) -> None:
        import torchvision
        from torchvision.io import VideoReader, read_video_timestamps

        self._torchvision = torchvision
        self._VideoReader = VideoReader
        self._read_video_timestamps = read_video_timestamps

    def get_num_frames(self, path: str) -> int:
        pts, _ = self._read_video_timestamps(path, pts_unit="sec")
        return len(pts)

    def read_frames(self, path: str, indices: Sequence[int]) -> torch.Tensor:
        if len(indices) == 0:
            return torch.empty((0, 0, 0, 3), dtype=torch.uint8)

        wanted = sorted(int(i) for i in indices)
        last = wanted[-1]
        ptr = 0
        out = []

        vr = self._VideoReader(path, "video")
        for i, frame in enumerate(vr):
            if i > last:
                break
            if i == wanted[ptr]:
                out.append(frame["data"])
                ptr += 1
                if ptr >= len(wanted):
                    break

        if len(out) != len(wanted):
            raise RuntimeError(
                f"Requested {len(wanted)} frames from {path}, got {len(out)} (last_index={last})"
            )

        return torch.stack(out, dim=0).to(torch.uint8)


class _PyAVBackend:
    name = "pyav"

    def __init__(self) -> None:
        import av

        self._av = av

    def get_num_frames(self, path: str) -> int:
        with self._av.open(path) as container:
            stream = container.streams.video[0]
            if stream.frames and stream.frames > 0:
                return int(stream.frames)
            count = 0
            for _ in container.decode(video=0):
                count += 1
            return count

    def read_frames(self, path: str, indices: Sequence[int]) -> torch.Tensor:
        if len(indices) == 0:
            return torch.empty((0, 0, 0, 3), dtype=torch.uint8)

        wanted = sorted(int(i) for i in indices)
        last = wanted[-1]
        ptr = 0
        out = []

        with self._av.open(path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i > last:
                    break
                if i == wanted[ptr]:
                    arr = frame.to_ndarray(format="rgb24")
                    out.append(torch.from_numpy(arr))
                    ptr += 1
                    if ptr >= len(wanted):
                        break

        if len(out) != len(wanted):
            raise RuntimeError(
                f"Requested {len(wanted)} frames from {path}, got {len(out)} (last_index={last})"
            )

        return torch.stack(out, dim=0).to(torch.uint8)


class _DecordBackend:
    name = "decord"

    def __init__(self) -> None:
        import decord

        self._decord = decord

    def get_num_frames(self, path: str) -> int:
        vr = self._decord.VideoReader(path, ctx=self._decord.cpu(0))
        return len(vr)

    def read_frames(self, path: str, indices: Sequence[int]) -> torch.Tensor:
        if len(indices) == 0:
            return torch.empty((0, 0, 0, 3), dtype=torch.uint8)

        vr = self._decord.VideoReader(path, ctx=self._decord.cpu(0))
        arr = vr.get_batch(list(indices)).asnumpy()
        return torch.from_numpy(arr).to(torch.uint8)


class VideoClipReader:
    """Read clips directly from raw video files with backend fallbacks.

    Backend order:
      1. torchvision.io.VideoReader
      2. pyav
      3. decord
    """

    def __init__(
        self,
        backend: str = "auto",
        strict: bool = False,
        max_retries: int = 3,
    ) -> None:
        self.strict = strict
        self.max_retries = max_retries
        self._num_frames_cache: dict[str, int] = {}

        candidates = []
        if backend == "auto":
            candidates = [_TorchvisionBackend, _PyAVBackend, _DecordBackend]
        elif backend == "torchvision":
            candidates = [_TorchvisionBackend]
        elif backend == "pyav":
            candidates = [_PyAVBackend]
        elif backend == "decord":
            candidates = [_DecordBackend]
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend = None
        errors = []
        for cls in candidates:
            try:
                self.backend = cls()
                break
            except Exception as e:
                errors.append(f"{cls.__name__}: {e}")

        if self.backend is None:
            raise RuntimeError(
                "Unable to initialize any video backend. "
                "Install torchvision (FFmpeg), pyav, or decord. "
                f"Errors: {' | '.join(errors)}"
            )

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def get_num_frames(self, path: str) -> int:
        path = str(Path(path))
        if path not in self._num_frames_cache:
            self._num_frames_cache[path] = int(self.backend.get_num_frames(path))
        return self._num_frames_cache[path]

    def read_clip(
        self,
        path: str,
        num_frames: int,
        stride: int,
        start_frame: Optional[int] = None,
        random_start: bool = True,
    ) -> ClipSample:
        path = str(Path(path))
        n_total = self.get_num_frames(path)
        if n_total <= 0:
            raise RuntimeError(f"Video has no frames: {path}")

        needed = (num_frames - 1) * stride + 1
        max_start = max(0, n_total - needed)

        if start_frame is None:
            if random_start:
                start = random.randint(0, max_start) if max_start > 0 else 0
            else:
                start = max_start // 2
        else:
            start = int(max(0, min(start_frame, max_start)))

        idx = start + torch.arange(num_frames, dtype=torch.long) * stride
        idx = torch.clamp(idx, min=0, max=n_total - 1)

        last_err = None
        for _ in range(self.max_retries):
            try:
                frames = self.backend.read_frames(path, idx.tolist())
                if frames.shape[0] != num_frames:
                    # Pad by repeating last frame when backend returned fewer frames.
                    if frames.shape[0] == 0:
                        raise RuntimeError("Decoded 0 frames")
                    pad = frames[-1:].repeat(num_frames - frames.shape[0], 1, 1, 1)
                    frames = torch.cat([frames, pad], dim=0)
                return ClipSample(frames=frames, start_frame=start, indices=idx)
            except Exception as e:
                last_err = e
                if random_start and max_start > 0:
                    start = random.randint(0, max_start)
                    idx = start + torch.arange(num_frames, dtype=torch.long) * stride
                    idx = torch.clamp(idx, min=0, max=n_total - 1)
                continue

        if self.strict:
            raise RuntimeError(f"Failed to decode clip from {path}: {last_err}")

        # Graceful fallback to a zero clip for fault-tolerant training.
        h = 112
        w = 112
        frames = torch.zeros((num_frames, h, w, 3), dtype=torch.uint8)
        return ClipSample(frames=frames, start_frame=start, indices=idx)
