from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _center_square_crop(frame: torch.Tensor) -> torch.Tensor:
    h, w, _ = frame.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return frame[top : top + side, left : left + side, :]


def _resize_frames(frames: torch.Tensor, size: int) -> torch.Tensor:
    # frames: [T,H,W,C] uint8
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous()


class FaceAligner:
    """Optional face alignment with cache.

    Modes:
      - off: no face detection/alignment
      - mtcnn: facenet-pytorch MTCNN (if installed)
      - insightface: InsightFace detector (if installed)
    """

    def __init__(
        self,
        mode: str = "off",
        cache_dir: Optional[str] = None,
        output_size: int = 112,
        device: Optional[str] = None,
    ) -> None:
        self.mode = mode.lower()
        self.output_size = int(output_size)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
        self.enabled = self.mode != "off"

        if self.mode == "off":
            return

        if self.mode == "mtcnn":
            try:
                from facenet_pytorch import MTCNN

                self.detector = MTCNN(keep_all=False, device=self.device)
            except Exception as e:
                warnings.warn(f"MTCNN unavailable ({e}); face alignment disabled.", stacklevel=2)
                self.mode = "off"
                self.enabled = False
        elif self.mode == "insightface":
            try:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(name="buffalo_l")
                ctx_id = 0 if self.device.startswith("cuda") else -1
                app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self.detector = app
            except Exception as e:
                warnings.warn(f"InsightFace unavailable ({e}); face alignment disabled.", stacklevel=2)
                self.mode = "off"
                self.enabled = False
        else:
            warnings.warn(f"Unknown face align mode '{self.mode}', disabling.", stacklevel=2)
            self.mode = "off"
            self.enabled = False

    def _cache_path(
        self,
        video_path: str,
        start_frame: int,
        stride: int,
        clip_len: int,
    ) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = f"{video_path}|{start_frame}|{stride}|{clip_len}|{self.mode}|{self.output_size}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.pt"

    def _align_frame_mtcnn(self, frame: torch.Tensor) -> torch.Tensor:
        img = Image.fromarray(frame.numpy())
        boxes, _ = self.detector.detect(img)
        if boxes is None or len(boxes) == 0:
            crop = _center_square_crop(frame)
            return _resize_frames(crop.unsqueeze(0), self.output_size)[0]

        x1, y1, x2, y2 = boxes[0]
        h, w, _ = frame.shape
        x1 = int(max(0, min(w - 1, x1)))
        x2 = int(max(x1 + 1, min(w, x2)))
        y1 = int(max(0, min(h - 1, y1)))
        y2 = int(max(y1 + 1, min(h, y2)))
        crop = frame[y1:y2, x1:x2, :]
        return _resize_frames(crop.unsqueeze(0), self.output_size)[0]

    def _align_frame_insightface(self, frame: torch.Tensor) -> torch.Tensor:
        np_frame = frame.numpy()
        bgr = np_frame[:, :, ::-1]
        faces = self.detector.get(bgr)
        if len(faces) == 0:
            crop = _center_square_crop(frame)
            return _resize_frames(crop.unsqueeze(0), self.output_size)[0]

        # Largest face by bbox area.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(np.int32).tolist()
        h, w, _ = frame.shape
        x1 = int(max(0, min(w - 1, x1)))
        x2 = int(max(x1 + 1, min(w, x2)))
        y1 = int(max(0, min(h - 1, y1)))
        y2 = int(max(y1 + 1, min(h, y2)))
        crop = frame[y1:y2, x1:x2, :]
        return _resize_frames(crop.unsqueeze(0), self.output_size)[0]

    def align_frames(
        self,
        frames: torch.Tensor,
        video_path: str,
        start_frame: int,
        stride: int,
        clip_len: int,
    ) -> torch.Tensor:
        if not self.enabled:
            return frames

        cache_path = self._cache_path(video_path, start_frame, stride, clip_len)
        if cache_path and cache_path.exists():
            cached = torch.load(cache_path)
            if isinstance(cached, dict) and "frames" in cached:
                return cached["frames"].to(torch.uint8)
            if isinstance(cached, torch.Tensor):
                return cached.to(torch.uint8)

        aligned = []
        for i in range(frames.shape[0]):
            frame = frames[i]
            if self.mode == "mtcnn":
                out = self._align_frame_mtcnn(frame)
            elif self.mode == "insightface":
                out = self._align_frame_insightface(frame)
            else:
                out = frame
            aligned.append(out)

        out_frames = torch.stack(aligned, dim=0).to(torch.uint8)
        if cache_path:
            torch.save(
                {
                    "frames": out_frames,
                    "video_path": video_path,
                    "start_frame": start_frame,
                    "stride": stride,
                    "clip_len": clip_len,
                    "mode": self.mode,
                },
                cache_path,
            )
        return out_frames
