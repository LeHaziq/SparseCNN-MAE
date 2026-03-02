from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from preprocess.face_align import FaceAligner
from src.data.video_reader import VideoClipReader

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _sorted_image_files(frame_dir: str) -> list[Path]:
    files = [p for p in Path(frame_dir).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def _load_frames_from_dir(frame_dir: str, indices: list[int]) -> torch.Tensor:
    files = _sorted_image_files(frame_dir)
    if len(files) == 0:
        raise RuntimeError(f"No image files found in frame_dir={frame_dir}")
    out = []
    max_idx = len(files) - 1
    for idx in indices:
        idx = int(max(0, min(idx, max_idx)))
        with Image.open(files[idx]).convert("RGB") as img:
            out.append(torch.from_numpy(np.array(img, dtype=np.uint8)))
    return torch.stack(out, dim=0)


def _make_starts(total_frames: int, clip_len: int, stride: int, clips_per_video: int) -> list[int]:
    need = (clip_len - 1) * stride + 1
    max_start = max(0, total_frames - need)
    if clips_per_video <= 1:
        return [max_start // 2]
    starts = np.linspace(0, max_start, num=clips_per_video, dtype=np.int64)
    return [int(s) for s in starts.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline face alignment/cache for clip datasets")
    parser.add_argument("--input_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--mode", type=str, default="mtcnn", choices=["off", "mtcnn", "insightface"])
    parser.add_argument("--clip_len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--clips_per_video", type=int, default=1)
    parser.add_argument("--video_backend", type=str, default="auto")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise RuntimeError(f"Manifest has no rows: {args.input_manifest}")

    reader = VideoClipReader(backend=args.video_backend)
    aligner = FaceAligner(mode=args.mode, cache_dir=None, output_size=112)

    new_rows = []
    for row in tqdm(rows, desc="align"):
        video_path = row.get("video_path", "")
        frame_dir = row.get("frame_dir", "")
        source = video_path or frame_dir
        if not source:
            continue

        if video_path and Path(video_path).exists():
            total_frames = int(row.get("num_frames", 0)) or reader.get_num_frames(video_path)
        elif frame_dir and Path(frame_dir).exists():
            total_frames = len(_sorted_image_files(frame_dir))
        else:
            continue

        starts = _make_starts(total_frames, args.clip_len, args.stride, args.clips_per_video)
        for start in starts:
            idx = start + torch.arange(args.clip_len, dtype=torch.long) * args.stride
            idx = torch.clamp(idx, min=0, max=max(total_frames - 1, 0))

            if video_path and Path(video_path).exists():
                sample = reader.read_clip(
                    path=video_path,
                    num_frames=args.clip_len,
                    stride=args.stride,
                    start_frame=start,
                    random_start=False,
                )
                frames = sample.frames
            else:
                frames = _load_frames_from_dir(frame_dir, idx.tolist())

            aligned = aligner.align_frames(
                frames,
                video_path=source,
                start_frame=start,
                stride=args.stride,
                clip_len=args.clip_len,
            )

            cache_name = f"{len(new_rows):08d}.pt"
            cache_path = out_dir / cache_name
            torch.save(
                {
                    "frames": aligned,
                    "start_frame": start,
                    "stride": args.stride,
                    "clip_len": args.clip_len,
                    "source": source,
                    "align_mode": args.mode,
                },
                cache_path,
            )

            updated = dict(row)
            updated["cache_path"] = str(cache_path.resolve())
            updated["start_frame"] = str(start)
            updated["num_frames"] = str(args.clip_len)
            new_rows.append(updated)

    fieldnames = sorted({k for r in new_rows for k in r.keys()})
    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"Wrote aligned clips: {out_dir}")
    print(f"Wrote updated manifest: {out_manifest} ({len(new_rows)} rows)")


if __name__ == "__main__":
    main()
