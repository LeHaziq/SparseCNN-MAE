#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def stable_split_key(path: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{path}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def scan_videos(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    files.sort()
    return files


def write_manifest(paths: list[Path], out_path: Path, split: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "split"])
        writer.writeheader()
        for p in paths:
            writer.writerow({"video_path": str(p.resolve()), "split": split})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VoxCeleb2 train/val CSV manifests from raw videos")
    parser.add_argument("--root", type=str, required=True, help="Path to VoxCeleb2 video root")
    parser.add_argument("--out_dir", type=str, default="manifests")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    videos = scan_videos(root)
    if len(videos) == 0:
        raise RuntimeError(f"No video files found under {root}")

    train, val = [], []
    for p in videos:
        key = stable_split_key(str(p), args.seed)
        if key < args.val_ratio:
            val.append(p)
        else:
            train.append(p)

    train_csv = out_dir / "voxceleb2_train.csv"
    val_csv = out_dir / "voxceleb2_val.csv"
    write_manifest(train, train_csv, "train")
    write_manifest(val, val_csv, "val")

    print(f"Wrote {train_csv} ({len(train)} rows)")
    print(f"Wrote {val_csv} ({len(val)} rows)")


if __name__ == "__main__":
    main()
