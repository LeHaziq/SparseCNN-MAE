#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def stable_key(value: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def infer_subject_id(path: Path) -> str:
    text = str(path)
    patterns = [r"SN\d+", r"S\d+", r"subj\d+", r"subject\d+", r"\b\d{2,}\b"]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return path.stem


def scan_video_sequences(root: Path) -> list[dict[str, str]]:
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rows.append({"video_path": str(p.resolve()), "frame_dir": "", "num_frames": ""})
    return rows


def scan_frame_sequences(root: Path) -> list[dict[str, str]]:
    rows = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        n_img = sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if n_img >= 32:
            rows.append({"video_path": "", "frame_dir": str(d.resolve()), "num_frames": str(n_img)})
    return rows


def find_annotation_for_subject(ann_root: Path, subject_id: str) -> str:
    cands = [p for p in ann_root.rglob("*") if p.is_file() or p.is_dir()]
    cands = [p for p in cands if subject_id.lower() in str(p).lower()]
    cands.sort(key=lambda p: len(str(p)))
    for p in cands:
        if p.is_dir():
            return str(p.resolve())
        if p.suffix.lower() in {".csv", ".txt", ".npy", ".npz", ".pt", ".pth"}:
            return str(p.resolve())
    return ""


def assign_split_by_subject(subject_ids: list[str], seed: int, val_ratio: float, test_ratio: float) -> dict[str, str]:
    split_map = {}
    for sid in sorted(set(subject_ids)):
        x = stable_key(sid, seed)
        if x < test_ratio:
            split_map[sid] = "test"
        elif x < test_ratio + val_ratio:
            split_map[sid] = "val"
        else:
            split_map[sid] = "train"
    return split_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DISFA manifest (video/frame folders + annotations)")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--videos_root", type=str, default=None)
    parser.add_argument("--frames_root", type=str, default=None)
    parser.add_argument("--annotations_root", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default="manifests/disfa_manifest.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    videos_root = Path(args.videos_root) if args.videos_root else root
    frames_root = Path(args.frames_root) if args.frames_root else root
    annotations_root = Path(args.annotations_root) if args.annotations_root else root

    rows = []
    rows.extend(scan_video_sequences(videos_root))
    rows.extend(scan_frame_sequences(frames_root))

    # Deduplicate same source path.
    dedup = {}
    for r in rows:
        key = r["video_path"] or r["frame_dir"]
        dedup[key] = r
    rows = list(dedup.values())

    if len(rows) == 0:
        raise RuntimeError("No video/frame sequences found")

    for r in rows:
        source = Path(r["video_path"] or r["frame_dir"])
        sid = infer_subject_id(source)
        r["subject_id"] = sid
        r["annotation_path"] = find_annotation_for_subject(annotations_root, sid)

    # Deterministic subject-level split when subject IDs are available.
    split_map = assign_split_by_subject(
        [r["subject_id"] for r in rows],
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    for r in rows:
        r["split"] = split_map[r["subject_id"]]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject_id",
        "split",
        "video_path",
        "frame_dir",
        "annotation_path",
        "num_frames",
        "cache_path",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "subject_id": r.get("subject_id", ""),
                    "split": r.get("split", "train"),
                    "video_path": r.get("video_path", ""),
                    "frame_dir": r.get("frame_dir", ""),
                    "annotation_path": r.get("annotation_path", ""),
                    "num_frames": r.get("num_frames", ""),
                    "cache_path": "",
                }
            )

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(f"Wrote manifest: {out_csv}")
    print(f"Rows: total={len(rows)} train={n_train} val={n_val} test={n_test}")


if __name__ == "__main__":
    main()
