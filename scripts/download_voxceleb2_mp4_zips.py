#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen

from urllib.error import HTTPError, URLError
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only VoxCeleb2 MP4 zip/part archives from Hugging Face."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Reverb/voxceleb2",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Repo revision to download from",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/voxceleb2",
        help="Output directory",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=r"mp4.*(zip|part)",
        help=(
            "Regex applied to each repo file path (case-insensitive). "
            "Default matches mp4 zip files and split zip parts."
        ),
    )
    parser.add_argument(
        "--strict_zip_ext",
        action="store_true",
        help="If set, only keep files ending in .zip",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (or use HF_TOKEN/HUGGINGFACE_TOKEN env var)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when destination file already exists",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print matched files without downloading",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of files to download",
    )
    return parser.parse_args()


def build_auth_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "voxceleb2-mp4-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def request_json(url: str, headers: dict[str, str]) -> dict:
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def list_dataset_files(repo_id: str, token: str | None) -> list[str]:
    api_url = f"https://huggingface.co/api/datasets/{quote(repo_id, safe='/')}"
    data = request_json(api_url, build_auth_headers(token))
    siblings = data.get("siblings", [])

    paths: list[str] = []
    for item in siblings:
        if isinstance(item, str):
            paths.append(item)
            continue
        if not isinstance(item, dict):
            continue
        path = item.get("rfilename") or item.get("path") or item.get("name")
        if isinstance(path, str):
            paths.append(path)

    # Keep stable order, remove duplicates.
    deduped = list(dict.fromkeys(paths))
    deduped.sort()
    return deduped


def filter_files(paths: Iterable[str], pattern: str, strict_zip_ext: bool) -> list[str]:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    matched = [p for p in paths if regex.search(p)]
    if strict_zip_ext:
        matched = [p for p in matched if p.lower().endswith(".zip")]
    return matched


def resolve_url(repo_id: str, revision: str, repo_path: str) -> str:
    repo = quote(repo_id, safe="/")
    rev = quote(revision, safe="")
    path = quote(repo_path, safe="/")
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}?download=true"


def infer_total_size(
    content_range: str | None, content_length: str | None, start: int, code: int
) -> int | None:
    if content_range:
        match = re.match(r"bytes \d+-\d+/(\d+|\*)", content_range)
        if match and match.group(1) != "*":
            return int(match.group(1))

    if content_length:
        try:
            length = int(content_length)
        except ValueError:
            return None
        if start > 0 and code == 206:
            return start + length
        return length

    return None


def download_file(url: str, dst: Path, headers: dict[str, str], force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if force and dst.exists():
        dst.unlink()

    start = dst.stat().st_size if dst.exists() else 0
    req_headers = dict(headers)
    if start > 0:
        req_headers["Range"] = f"bytes={start}-"

    req = Request(url, headers=req_headers)

    try:
        with urlopen(req) as resp:
            code = getattr(resp, "status", resp.getcode())
            mode = "ab" if (start > 0 and code == 206) else "wb"
            if mode == "wb" and start > 0:
                start = 0
            total = infer_total_size(
                resp.headers.get("Content-Range"),
                resp.headers.get("Content-Length"),
                start,
                code,
            )
            with dst.open(mode) as f, tqdm(
                total=total,
                initial=start,
                desc=dst.name,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    except HTTPError as e:
        if e.code == 416:
            # Already fully downloaded.
            return
        raise


def main() -> int:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    out_dir = Path(args.out_dir)

    try:
        all_files = list_dataset_files(args.repo_id, token)
    except (HTTPError, URLError) as e:
        print(f"Failed to list dataset files: {e}", file=sys.stderr)
        return 1

    targets = filter_files(all_files, args.pattern, args.strict_zip_ext)
    if args.max_files is not None:
        targets = targets[: args.max_files]

    if not targets:
        print("No files matched the filter. Adjust --pattern / --strict_zip_ext.")
        return 1

    print(f"Found {len(targets)} matching files.")
    for path in targets:
        print(f" - {path}")

    if args.dry_run:
        return 0

    headers = build_auth_headers(token)
    for i, repo_path in enumerate(targets, start=1):
        dst = out_dir / repo_path
        url = resolve_url(args.repo_id, args.revision, repo_path)
        print(f"[{i}/{len(targets)}] Downloading {repo_path}")
        try:
            download_file(url, dst, headers, force=args.force)
        except (HTTPError, URLError) as e:
            print(f"Failed to download {repo_path}: {e}", file=sys.stderr)
            return 1

    print(f"Done. Files saved under: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
