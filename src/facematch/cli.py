"""Command-line interface for FaceMatch."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

from .engine import DEFAULT_THRESHOLD, FaceMatcher
from .sources import resolve_repository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facematch",
        description="Find photos of specific people inside an image repository.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Image repository: a folder, a .zip file, or a Google Drive folder URL.",
    )
    parser.add_argument(
        "--query",
        required=True,
        action="append",
        metavar="IMAGE",
        help="Reference face image. Repeat to search for multiple people.",
    )
    parser.add_argument(
        "--out",
        help="Directory to copy matching images into (created if missing).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Max face distance to count as a match (default {DEFAULT_THRESHOLD}; "
        "lower is stricter).",
    )
    parser.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detector: 'hog' (fast, CPU) or 'cnn' (accurate, GPU).",
    )
    parser.add_argument(
        "--cache",
        default=".facematch_cache.json",
        help="Embedding cache file (default .facematch_cache.json).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the embedding cache.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the match report as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the progress bar.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    query_paths = [Path(q).expanduser() for q in args.query]
    missing = [str(q) for q in query_paths if not q.is_file()]
    if missing:
        print(f"error: query image(s) not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        repo_dir = resolve_repository(args.repo)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cache_path = None if args.no_cache else Path(args.cache)
    matcher = FaceMatcher(
        model=args.model,
        threshold=args.threshold,
        cache_path=cache_path,
    )
    if cache_path:
        matcher.load_cache()

    try:
        results = matcher.match(repo_dir, query_paths, progress=not args.quiet)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if cache_path:
            matcher.save_cache()

    if args.out:
        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            shutil.copy2(r.path, out_dir / r.path.name)

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
    else:
        if not results:
            print("No matches found.")
        else:
            print(f"Found {len(results)} matching image(s):")
            for r in results:
                print(f"  {r.distance:.4f}  (query #{r.query_index})  {r.path}")
            if args.out:
                print(f"\nCopied {len(results)} image(s) into {args.out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
