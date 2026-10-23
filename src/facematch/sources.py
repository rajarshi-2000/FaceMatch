"""Resolve an image repository argument to a local folder of image files.

Supported inputs:
  * a local directory
  * a local ``.zip`` archive
  * a Google Drive folder URL (downloaded via ``gdown``)
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

_DRIVE_URL_RE = re.compile(r"https?://(drive|docs)\.google\.com/", re.IGNORECASE)


def is_drive_url(value: str) -> bool:
    """Return True if ``value`` looks like a Google Drive/Docs URL."""
    return bool(_DRIVE_URL_RE.match(value.strip()))


def list_images(folder: Path) -> List[Path]:
    """Recursively list image files under ``folder``, sorted for determinism."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _extract_zip(zip_path: Path, dest: Path) -> Path:
    """Safely extract ``zip_path`` into ``dest``, guarding against path traversal."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Reject absolute paths and parent-directory traversal (Zip Slip).
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Unsafe path in archive: {member!r}")
        zf.extractall(dest)
    return dest


def _download_drive_folder(url: str, dest: Path) -> Path:
    """Download a public Google Drive folder into ``dest`` using gdown."""
    try:
        import gdown  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "gdown is required to fetch Google Drive folders. "
            "Install it with `pip install gdown`."
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url=url, output=str(dest), quiet=False, use_cookies=False)
    return dest


def resolve_repository(repo: str, workdir: Path | None = None) -> Path:
    """Resolve ``repo`` to a local directory containing images.

    ``workdir`` is where zips are extracted / Drive folders downloaded. When not
    given, a temporary directory is created (and left for the OS to clean up).
    """
    repo = repo.strip()
    workdir = workdir or Path(tempfile.mkdtemp(prefix="facematch_"))

    if is_drive_url(repo):
        return _download_drive_folder(repo, workdir / "drive")

    path = Path(repo).expanduser()
    if path.is_dir():
        return path
    if path.is_file() and path.suffix.lower() == ".zip":
        return _extract_zip(path, workdir / "unzipped")

    raise FileNotFoundError(
        f"Repository {repo!r} is not a folder, a .zip file, or a Google Drive URL."
    )
