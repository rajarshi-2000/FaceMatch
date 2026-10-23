"""Face detection, embedding, caching, and matching.

Uses the open-source ``face_recognition`` library (dlib, MIT-licensed) to detect
faces and produce 128-dimensional encodings, then matches repository faces
against query faces by Euclidean distance.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .sources import list_images

# dlib's recommended match threshold. Lower = stricter.
DEFAULT_THRESHOLD = 0.6


@dataclass
class MatchResult:
    """A single repository image that matched at least one query face."""

    path: Path
    distance: float  # best (smallest) distance across all faces/queries
    query_index: int  # which query face produced the best match

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "distance": round(self.distance, 4),
            "query_index": self.query_index,
        }


@dataclass
class _ImageEncodings:
    """Face encodings for one image, plus the file signature used for caching."""

    encodings: List[np.ndarray]
    mtime: float
    size: int


@dataclass
class FaceMatcher:
    """Detect, cache, and match faces across an image repository."""

    model: str = "hog"  # "hog" (CPU) or "cnn" (GPU)
    threshold: float = DEFAULT_THRESHOLD
    cache_path: Optional[Path] = None
    _cache: Dict[str, _ImageEncodings] = field(default_factory=dict, repr=False)

    # ---- lazy import so `import facematch` works without dlib present ----
    @staticmethod
    def _fr():
        try:
            import face_recognition  # type: ignore

            return face_recognition
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "face_recognition is required. Install it with "
                "`pip install face_recognition` (needs CMake + a C++ toolchain)."
            ) from exc

    # ---- caching ----
    def load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            raw = json.loads(self.cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for key, entry in raw.items():
            self._cache[key] = _ImageEncodings(
                encodings=[np.array(e, dtype=float) for e in entry["encodings"]],
                mtime=entry["mtime"],
                size=entry["size"],
            )

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        serializable = {
            key: {
                "encodings": [e.tolist() for e in val.encodings],
                "mtime": val.mtime,
                "size": val.size,
            }
            for key, val in self._cache.items()
        }
        self.cache_path.write_text(json.dumps(serializable))

    def _cache_valid(self, path: Path, entry: _ImageEncodings) -> bool:
        try:
            stat = path.stat()
        except OSError:
            return False
        return entry.mtime == stat.st_mtime and entry.size == stat.st_size

    # ---- encoding ----
    def encode_image(self, path: Path) -> List[np.ndarray]:
        """Return all face encodings in ``path`` (cached when possible)."""
        key = str(path)
        cached = self._cache.get(key)
        if cached and self._cache_valid(path, cached):
            return cached.encodings

        fr = self._fr()
        try:
            image = fr.load_image_file(str(path))
        except Exception:
            # Unreadable / corrupt image — treat as zero faces.
            encodings: List[np.ndarray] = []
        else:
            locations = fr.face_locations(image, model=self.model)
            encodings = fr.face_encodings(image, known_face_locations=locations)

        stat = path.stat()
        self._cache[key] = _ImageEncodings(
            encodings=encodings, mtime=stat.st_mtime, size=stat.st_size
        )
        return encodings

    def encode_queries(self, query_paths: Sequence[Path]) -> List[np.ndarray]:
        """Return one encoding per query image (first face found).

        Raises ``ValueError`` if any query image contains no detectable face.
        """
        fr = self._fr()
        encs: List[np.ndarray] = []
        for qp in query_paths:
            image = fr.load_image_file(str(qp))
            found = fr.face_encodings(image)
            if not found:
                raise ValueError(f"No face found in query image: {qp}")
            encs.append(found[0])
        return encs

    # ---- matching ----
    def match(
        self,
        repo_dir: Path,
        query_paths: Sequence[Path],
        progress: bool = True,
    ) -> List[MatchResult]:
        """Find repository images containing any of the query faces."""
        query_encs = self.encode_queries(query_paths)
        images = list_images(repo_dir)

        iterator = images
        if progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(images, desc="Scanning", unit="img")
            except ImportError:
                iterator = images

        results: List[MatchResult] = []
        for img_path in iterator:
            face_encs = self.encode_image(img_path)
            if not face_encs:
                continue
            best_dist = float("inf")
            best_q = -1
            for q_idx, q_enc in enumerate(query_encs):
                # Euclidean distance between this query and every face in image.
                dists = np.linalg.norm(np.array(face_encs) - q_enc, axis=1)
                local_min = float(dists.min())
                if local_min < best_dist:
                    best_dist = local_min
                    best_q = q_idx
            if best_dist <= self.threshold:
                results.append(
                    MatchResult(path=img_path, distance=best_dist, query_index=best_q)
                )

        results.sort(key=lambda r: r.distance)
        return results
