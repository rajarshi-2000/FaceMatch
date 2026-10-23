"""Face detection, embedding, caching, and matching.

Uses the open-source ``face_recognition`` library (dlib, MIT-licensed) to detect
faces and produce 128-dimensional encodings, then matches repository faces
against query faces by Euclidean distance.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sources import list_images

# dlib's recommended match threshold. Lower = stricter.
DEFAULT_THRESHOLD = 0.6


def _encode_file(path_str: str, model: str) -> Tuple[str, List[list], float, int]:
    """Encode one image in a worker process.

    Returns ``(path, encodings_as_lists, mtime, size)``. Numpy arrays are
    converted to plain lists so they pickle cheaply across the process boundary.
    This is a module-level function (not a method) so it is picklable.
    """
    import face_recognition  # imported in the worker

    try:
        image = face_recognition.load_image_file(path_str)
    except Exception:
        encodings: List[np.ndarray] = []
    else:
        locations = face_recognition.face_locations(image, model=model)
        encodings = face_recognition.face_encodings(
            image, known_face_locations=locations
        )
    try:
        stat = os.stat(path_str)
        mtime, size = stat.st_mtime, stat.st_size
    except OSError:
        mtime, size = 0.0, 0
    return path_str, [e.tolist() for e in encodings], mtime, size


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
    workers: int = 1  # parallel worker processes for encoding (1 = serial)
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
    def _compare(
        self, img_path: Path, face_encs, query_encs
    ) -> Optional[MatchResult]:
        """Return a MatchResult if ``img_path``'s closest face is within threshold."""
        if not face_encs:
            return None
        best_dist = float("inf")
        best_q = -1
        face_matrix = np.array(face_encs)
        for q_idx, q_enc in enumerate(query_encs):
            dists = np.linalg.norm(face_matrix - q_enc, axis=1)
            local_min = float(dists.min())
            if local_min < best_dist:
                best_dist = local_min
                best_q = q_idx
        if best_dist <= self.threshold:
            return MatchResult(path=img_path, distance=best_dist, query_index=best_q)
        return None

    def _iter_encodings(self, images, on_progress):
        """Yield ``(path, encodings)`` for each image, serially or in parallel.

        Cached images are served without re-encoding. Only cache-misses are
        dispatched to the worker pool. Progress is reported as each image is
        resolved (cache hit or worker completion), so counts are monotonic
        regardless of completion order. Results are independent of order because
        the caller sorts by distance.
        """
        total = len(images)
        done = 0

        # Partition into cache hits (free) and misses (need encoding).
        misses: List[Path] = []
        for p in images:
            cached = self._cache.get(str(p))
            if cached and self._cache_valid(p, cached):
                done += 1
                if on_progress:
                    on_progress(done, total)
                yield p, cached.encodings
            else:
                misses.append(p)

        if not misses:
            return

        # Serial path (workers <= 1): keeps the simple, dependency-light flow.
        if self.workers <= 1:
            for p in misses:
                encs = self.encode_image(p)
                done += 1
                if on_progress:
                    on_progress(done, total)
                yield p, encs
            return

        # Parallel path: dispatch misses to a process pool. If the pool cannot
        # be used (spawn/pickling/resource failure), fall back to serial so a
        # scan never crashes just because parallelism was unavailable.
        try:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(_encode_file, str(p), self.model): p for p in misses
                }
                for fut in as_completed(futures):
                    path_str, enc_lists, mtime, size = fut.result()
                    encs = [np.array(e, dtype=float) for e in enc_lists]
                    self._cache[path_str] = _ImageEncodings(
                        encodings=encs, mtime=mtime, size=size
                    )
                    done += 1
                    if on_progress:
                        on_progress(done, total)
                    yield Path(path_str), encs
        except Exception:
            # Pool unusable — degrade gracefully to serial for the remaining
            # (still-unencoded) images.
            for p in misses:
                if str(p) in self._cache and self._cache_valid(p, self._cache[str(p)]):
                    continue  # already encoded by the pool before it died
                encs = self.encode_image(p)
                done += 1
                if on_progress:
                    on_progress(done, total)
                yield p, encs

    def match(
        self,
        repo_dir: Path,
        query_paths: Sequence[Path],
        progress: bool = True,
        on_progress=None,
    ) -> List[MatchResult]:
        """Find repository images containing any of the query faces.

        ``on_progress`` is an optional callable invoked after each image with
        ``(processed, total)`` — used by the web UI to drive a progress bar.

        Encoding runs in parallel across ``self.workers`` processes when that is
        greater than 1. Parallelism does not change the result: the same dlib
        detector and encodings are used, and matches are sorted by distance.
        """
        query_encs = self.encode_queries(query_paths)
        images = list_images(repo_dir)
        total = len(images)
        if on_progress:
            on_progress(0, total)

        use_tqdm = progress
        bar = None
        if use_tqdm:
            try:
                from tqdm import tqdm  # type: ignore

                bar = tqdm(total=total, desc="Scanning", unit="img")
            except ImportError:
                bar = None

        results: List[MatchResult] = []
        for img_path, face_encs in self._iter_encodings(images, on_progress):
            match = self._compare(img_path, face_encs, query_encs)
            if match is not None:
                results.append(match)
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

        results.sort(key=lambda r: r.distance)
        return results
