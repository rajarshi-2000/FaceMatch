"""Small local web UI for FaceMatch.

Upload a repository (a ``.zip`` of images, or point at a local folder path) plus
one or more query face images, run the matcher as a background job, watch a live
progress bar, and view matches as thumbnails with their distance scores.

Runs on 127.0.0.1 only — this tool processes biometric data and must not be
exposed on a network. See ``NOTICE.md``.
"""

from __future__ import annotations

import io
import os
import threading
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

from .engine import DEFAULT_THRESHOLD, FaceMatcher, MatchResult
from .sources import IMAGE_EXTENSIONS, resolve_repository

# Upload / request guards.
MAX_CONTENT_LENGTH = 512 * 1024 * 1024  # 512 MB total request cap
THUMBNAIL_MAX = (256, 256)


@dataclass
class Job:
    """State for one background search job."""

    id: str
    status: str = "starting"  # starting | preparing | scanning | done | error
    processed: int = 0
    total: int = 0
    error: Optional[str] = None
    threshold: float = DEFAULT_THRESHOLD
    results: List[MatchResult] = field(default_factory=list)
    # token -> absolute path, so the thumbnail route only serves match files.
    files: Dict[str, Path] = field(default_factory=dict)

    def as_status(self) -> dict:
        pct = int(self.processed / self.total * 100) if self.total else 0
        return {
            "status": self.status,
            "processed": self.processed,
            "total": self.total,
            "percent": pct,
            "error": self.error,
        }


_JOBS: Dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def _save_uploads(files, dest: Path) -> List[Path]:
    """Persist uploaded query images into ``dest``; return their paths."""
    dest.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for f in files:
        if not f or not f.filename:
            continue
        name = Path(f.filename).name  # basename only — no traversal
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target = dest / name
        f.save(str(target))
        saved.append(target)
    return saved


def _resolve_repo(upload, repo_path: str, workdir: Path) -> Path:
    """Resolve the repository from an uploaded zip or a folder path string."""
    if upload and upload.filename:
        if not upload.filename.lower().endswith(".zip"):
            raise ValueError("Repository upload must be a .zip file.")
        zip_path = workdir / "repo.zip"
        upload.save(str(zip_path))
        return resolve_repository(str(zip_path), workdir=workdir)
    folder = (repo_path or "").strip()
    if folder:
        return resolve_repository(folder, workdir=workdir)
    raise ValueError("Provide a repository: upload a .zip or enter a folder path.")


def _run_job(job: Job, repo_dir: Path, query_paths: List[Path]) -> None:
    """Background worker: run the match, updating job progress as it goes."""
    try:
        job.status = "scanning"

        def on_progress(processed: int, total: int) -> None:
            job.processed = processed
            job.total = total

        # Cache encodings next to the repo so repeat searches of the same
        # library (e.g. a different person) skip re-encoding. Parallelise the
        # encode step across CPU cores for large repositories.
        cache_path = repo_dir / ".facematch_cache.json"
        workers = max(1, (os.cpu_count() or 1))
        matcher = FaceMatcher(
            threshold=job.threshold, cache_path=cache_path, workers=workers
        )
        try:
            matcher.load_cache()
        except Exception:
            pass

        results = matcher.match(
            repo_dir, query_paths, progress=False, on_progress=on_progress
        )

        try:
            matcher.save_cache()
        except Exception:
            pass

        for i, r in enumerate(results):
            token = f"{i}_{r.path.name}"
            job.files[token] = r.path
        job.results = results
        job.status = "done"
    except Exception as exc:  # surface any failure to the UI
        job.error = str(exc)
        job.status = "error"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.get("/")
    def index():
        return render_template("index.html", default_threshold=DEFAULT_THRESHOLD)

    @app.post("/search")
    def search():
        queries = request.files.getlist("queries")
        if not queries or all(not q.filename for q in queries):
            return jsonify(error="Upload at least one query face image."), 400

        try:
            threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
        except ValueError:
            threshold = DEFAULT_THRESHOLD

        workdir = Path(tempfile.mkdtemp(prefix="facematch_web_"))
        try:
            repo_dir = _resolve_repo(
                request.files.get("repo_zip"),
                request.form.get("repo_path", ""),
                workdir,
            )
            query_paths = _save_uploads(queries, workdir / "queries")
            if not query_paths:
                raise ValueError("No valid query images (supported: images only).")
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return jsonify(error=str(exc)), 400

        job = Job(id=uuid.uuid4().hex, status="preparing", threshold=threshold)
        with _JOBS_LOCK:
            _JOBS[job.id] = job
        thread = threading.Thread(
            target=_run_job, args=(job, repo_dir, query_paths), daemon=True
        )
        thread.start()
        return jsonify(job_id=job.id)

    @app.get("/progress/<job_id>")
    def progress(job_id: str):
        job = _JOBS.get(job_id)
        if not job:
            abort(404)
        return jsonify(job.as_status())

    @app.get("/results/<job_id>")
    def results(job_id: str):
        job = _JOBS.get(job_id)
        if not job:
            abort(404)
        if job.status == "error":
            return render_template(
                "index.html", default_threshold=DEFAULT_THRESHOLD, error=job.error
            )
        view = []
        for i, r in enumerate(job.results):
            token = f"{i}_{r.path.name}"
            view.append(
                {
                    "token": token,
                    "name": r.path.name,
                    "distance": round(r.distance, 4),
                    "query_index": r.query_index,
                    "thumb_url": url_for("thumb", job_id=job_id, token=token),
                }
            )
        return render_template(
            "results.html",
            results=view,
            count=len(view),
            threshold=job.threshold,
            job_id=job_id,
        )

    @app.get("/thumb/<job_id>/<token>")
    def thumb(job_id: str, token: str):
        job = _JOBS.get(job_id)
        if not job or token not in job.files:
            abort(404)
        path = job.files[token]
        try:
            from PIL import Image  # type: ignore

            img = Image.open(path)
            img.thumbnail(THUMBNAIL_MAX)
            buf = io.BytesIO()
            fmt = "PNG" if (img.mode in ("RGBA", "P")) else "JPEG"
            img.convert("RGB").save(buf, format=fmt)
            buf.seek(0)
            return send_file(buf, mimetype=f"image/{fmt.lower()}")
        except Exception:
            return send_file(str(path))

    @app.get("/download/<job_id>/<token>")
    def download_one(job_id: str, token: str):
        """Download a single matched image at full resolution."""
        job = _JOBS.get(job_id)
        if not job or token not in job.files:
            abort(404)
        path = job.files[token]
        return send_file(
            str(path), as_attachment=True, download_name=path.name
        )

    @app.get("/download/<job_id>")
    def download_all(job_id: str):
        """Download every matched image for a job as a single ZIP."""
        job = _JOBS.get(job_id)
        if not job:
            abort(404)
        if not job.files:
            abort(404)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            used: Dict[str, int] = {}
            for path in job.files.values():
                # De-duplicate names inside the archive (two folders may share a
                # filename) by suffixing a counter.
                name = path.name
                if name in used:
                    used[name] += 1
                    stem, dot, ext = name.partition(".")
                    name = f"{stem}_{used[path.name]}{dot}{ext}"
                else:
                    used[name] = 0
                try:
                    zf.write(str(path), arcname=name)
                except OSError:
                    continue
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"facematch_{job_id[:8]}.zip",
        )

    return app


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="facematch-web",
        description="Local web UI for FaceMatch (binds to 127.0.0.1 only).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default 5000).")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error(
            "Refusing to bind to a non-loopback host. FaceMatch processes "
            "biometric data and must stay local."
        )

    # threaded=True so progress polling works while a job runs.
    create_app().run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
