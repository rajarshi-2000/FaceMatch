"""Small local web UI for FaceMatch.

Upload a repository (a ``.zip`` of images, or point at a local folder path) plus
one or more query face images, run the matcher, and view matches as thumbnails
with their distance scores.

Runs on 127.0.0.1 only — this tool processes biometric data and must not be
exposed on a network. See ``NOTICE.md``.
"""

from __future__ import annotations

import io
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Dict

from flask import (
    Flask,
    abort,
    render_template,
    request,
    send_file,
    url_for,
)

from .engine import DEFAULT_THRESHOLD, FaceMatcher
from .sources import IMAGE_EXTENSIONS, resolve_repository

# Upload / request guards.
MAX_CONTENT_LENGTH = 512 * 1024 * 1024  # 512 MB total request cap
THUMBNAIL_MAX = (256, 256)

# In-memory map of run id -> {image name -> absolute path on disk}, so the
# thumbnail route can only ever serve files this process produced as matches.
_RUNS: Dict[str, Dict[str, Path]] = {}


def _save_uploads(files, dest: Path) -> list[Path]:
    """Persist uploaded query images into ``dest``; return their paths."""
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in files:
        if not f or not f.filename:
            continue
        # Keep only the basename to avoid directory traversal from the filename.
        name = Path(f.filename).name
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target = dest / name
        f.save(str(target))
        saved.append(target)
    return saved


def _resolve_repo_from_request(req, workdir: Path) -> Path:
    """Resolve the repository from either an uploaded zip or a folder path."""
    upload = req.files.get("repo_zip")
    if upload and upload.filename:
        if not upload.filename.lower().endswith(".zip"):
            raise ValueError("Repository upload must be a .zip file.")
        zip_path = workdir / "repo.zip"
        upload.save(str(zip_path))
        return resolve_repository(str(zip_path), workdir=workdir)

    folder = (req.form.get("repo_path") or "").strip()
    if folder:
        return resolve_repository(folder, workdir=workdir)

    raise ValueError("Provide a repository: upload a .zip or enter a folder path.")


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
            return render_template(
                "index.html",
                default_threshold=DEFAULT_THRESHOLD,
                error="Upload at least one query face image.",
            )

        try:
            threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
        except ValueError:
            threshold = DEFAULT_THRESHOLD

        workdir = Path(tempfile.mkdtemp(prefix="facematch_web_"))
        try:
            repo_dir = _resolve_repo_from_request(request, workdir)
            query_paths = _save_uploads(queries, workdir / "queries")
            if not query_paths:
                raise ValueError("No valid query images (supported: images only).")

            matcher = FaceMatcher(threshold=threshold, cache_path=None)
            results = matcher.match(repo_dir, query_paths, progress=False)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return render_template(
                "index.html",
                default_threshold=DEFAULT_THRESHOLD,
                error=str(exc),
            )

        run_id = uuid.uuid4().hex
        _RUNS[run_id] = {}
        view = []
        for i, r in enumerate(results):
            token = f"{i}_{r.path.name}"
            _RUNS[run_id][token] = r.path
            view.append(
                {
                    "token": token,
                    "name": r.path.name,
                    "distance": round(r.distance, 4),
                    "query_index": r.query_index,
                    "thumb_url": url_for("thumb", run_id=run_id, token=token),
                }
            )

        return render_template(
            "results.html",
            results=view,
            count=len(view),
            threshold=threshold,
        )

    @app.get("/thumb/<run_id>/<token>")
    def thumb(run_id: str, token: str):
        run = _RUNS.get(run_id)
        if not run or token not in run:
            abort(404)
        path = run[token]
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
            # Fall back to serving the original file if Pillow is unavailable.
            return send_file(str(path))

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

    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
