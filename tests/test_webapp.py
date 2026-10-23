import io
import zipfile
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from facematch import webapp
from facematch.engine import MatchResult


@pytest.fixture
def client():
    app = webapp.create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _png_bytes() -> bytes:
    """A tiny valid 1x1 PNG so Pillow can open it for thumbnailing."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"FaceMatch" in resp.data
    assert b"consent" in resp.data  # responsible-use notice present


def test_search_requires_query(client):
    resp = client.post("/search", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"at least one query" in resp.data


def test_search_happy_path(client, monkeypatch, tmp_path):
    # Build a zip repo with one image.
    img_bytes = _png_bytes()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("alice.png", img_bytes)
    zip_buf.seek(0)

    # Make the matcher return a deterministic match pointing at a real file,
    # so the thumbnail route has something to serve.
    match_file = tmp_path / "alice.png"
    match_file.write_bytes(img_bytes)

    def fake_match(self, repo_dir, query_paths, progress=False):
        return [MatchResult(path=match_file, distance=0.12, query_index=0)]

    monkeypatch.setattr("facematch.webapp.FaceMatcher.match", fake_match)

    data = {
        "repo_zip": (zip_buf, "repo.zip"),
        "queries": (io.BytesIO(img_bytes), "q.png"),
        "threshold": "0.6",
    }
    resp = client.post("/search", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"1 match" in resp.data
    assert b"dist 0.12" in resp.data


def test_thumb_unknown_run_404(client):
    resp = client.get("/thumb/nope/0_x.png")
    assert resp.status_code == 404


def test_web_main_rejects_non_loopback(monkeypatch):
    with pytest.raises(SystemExit):
        webapp.main(["--host", "0.0.0.0"])
