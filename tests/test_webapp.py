import io
import time
import zipfile

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
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"FaceMatch" in resp.data
    assert b"consent" in resp.data
    assert b"progress-bar" in resp.data  # progress UI present


def test_search_requires_query_returns_400_json(client):
    resp = client.post("/search", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"at least one query" in resp.data


def _wait_for_done(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/progress/{job_id}").get_json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_async_search_flow(client, monkeypatch, tmp_path):
    img_bytes = _png_bytes()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("alice.png", img_bytes)
    zip_buf.seek(0)

    match_file = tmp_path / "alice.png"
    match_file.write_bytes(img_bytes)

    def fake_match(self, repo_dir, query_paths, progress=False, on_progress=None):
        if on_progress:
            on_progress(0, 1)
            on_progress(1, 1)
        return [MatchResult(path=match_file, distance=0.12, query_index=0)]

    monkeypatch.setattr("facematch.webapp.FaceMatcher.match", fake_match)

    data = {
        "repo_zip": (zip_buf, "repo.zip"),
        "queries": (io.BytesIO(img_bytes), "q.png"),
        "threshold": "0.6",
    }
    resp = client.post("/search", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]

    status = _wait_for_done(client, job_id)
    assert status["status"] == "done"
    assert status["percent"] == 100

    results = client.get(f"/results/{job_id}")
    assert results.status_code == 200
    assert b"1 match" in results.data
    assert b"dist 0.12" in results.data


def test_progress_unknown_job_404(client):
    assert client.get("/progress/nope").status_code == 404


def test_thumb_unknown_job_404(client):
    assert client.get("/thumb/nope/0_x.png").status_code == 404


def test_download_all_zip(client, monkeypatch, tmp_path):
    img_bytes = _png_bytes()
    f1 = tmp_path / "a.png"
    f2 = tmp_path / "b.png"
    f1.write_bytes(img_bytes)
    f2.write_bytes(img_bytes)

    def fake_match(self, repo_dir, query_paths, progress=False, on_progress=None):
        if on_progress:
            on_progress(2, 2)
        return [
            MatchResult(path=f1, distance=0.1, query_index=0),
            MatchResult(path=f2, distance=0.2, query_index=0),
        ]

    monkeypatch.setattr("facematch.webapp.FaceMatcher.match", fake_match)

    data = {
        "repo_path": str(tmp_path),
        "queries": (io.BytesIO(img_bytes), "q.png"),
    }
    job_id = client.post(
        "/search", data=data, content_type="multipart/form-data"
    ).get_json()["job_id"]
    _wait_for_done(client, job_id)

    resp = client.get(f"/download/{job_id}")
    assert resp.status_code == 200
    assert resp.mimetype == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(resp.data))
    names = zf.namelist()
    assert "a.png" in names and "b.png" in names


def test_download_one(client, monkeypatch, tmp_path):
    img_bytes = _png_bytes()
    f1 = tmp_path / "solo.png"
    f1.write_bytes(img_bytes)

    def fake_match(self, repo_dir, query_paths, progress=False, on_progress=None):
        if on_progress:
            on_progress(1, 1)
        return [MatchResult(path=f1, distance=0.1, query_index=0)]

    monkeypatch.setattr("facematch.webapp.FaceMatcher.match", fake_match)
    data = {"repo_path": str(tmp_path), "queries": (io.BytesIO(img_bytes), "q.png")}
    job_id = client.post(
        "/search", data=data, content_type="multipart/form-data"
    ).get_json()["job_id"]
    _wait_for_done(client, job_id)

    resp = client.get(f"/download/{job_id}/0_solo.png")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("Content-Disposition", "")
    assert resp.data == img_bytes


def test_download_unknown_job_404(client):
    assert client.get("/download/nope").status_code == 404
    assert client.get("/download/nope/0_x.png").status_code == 404


def test_web_main_rejects_non_loopback():
    with pytest.raises(SystemExit):
        webapp.main(["--host", "0.0.0.0"])
