from pathlib import Path

import numpy as np

from facematch.engine import FaceMatcher, MatchResult


def _write_fake_images(folder: Path, names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"not-a-real-image")


def test_match_uses_injected_encodings(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    _write_fake_images(repo, ["alice.jpg", "stranger.jpg", "noface.jpg"])

    # Query encoding is a simple unit vector.
    query = np.zeros(128)
    query[0] = 1.0

    # Fake per-image encodings: alice is identical to the query (distance 0),
    # stranger is far away, noface has no faces.
    fake = {
        repo / "alice.jpg": [query.copy()],
        repo / "stranger.jpg": [np.ones(128)],
        repo / "noface.jpg": [],
    }

    matcher = FaceMatcher(threshold=0.6)
    monkeypatch.setattr(matcher, "encode_image", lambda p: fake[p])
    monkeypatch.setattr(matcher, "encode_queries", lambda qs: [query])

    results = matcher.match(repo, [Path("ignored.jpg")], progress=False)

    assert [r.path.name for r in results] == ["alice.jpg"]
    assert results[0].distance == 0.0
    assert results[0].query_index == 0


def test_threshold_excludes_distant_faces(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    _write_fake_images(repo, ["x.jpg"])
    query = np.zeros(128)

    far = np.zeros(128)
    far[0] = 10.0  # distance 10 > any sane threshold

    matcher = FaceMatcher(threshold=0.6)
    monkeypatch.setattr(matcher, "encode_image", lambda p: [far])
    monkeypatch.setattr(matcher, "encode_queries", lambda qs: [query])

    results = matcher.match(repo, [Path("ignored.jpg")], progress=False)
    assert results == []


def test_cache_round_trip(tmp_path: Path):
    cache = tmp_path / "cache.json"
    img = tmp_path / "pic.jpg"
    img.write_bytes(b"data")

    m1 = FaceMatcher(cache_path=cache)
    stat = img.stat()
    from facematch.engine import _ImageEncodings

    m1._cache[str(img)] = _ImageEncodings(
        encodings=[np.arange(128, dtype=float)],
        mtime=stat.st_mtime,
        size=stat.st_size,
    )
    m1.save_cache()
    assert cache.exists()

    m2 = FaceMatcher(cache_path=cache)
    m2.load_cache()
    assert m2._cache_valid(img, m2._cache[str(img)])
    np.testing.assert_array_equal(
        m2._cache[str(img)].encodings[0], np.arange(128, dtype=float)
    )


def test_match_result_as_dict():
    r = MatchResult(path=Path("/tmp/a.jpg"), distance=0.12345, query_index=1)
    d = r.as_dict()
    assert d == {"path": "/tmp/a.jpg", "distance": 0.1235, "query_index": 1}
