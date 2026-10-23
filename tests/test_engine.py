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


def test_parallel_matches_serial(tmp_path: Path, monkeypatch):
    """The parallel path must produce identical results to the serial path."""
    import facematch.engine as engine

    repo = tmp_path / "repo"
    repo.mkdir()
    names = [f"img{i}.jpg" for i in range(6)]
    for n in names:
        (repo / n).write_bytes(b"x")

    query = np.zeros(128)

    # Deterministic synthetic encodings keyed by filename: even-indexed images
    # are a match (distance 0), odd-indexed are far away.
    def enc_for(name: str):
        idx = int(name.replace("img", "").replace(".jpg", ""))
        if idx % 2 == 0:
            return [np.zeros(128)]
        far = np.zeros(128)
        far[0] = 5.0
        return [far]

    # Serial path: monkeypatch the method.
    def fake_encode_image(self, path):
        return enc_for(Path(path).name)

    monkeypatch.setattr(engine.FaceMatcher, "encode_image", fake_encode_image)
    monkeypatch.setattr(
        engine.FaceMatcher, "encode_queries", lambda self, qs: [query]
    )

    serial = engine.FaceMatcher(threshold=0.6, workers=1)
    serial_results = serial.match(repo, [Path("q.jpg")], progress=False)

    # Parallel path: monkeypatch the module-level worker (runs in-process here
    # because we also force a thread-free code path via workers>1 + patched
    # _encode_file). We patch ProcessPoolExecutor to a synchronous shim so the
    # patched worker is actually used without real subprocesses.
    def fake_encode_file(path_str, model):
        encs = enc_for(Path(path_str).name)
        return path_str, [e.tolist() for e in encs], 1.0, 1

    monkeypatch.setattr(engine, "_encode_file", fake_encode_file)

    class _SyncPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args):
            class _F:
                def __init__(self, v):
                    self._v = v

                def result(self):
                    return self._v

            return _F(fn(*args))

    monkeypatch.setattr(engine, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(engine, "as_completed", lambda futs: list(futs))

    parallel = engine.FaceMatcher(threshold=0.6, workers=4)
    parallel_results = parallel.match(repo, [Path("q.jpg")], progress=False)

    assert [r.path.name for r in serial_results] == [
        r.path.name for r in parallel_results
    ]
    assert [round(r.distance, 6) for r in serial_results] == [
        round(r.distance, 6) for r in parallel_results
    ]
    # Sanity: exactly the 3 even-indexed images matched.
    assert sorted(r.path.name for r in parallel_results) == [
        "img0.jpg",
        "img2.jpg",
        "img4.jpg",
    ]


def test_progress_monotonic(tmp_path: Path, monkeypatch):
    import facematch.engine as engine

    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(4):
        (repo / f"p{i}.jpg").write_bytes(b"x")

    monkeypatch.setattr(
        engine.FaceMatcher, "encode_image", lambda self, p: [np.zeros(128)]
    )
    monkeypatch.setattr(
        engine.FaceMatcher, "encode_queries", lambda self, qs: [np.zeros(128)]
    )

    seen = []
    m = engine.FaceMatcher(threshold=0.6, workers=1)
    m.match(repo, [Path("q.jpg")], progress=False, on_progress=lambda p, t: seen.append((p, t)))

    assert seen[0] == (0, 4)
    assert seen[-1] == (4, 4)
    # processed counts never decrease
    processed = [p for p, _ in seen]
    assert processed == sorted(processed)
