import zipfile
from pathlib import Path

import pytest

from facematch.sources import is_drive_url, list_images, resolve_repository


def test_is_drive_url():
    assert is_drive_url("https://drive.google.com/drive/folders/abc")
    assert is_drive_url("https://docs.google.com/foo")
    assert not is_drive_url("./local/folder")
    assert not is_drive_url("https://example.com/photos")


def test_list_images_filters_and_sorts(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("nope")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpeg").write_bytes(b"x")

    images = list_images(tmp_path)
    names = [p.name for p in images]
    assert names == ["a.jpg", "b.PNG", "c.jpeg"]


def test_resolve_folder(tmp_path: Path):
    assert resolve_repository(str(tmp_path)) == tmp_path


def test_resolve_zip(tmp_path: Path):
    zip_path = tmp_path / "album.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("photo.jpg", b"data")
    out = resolve_repository(str(zip_path), workdir=tmp_path / "work")
    assert (out / "photo.jpg").is_file()


def test_resolve_zip_slip_rejected(tmp_path: Path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.jpg", b"data")
    with pytest.raises(ValueError):
        resolve_repository(str(zip_path), workdir=tmp_path / "work")


def test_resolve_unknown(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_repository(str(tmp_path / "does_not_exist"))
