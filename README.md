# FaceMatch

Find every photo of a given person inside a repository of images.

FaceMatch takes a **repository of images** (a local folder, a `.zip`, or a Google
Drive folder) and one or more **query faces**, detects every face in the
repository, and returns the images that contain a matching face — ranked by
similarity and optionally copied into an output folder.

Everything runs **locally and offline**. No image ever leaves your machine. The
whole stack is open source.

> **Responsible use.** Only run FaceMatch on images you own or are authorized to
> process (your own photo library, an event you organized, consenting subjects).
> Do not use it to identify, track, or profile people without their consent.
> See [`NOTICE.md`](NOTICE.md).

---

## Features

- **Multiple input sources**: local folder, `.zip` archive, or Google Drive
  folder (public link via `gdown`, or a folder you already downloaded).
- **Multiple query faces**: pass several reference images; an image matches if it
  contains any of them.
- **On-disk embedding cache**: the repository is embedded once; re-runs against
  new query faces are fast.
- **Tunable threshold**: control how strict a match must be.
- **Results as data or files**: prints a ranked report and can copy matches into
  an output directory.
- Pure open-source stack ([`face_recognition`](https://github.com/ageitgey/face_recognition) / dlib, MIT).

## Install

Requires Python 3.8+. `face_recognition` depends on `dlib`, which needs a C++
toolchain and CMake.

```bash
# macOS
brew install cmake

# Debian/Ubuntu
sudo apt-get install -y build-essential cmake

git clone https://github.com/rajarshi-2000/FaceMatch.git
cd FaceMatch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Local folder
facematch --repo ./my_photos --query ./alice.jpg --out ./matches

# Zip archive, two query faces, stricter threshold
facematch --repo ./album.zip --query alice.jpg --query bob.jpg \
          --threshold 0.55 --out ./matches

# Public Google Drive folder
facematch --repo "https://drive.google.com/drive/folders/XXXXXXXX" \
          --query ./alice.jpg --out ./matches

# JSON report, no file copy
facematch --repo ./my_photos --query ./alice.jpg --json
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--repo` | Image repository: folder path, `.zip`, or Google Drive folder URL | required |
| `--query` | Reference face image (repeatable) | required |
| `--out` | Copy matching images into this folder | none |
| `--threshold` | Max face distance to count as a match (lower = stricter) | `0.6` |
| `--model` | dlib detector: `hog` (fast, CPU) or `cnn` (accurate, GPU) | `hog` |
| `--cache` | Embedding cache file | `.facematch_cache.json` |
| `--no-cache` | Disable the embedding cache | off |
| `--json` | Emit the report as JSON | off |
| `--workers` | Parallel worker processes for embedding | CPU count |

## How it works

1. **Resolve the repository** to a local folder (unzip / download as needed).
2. **Detect & embed** every face in every image into a 128-d vector, cached on disk.
3. **Embed the query faces** the same way.
4. **Compare** each repository face to each query face by Euclidean distance; an
   image matches if its closest face is within `--threshold`.
5. **Report / copy** the matches, ranked by best (smallest) distance.

The `0.6` default threshold is dlib's recommended operating point. Lower it
(e.g. `0.5`) for fewer false positives, raise it for higher recall.

## Web UI

A small local web UI is included for upload-and-browse workflows. It binds to
`127.0.0.1` only (it processes biometric data and must not be exposed on a
network).

```bash
pip install -e ".[web]"   # adds Flask
facematch-web             # serves on http://127.0.0.1:5000
```

Then open http://127.0.0.1:5000, upload a `.zip` of images (or enter a local
folder path), upload one or more query face images, and view the matches as a
thumbnail grid with distance scores.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE).
