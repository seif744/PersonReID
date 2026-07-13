# Multi-Camera Person Re-Identification

Detect and track people across multiple camera videos, embed their appearance,
and assign a **global ID** that stays the same for one real person **across
cameras** and **across re-appearances**. Produces annotated videos.

> **How it works internally:** see **[ARCHITECTURE.md](ARCHITECTURE.md)** for the
> full data flow, every component, the concurrency model, and design rationale.
> This README is the *get-it-running* guide.

---

## Table of contents
1. [Prerequisites](#1-prerequisites)
2. [Install the Python environment](#2-install-the-python-environment)
3. [Model weights](#3-model-weights)
4. [Set up Qdrant (the vector store)](#4-set-up-qdrant-the-vector-store)
5. [Configure your run](#5-configure-your-run)
6. [Run the pipeline](#6-run-the-pipeline)
7. [Understand the output](#7-understand-the-output)
8. [Troubleshooting](#8-troubleshooting)
9. [Project layout](#9-project-layout)
10. [Known limitations & roadmap](#10-known-limitations--roadmap)

---

## 1. Prerequisites

- **Python 3.10** (the project is verified on 3.10.12).
- **Docker** + **Docker Compose** — used to run the Qdrant vector database.
  Install Docker Desktop (Windows/macOS) or Docker Engine (Linux). Verify:
  ```bash
  docker --version
  docker compose version
  ```
- **One or more video files** (e.g. `.avi`, `.mp4`). RTSP URLs also work.
  CPU-only is fine — the whole pipeline runs on CPU.

---

## 2. Install the Python environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the exact verified versions (ultralytics, torch,
torchvision, torchreid, qdrant-client, numpy, opencv-python, PyYAML).

> **CPU vs GPU torch:** `requirements.txt` pins CPU-compatible `torch`/
> `torchvision`. For a specific CUDA build, install torch from the official
> PyTorch wheel index for your platform *before* `-r requirements.txt`.

---

## 3. Model weights

Two model files are needed:

| File | How to get it | In a fresh clone? |
|---|---|---|
| `yolo11n.pt` (detector) | **Auto-downloaded** by ultralytics on first run | No (gitignored; fetched automatically) |
| `src/reid/weights/osnet_x1_0_market1501.pth` (ReID) | **Committed to the repo** | Yes — already present |

The OSNet checkpoint is committed, so a fresh clone already has it — no download
step. (If it is ever missing, get `osnet_x1_0_market1501.pth` from the torchreid
model zoo and place it at `src/reid/weights/`.) The path is set by `reid.weights`
in `config.yaml`.

---

## 4. Set up Qdrant (the vector store)

Embeddings are stored in **Qdrant**. The recommended setup is the Dockerized
server (supports concurrent writes from all camera threads; the embedded mode
locks to one process).

### 4a. Start the Qdrant server (recommended)

A `docker-compose.yml` is included. From the project root:

```bash
docker compose up -d          # start Qdrant in the background
docker compose logs -f        # (optional) watch logs; Ctrl-C to stop watching
```

This launches `qdrant/qdrant:latest` and exposes:
- **REST API + dashboard:** http://localhost:6333/dashboard
- **gRPC (optional):** `localhost:6334`

Data persists in `./qdrant_storage/` (gitignored), so it survives restarts.

Verify it's up:
```bash
curl http://localhost:6333/readyz        # -> "all shards are ready"
```

Stop it later with `docker compose down` (your data is kept).

### 4b. Tell the pipeline where Qdrant is

The store backend is chosen with this precedence: **`QDRANT_URL` env var → 
`store.url` in config.yaml → `store.path` (embedded fallback)**.

Out of the box, `config.yaml` already has:
```yaml
store:
  url: http://localhost:6333
```
so no extra step is needed for local Docker. If you prefer env-based config,
create a `.env` file in the project root:
```
QDRANT_URL=http://localhost:6333
```

For **Qdrant Cloud** instead of local Docker, set both (the API key is a secret —
`.env` is gitignored, never commit it):
```
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-key>
```

### 4c. Alternative: embedded mode (no Docker)

For a quick single-process run without Docker, clear the URL so it falls back to
an on-disk embedded store:
```yaml
store:
  url:                 # leave blank
  path: qdrant_data    # local folder, created automatically
```
Note: embedded mode locks the folder to one process and is dev-only.

---

## 5. Configure your run

> **Runs out of the box:** the repo ships with two sample videos and the default
> `config.yaml` already points at them. With Qdrant running (section 4),
> `python main.py` works with no config changes. Edit `source.videos` below to
> process your own footage.

Edit [config.yaml](config.yaml). The key sections:

```yaml
source:
  videos:                        # one entry per camera
    - name: cam_219
      path: your_camera_1.avi
    - name: cam_224
      path: your_camera_2.avi
  max_frames: 0                  # 0 = whole video; N = stop early (quick test)
  resize_width: 0                # 0 = native; e.g. 1280 = faster on CPU

detector:
  model: yolo11n.pt              # auto-downloaded
  confidence_threshold: 0.4

tracker:
  enabled: true
  config: bytetrack.yaml


reid:
  enabled: true
  weights: src/reid/weights/osnet_x1_0_market1501.pth
  device: cpu                    # "cuda" if you have a GPU
  interval: 10                   # re-embed a track at most every N frames

store:
  enabled: true
  url: http://localhost:6333     # Qdrant server (see section 4)

identity:
  enabled: true
  threshold: 0.85                # cosine to accept a live match
  reconcile:                     # offline cross-camera pass (runs after all cams)
    enabled: true
    same_camera_threshold: 0.90
    min_tracklet_observations: 3
    require_reciprocal_best: true

display:
  show_window: false             # true = live OpenCV windows; false = headless
  save_annotated: true           # write output_<camera>.mp4
```

See [ARCHITECTURE.md §7](ARCHITECTURE.md) for what every knob does.

---

## 6. Run the pipeline

With the venv active and Qdrant running:

```bash
python main.py                       # uses the videos in config.yaml (the samples)
```

### Run on your OWN videos (dynamic input — no config edits)

Point the pipeline at any videos from the command line; this **overrides**
`config.yaml`, so you never have to hardcode paths:

```bash
# one or more explicit files (camera name = the file's base name):
python main.py --videos /path/cam_a.mp4 /path/cam_b.mp4

# every video in a folder (.mp4/.avi/.mov/.mkv/...):
python main.py --videos-dir /path/to/footage

```

Precedence: `--videos` > `--videos-dir` > `config.yaml source.videos`. Missing
files fail fast with a clear message; duplicate names are made unique
automatically.

Start from a clean gallery (wipes the store first) — combine with any input:

```bash
python main.py --reset --videos-dir /path/to/footage
```

Headless (`display.show_window: false`) runs to completion and prints a summary.
With windows enabled, press `q` in a window to stop early.

---

## 7. Understand the output

A run produces:

| Artifact | Location | What it is |
|---|---|---|

| Annotated videos | `output_<camera>.mp4` | source video with boxes + `GID n  IDk` labels drawn |
| Vector store | Qdrant (`qdrant_storage/` or `qdrant_data/`) | every embedding + metadata |



The console prints a **RUN SUMMARY** at the end, e.g.:
```
Store: 516 observations -> 11 distinct people (global_ids)
Cross-camera people: 1
  GID 1: cam_219 (track 0004 + 0025) + cam_224 (track 0001)
```
- **distinct people** = number of global IDs after reconciliation.
- **Cross-camera people** = global IDs seen in more than one camera (the product's
  core result).

The annotated `output_<camera>.mp4` videos are drawn in a **second pass that runs
after cross-camera reconciliation**, so the labels use the *final* global IDs.
This means a person who walks from one camera into another shows the **same
`GID n`** (and the same box colour) in **both** videos — the visual proof the
cameras are linked. Each box is labelled `GID n  IDk` (`n` = cross-camera global
ID, `k` = that camera's local track ID). In the run above, `cam_219` track 1 and
`cam_224` track 7 are the same person, so both videos label them `GID 1`.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Connection refused` / store errors | Qdrant isn't running. `docker compose up -d`, then `curl http://localhost:6333/readyz`. |
| `Unexpected checkpoint keys dropped` | Wrong/corrupt ReID weights. Re-fetch `osnet_x1_0_market1501.pth` to `src/reid/weights/`. |
| Hangs on first run for a while | `yolo11n.pt` is downloading; subsequent runs are fast. |
| Very slow on CPU | Set `source.resize_width: 1280` and/or `source.max_frames` for tests. |
| "database is locked" | Embedded mode (`store.path`) is single-process. Use the Docker server (`store.url`). |
| No windows appear | `display.show_window: false` (headless). Set `true` for live windows. |
| Counts look too high / people split | Expected on out-of-domain footage — see limitations below. |

---

## 9. Project layout

```
main.py                     entry point + per-camera orchestration
config.yaml                 all runtime configuration
requirements.txt            pinned dependencies
docker-compose.yml          Qdrant server
ARCHITECTURE.md             deep-dive: data flow, components, design
src/
  video_source.py           frame decoding
  detector.py               YOLO11 + ByteTrack, Detection, crop_person
  crop_saver.py             per-track crop persistence
  drawing.py                boxes / HUD overlay
  reid/
    extractor.py            crop -> 512-d L2-normalized embedding (OSNet)
    service.py              TrackEmbedder: throttle, cache, quality + occlusion gates
    weights/                ReID model checkpoint
  database/
    store.py                Qdrant wrapper (PersonVectorStore)
  identity/
    service.py              live global-ID assignment (candidate->decide->commit)
    reconcile.py            offline cross-camera reconciliation
    DESIGN.md               why the layers are separated
tools/
  prepare_randperson_subset.py   dataset prep for the model-fix path
notebooks/                  training/experiment notebooks (incl. RandPerson)
```

Generated at runtime (gitignored): `output_*.mp4`,`qdrant_storage/`, `qdrant_data/`.

---
