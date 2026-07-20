# Multi-Camera Person Re-Identification

Detect and track people across multiple camera videos, embed their appearance,
and assign a **reid id** that stays the same for one real person **across
cameras** and **across re-appearances**. The pipeline is fully self-contained —
it builds and owns its own Qdrant gallery; there is no separate registration
step or external service. Produces annotated videos.

> **How it works internally:** see **[ARCHITECTURE.md](ARCHITECTURE.md)** for the
> full data flow, every component, the concurrency model, and design rationale.


---

## How it flows

```mermaid
flowchart TD
    CFG(["⚙️ config.yaml / CLI"]) --> A1 & B1

    subgraph LIVE["🎥 LIVE — one thread per camera, running at the same time"]
        direction LR
        subgraph CAM_A["Camera A"]
            direction TB
            A1(["Video frame"]) --> A2["Detect + Track\n(YOLO11n + ByteTrack)"]
            A2 --> A3["Embed crop\n(OSNet, 512-d)"]
            A3 --> A4["Re-rank + verify\ncandidates -> assign\na reid id"]
        end
        subgraph CAM_B["Camera B"]
            direction TB
            B1(["Video frame"]) --> B2["Detect + Track\n(YOLO11n + ByteTrack)"]
            B2 --> B3["Embed crop\n(OSNet, 512-d)"]
            B3 --> B4["Re-rank + verify\ncandidates -> assign\na reid id"]
        end
    end

    A4 --> QD
    B4 --> QD

    QD[("Qdrant\nown gallery\n(all cameras write here)")]

    QD --> RECON

    subgraph FINAL["🏁 OFFLINE — runs ONCE, after every camera finishes"]
        direction TB
        RECON["Reconcile tracklets\nlink the same person\nacross cameras"]
        RECON --> RENDER["Re-render videos\nusing the FINAL\nreid ids"]
        RECON --> SUMMARY["Print run summary\n(console only)"]
    end

    RENDER --> OUT[("🎬 output_camA.mp4\n🎬 output_camB.mp4\nsame person = same REID\nin BOTH videos")]

    style CFG fill:#e8eef7,stroke:#4a6fa5,color:#1a1a1a
    style QD fill:#fdf3d7,stroke:#b8860b,color:#1a1a1a,stroke-width:2px
    style OUT fill:#e3f5e6,stroke:#2e7d32,color:#1a1a1a,stroke-width:2px
    style LIVE fill:#f7f9fc,stroke:#4a6fa5
    style FINAL fill:#f2f8f3,stroke:#2e7d32
    style CAM_A fill:#ffffff,stroke:#9fb3cf
    style CAM_B fill:#ffffff,stroke:#9fb3cf
```

Each camera runs the **top** section live, on its own thread, against ONE
shared Qdrant gallery that this pipeline itself writes to and reads from —
no other service is involved. Only after **every** camera finishes does the
**bottom** section run — once: it links the same person across cameras, then
re-renders the annotated videos with those final IDs, so one person carries
the same `REID n` in every camera's output. `global_id` is still written for
compatibility, but `reid_id` is the label the pipeline presents. Full per-stage detail:
**[ARCHITECTURE.md §2](ARCHITECTURE.md)**.

---

## Table of contents
1. [Prerequisites](#1-prerequisites)
2. [Install the Python environment](#2-install-the-python-environment)
3. [Model weights](#3-model-weights)
4. [Connect to Qdrant](#4-connect-to-qdrant)
5. [Configure your run](#5-configure-your-run)
6. [Run the pipeline](#6-run-the-pipeline)
7. [Understand the output](#7-understand-the-output)
8. [Troubleshooting](#8-troubleshooting)
9. [Project layout](#9-project-layout)
10. [Known limitations & roadmap](#10-known-limitations--roadmap)

---

## 1. Prerequisites

- **Python 3.10** (the project is verified on 3.10.12).
- **Docker** + **Docker Compose** — used to run the Qdrant server this
  pipeline writes its own gallery to. Install Docker Desktop (Windows/macOS)
  or Docker Engine (Linux). Verify:
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
| `src/reid/weights/osnet_x1_0_msmt17.pth` (ReID, default) | Committed to the repo | Yes — already present |

The default ReID checkpoint is OSNet x1_0 trained on **MSMT17** (set by
`reid.weights` in `config.yaml`), not the more commonly-referenced Market1501
checkpoint. On this project's own footage, MSMT17 measurably separates
same-person from different-person matches better — Market1501 gave same- and
different-person cosine scores that overlapped in the same 0.70-0.85 range,
while MSMT17 keeps different people under ~0.57 and genuine matches at
0.70-0.80 (see [ARCHITECTURE.md §6](ARCHITECTURE.md)). If you ever need to
re-fetch it: it comes from the official `deep-person-reid` MODEL_ZOO
(`osnet_x1_0`, trained on MSMT17) — download via `gdown` from the Google Drive
link on that page and place it at `src/reid/weights/`.

`src/reid/weights/osnet_x1_0_market1501.pth` is also still present (the
original default) — swap `reid.weights` back to it if you want to compare, but
it is **not recommended** for this project's footage given the measured overlap
above.

---

## 4. Connect to Qdrant

The pipeline needs somewhere to store its own gallery of embeddings + global
ids. It both writes to and reads from this store — nothing else does.

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
If you already run Registry against another Qdrant endpoint, point Inference at
that same server instead.

Verify it's up:
```bash
curl http://localhost:6333/readyz        # -> "all shards are ready"
```

Stop it later with `docker compose down` (your data is kept).

### 4b. Tell the pipeline where Qdrant is

Backend precedence: **`QDRANT_URL` env var → `store.url` in config.yaml →
`store.path`** (a local embedded folder, single-process only).

Out of the box, `config.yaml` already has:
```yaml
store:
  enabled: true
  url: http://localhost:6333

store:
  enabled: false
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

### 4c. Embedded mode (no Docker, single process only)

```yaml
store:
  enabled: true
  url:                 # leave blank
  path: qdrant_data    # local folder, created automatically
```
This locks the folder to one process — fine for a quick local test, not for
multiple concurrent runs.

---

## 5. Configure your run

> **You must provide your own video files.** Sample footage is **not** shipped in
> the repo (videos are gitignored). Either drop your files in and point
> `source.videos` in `config.yaml` at them, or pass them on the command line with
> `--videos` / `--videos-dir` (see section 6). The default `config.yaml` lists two
> example filenames purely as a template — edit them to your paths.

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
  weights: src/reid/weights/osnet_x1_0_msmt17.pth
  device: cpu                    # "cuda" if you have a GPU
  interval: 10                   # re-embed a track at most every N frames

registry:
  enabled: true
  url: http://localhost:6333     # shared Qdrant server (see section 4)

identity:
  enabled: true
  threshold: 0.63                # plain-path cosine acceptance (used only
                                  # when verification.enabled is false) --
                                  # calibrated for the MSMT17 checkpoint
  min_score_gap: 0.03
  rerank:                        # ADR-002 Upgrade 1: camera-aware re-ranking
    enabled: true
    k1: 8
    cross_camera_k1_boost: 4
    lambda_: 0.7
  verification:                  # ADR-002 Upgrade 2: scored verification layer
    enabled: true
    accept_threshold: 0.55
    min_prob_gap: 0.05
    log_path: logs/verification_decisions.jsonl
  reconcile:
    enabled: true
    threshold: 0.63               # cross-camera merge bar -- independent of the
                                   # verifier above; recalibrate both together
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

With the venv active and Qdrant reachable:

```bash
python main.py                       # uses the videos listed in config.yaml
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

`--reset` wipes the store's collection first, so a run starts from an empty
gallery. **Destructive** — deletes every stored embedding/identity:

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
| Annotated videos | `output_<camera>.mp4` | source video with boxes + `REID n  IDk` labels drawn |
| Gallery | Qdrant (`store.url` / `store.path`) | every observation + assigned reid id (plus compatibility `global_id`) — this pipeline's own data |
| Verification decision log | `logs/verification_decisions.jsonl` | one line per accept/reject decision, with its full feature vector — for calibrating or eventually training the verifier |

That's it — no crop images by default. (Per-person crop images can be turned
back on with `crops.save: true` in `config.yaml`, e.g. to collect ReID training
data or visually sanity-check identity assignment.)

Registry lookups do not change the gallery contents during inference. The
console still prints a **RUN SUMMARY** at the end for the legacy identity path,
e.g.:
```
Store: 516 observations -> 11 distinct people (reid_ids)
Cross-camera people: 1
  REID 1: cam_219 (track 0004 + 0025) + cam_224 (track 0001)
```
- **distinct people** = number of reid IDs after reconciliation.
- **Cross-camera people** = reid IDs seen in more than one camera (the product's
  core result).

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Connection refused` | Qdrant isn't running. `docker compose up -d`, then `curl http://localhost:6333/readyz`. |
| `Unexpected checkpoint keys dropped` | Wrong/corrupt ReID weights. Re-fetch `osnet_x1_0_msmt17.pth` to `src/reid/weights/`. |
| Hangs on first run for a while | `yolo11n.pt` is downloading; subsequent runs are fast. |
| Very slow on CPU | Set `source.resize_width: 1280` and/or `source.max_frames` for tests. |
| "database is locked" | Embedded mode (`store.path`) is single-process. Use the shared Docker server (`store.url`), or stop other runs using the same folder. |
| No windows appear | `display.show_window: false` (headless). Set `true` for live windows. |
| Same person shows up as two different reid ids | Expected on out-of-domain footage when cosine similarity is borderline — see limitations below. Check `logs/verification_decisions.jsonl` for that decision's actual scores. |
| Two different people merge into one reid id | The rarer, worse failure. If it happens often, tighten `identity.verification.accept_threshold` / `min_prob_gap` (or `identity.threshold` / `min_score_gap` if verification is disabled). |

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
  detector.py                YOLO11 + ByteTrack, Detection, crop_person
  crop_saver.py               per-track crop persistence (only when crops.save: true)
  drawing.py                  boxes / HUD overlay
  reid/
    extractor.py              crop -> 512-d L2-normalized embedding (OSNet)
    service.py                 TrackEmbedder: throttle, cache, quality + occlusion gates
    weights/                    ReID model checkpoint
  database/
    store.py                   Qdrant wrapper (PersonVectorStore) -- this pipeline's own gallery
  identity/
    service.py                  global-ID assignment (candidate -> re-rank -> verify -> decide -> commit)
    reranking.py                 ADR-002 Upgrade 1: camera-aware k-reciprocal + Jaccard re-ranking
    verifier.py                  ADR-002 Upgrade 2: scored verification layer + decision logging
    reconcile.py                 offline cross-camera reconciliation
    DESIGN.md                    why the layers are separated
```

Generated at runtime (gitignored): `output_*.mp4`, `logs/`, `qdrant_storage/`,
`qdrant_data/`.

---

## 10. Known limitations & roadmap

The pipeline is correct; the embedding model is the ceiling on this domain —
see **[ARCHITECTURE.md §6](ARCHITECTURE.md)** for the measured same-vs-different
score overlap and why no threshold (or hand-tuned heuristic on top of it) can
perfectly resolve it.

ADR-002's P0 items (camera-aware re-ranking, a scored verification layer) are
implemented; the following are deferred and documented but not built:
- Prototype confidence/variance with adaptive per-identity thresholds.
- A camera transition graph rejecting physically-impossible transitions.
- A MetaBIN training pass to reduce the underlying domain gap.
- Training a real classifier from `logs/verification_decisions.jsonl` once
  enough runs accumulate, to replace the current hand-set verifier weights.
