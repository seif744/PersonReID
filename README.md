# Person Re-ID Pipeline

This repository contains a multi-camera person re-identification prototype that combines:

- person detection with YOLO
- per-camera tracking
- crop saving for each tracked person
- appearance embedding with a ReID model
- vector storage with Qdrant
- identity assignment through a dedicated identity service

The current code is not a polished end-user application yet. It is a working research-style pipeline that can be run from the project root and produces artifacts such as crops, embeddings, annotated videos, and cross-camera reports.

## Current status

The pipeline in [main.py](main.py) now runs the following stages:

| Stage | Component | Status |
|---|---|---|
| 1-2 | Video input and frame reading | Working |
| 3 | Person detection | Working |
| 4 | Per-camera tracking | Working |
| 5 | Crop saving and annotation drawing | Working |
| 6 | ReID embedding and storage | Working |
| 7 | Identity assignment | Present, but still needs tuning and validation |
| 8 | Full clustering / production-grade identity logic | Not yet complete |

## What the repo does right now

From the project root, you can run [main.py](main.py) to process one or more video sources defined in [config.yaml](config.yaml). The pipeline will:

- load each configured video source
- detect people in each frame
- track them per camera
- save crops to the configured crops directory
- embed tracks with the ReID model
- store embeddings in a local Qdrant store
- optionally assign global IDs and write cross-camera summary reports

## Main files

| File | Purpose |
|---|---|
| [main.py](main.py) | Main entry point and orchestration for the full pipeline |
| [config.yaml](config.yaml) | Runtime configuration for sources, detection, tracking, crops, ReID, store, identity, and display |
| [src/detector.py](src/detector.py) | Detection and tracking wrapper around YOLO/ByteTrack |
| [src/reid/extractor.py](src/reid/extractor.py) | ReID embedding extraction |
| [src/reid/service.py](src/reid/service.py) | Throttled track embedding logic |
| [src/database/store.py](src/database/store.py) | Qdrant vector store wrapper |
| [src/identity/service.py](src/identity/service.py) | Identity assignment service |
| [src/video_source.py](src/video_source.py) | Frame source abstraction |
| [src/crop_saver.py](src/crop_saver.py) | Crop persistence per track |

## Setup

This repo currently expects a Python environment with the required ML and computer-vision dependencies installed manually.

The main packages used by the code are:

- `opencv-python`
- `numpy`
- `pyyaml`
- `ultralytics`
- `torch`
- `torchreid`
- `qdrant-client`

A pinned `requirements.txt` file is not present yet, so dependency installation is still a manual step.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python numpy pyyaml ultralytics torch qdrant-client
```

If you want the ReID extractor to work, the model checkpoint at [src/reid/weights/osnet_x1_0_market1501.pth](src/reid/weights/osnet_x1_0_market1501.pth) must be available.

## Configuration

Edit [config.yaml](config.yaml) before running:

- `source.videos`: add or change the input video files or RTSP sources
- `detector.*`: detection threshold and model settings
- `reid.*`: enable/disable ReID, choose the weights path, and tune the embedding interval
- `store.*`: point at a local Qdrant store or a remote Qdrant server
- `identity.*`: tune the identity threshold and related decision logic
- `display.*`: enable/disable windows and annotated video output

The current configuration points to local video files, so it is not yet fully portable out of the box.

## Run

From the repository root:

```bash
python main.py
```

Useful options:

```bash
python main.py --reset
```

The `--reset` flag clears the Qdrant store before the run so you can start from a fresh gallery.

## Outputs

A run can generate:

- per-track crops under [crops](crops)
- annotated videos such as `output_<camera>.mp4`
- a local Qdrant store under [qdrant_data](qdrant_data)
- cross-camera reports under [reports/cross_camera_matches](reports/cross_camera_matches)

## Current issues and what still needs work

The repository is functional as a prototype, but several things still need cleanup or hardening:

1. Dependency management is incomplete
   - There is no pinned requirements file yet.
   - Setup depends on manually installing the needed packages.

2. Configuration is still environment-specific
   - The default sources in [config.yaml](config.yaml) assume specific local video paths.
   - The repo would be easier to use with a sample input or a more generic example config.

3. Identity assignment needs validation
   - The identity service is present, but the matching threshold and policy still need calibration on real data.
   - Cross-camera matching quality is not yet trustworthy enough for production use.

4. Runtime behavior is still a research workflow
   - Headless mode and live-window mode both work, but the project is still optimized for local experimentation rather than a packaged deployment.
   - There is no complete evaluation suite or benchmark script yet.

5. Documentation and examples need to be kept in sync
   - Demo scripts in [src/reid](src/reid) and [src/identity](src/identity) are useful, but the repo would benefit from a clearer “quick start” and “advanced usage” split.

## Recommended next steps

If you want to turn this into a more reliable project, the next priorities should be:

- add a proper dependency file such as `requirements.txt`
- add a sample dataset or a clearly documented example input
- add smoke tests for detection, embedding, and store wiring
- improve identity threshold tuning and add evaluation metrics
- add clearer CLI flags and better error handling for missing files or missing Qdrant setup

This README now reflects the current codebase more closely than the previous version and highlights the main limitations that still need attention.
