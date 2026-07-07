# Person Re-ID Pipeline

A step-by-step **person re-identification** system: detect people across one or
more cameras and (eventually) recognise the *same* person across different
cameras and times.

This repo is being built **one stage at a time** so every part is understandable.

## The full concept (the end goal)

```
Camera(N) --RTSP--> Frame --YOLO--> bbox + classid --> Track ID --> Draw
                                                                       |
                                                                  Embedding
                                                                       |
                                                             Cross-camera match
                                                                       |
                                                              Cluster = identity
```

| Stage | Name | Status |
|------:|------|--------|
| 1-2 | RTSP/file -> frames (OpenCV) | built |
| 3 | Person detection -> bbox + classid (YOLO) | built |
| 5 | Draw boxes + labels | built |
| 4 | Track IDs (per-camera) | next |
| 6 | Appearance embeddings | later |
| 7 | Cross-camera matching | later |
| 8 | Clustering -> global identity | later |

## What's built right now (stages 1 -> 3)

Read from an RTSP camera (or a video file), find every person in each frame,
draw a green box + confidence label around them, and show a live window.

## Files

| File | Role |
|------|------|
| [config.yaml](config.yaml) | All settings: camera URL, model, thresholds |
| [src/video_source.py](src/video_source.py) | Stage 1-2: stream -> frames |
| [src/detector.py](src/detector.py) | Stage 3: frame -> person boxes |
| [src/drawing.py](src/drawing.py) | Stage 5: draw boxes/labels |
| [main.py](main.py) | Wires it all together; run this |

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

1. Open [config.yaml](config.yaml) and set `source.url`:
   - RTSP camera: `rtsp://user:pass@192.168.1.50:554/stream1`
   - or a local file: `sample.mp4`
   - or `"0"` to use your laptop webcam for a quick test.
2. Run:
   ```bash
   python main.py
   ```
3. A window opens showing detected people. Press **q** to quit.

The first run downloads the small YOLO model (`yolo11n.pt`, ~5 MB) automatically.
