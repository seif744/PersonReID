"""
demo_track_reid.py  --  PHASE 6:  ByteTrack  ->  embeddings attached to tracks.

============================ WHAT THIS SHOWS ==============================
    frame --(PersonDetector.track)--> detections WITH stable track_id
          --(TrackEmbedder.process)--> same detections WITH det.embedding

Two things to observe in the output:
  1. track_ids PERSIST across frames (that's ByteTrack working).
  2. Most frames do FEW forward passes: TrackEmbedder recomputes each track's
     embedding only every `interval` frames and reuses the cache otherwise. Watch
     "embedded" (forward passes) stay low while "tracks" stays high.

And a correctness check: each time a track IS re-embedded, its new vector should
still be close to its previous vector (same person over time) -- we print the
cosine between consecutive recomputes per track.

NO identity assignment here (det.global_id stays None). That's Stage 7.

Run from src/:   python -m reid.demo_track_reid
============================================================================
"""

import cv2
import numpy as np

from detector import PersonDetector
from reid.extractor import ReIDExtractor
from reid.service import TrackEmbedder

VIDEO = "../output_cam_219.mp4"
INTERVAL = 5          # recompute each track's embedding every 5 frames
MAX_FRAMES = 60       # cap the demo


def main():
    detector = PersonDetector(model_path="../yolo11n.pt")
    extractor = ReIDExtractor(weights="reid/weights/osnet_x1_0_market1501.pth")
    embedder = TrackEmbedder(extractor, interval=INTERVAL)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    prev_vec = {}       # track_id -> last recomputed embedding (for drift check)
    frame_index = -1

    while frame_index + 1 < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        # track() (not detect()) -> detections carry a persistent track_id.
        detections = detector.track(frame)
        # Attach embeddings, throttled + cached, in place.
        detections = embedder.process(frame, detections, frame_index)

        tracked = [d for d in detections if d.track_id is not None]
        ids = sorted(d.track_id for d in tracked)
        print(f"frame {frame_index:3d}: tracks={len(tracked)} ids={ids}  "
              f"forward-passes={embedder.last_num_embedded}  "
              f"cache={len(embedder._cache)}")

        # Drift check: when a track was recomputed this frame, compare to its
        # previous recompute. High cosine = stable identity over time.
        for d in tracked:
            if d.embedding is None:
                continue
            if d.track_id in prev_vec and embedder.last_num_embedded > 0:
                sim = float(prev_vec[d.track_id] @ d.embedding)
                # Only meaningful when this track was among those recomputed;
                # a reused (cached) vector would give sim exactly 1.0.
                if sim < 0.9999:
                    print(f"      track {d.track_id}: self-cosine vs previous "
                          f"recompute = {sim:.3f}")
            prev_vec[d.track_id] = d.embedding

    cap.release()
    print("\nDone. Note global_id is still None on every detection -- identity "
          "assignment is Stage 7, not this module.")


if __name__ == "__main__":
    main()
