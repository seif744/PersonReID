"""
inference.py  --  STAGE: detect + track + embed (Stage 1 = per-camera, bs=1).

Pulls batches from the scheduler and, for each frame, runs that camera's
detection+tracking and ReID embedding, attaching the results to the frame before
handing it to the identity stage.

Stage 1 is the SKELETON: a single inference thread processes frames one at a
time, reusing the proven per-camera `PersonDetector` (its own ByteTrack state)
and `TrackEmbedder` (crop-quality gate + throttle + batched embed). Because it is
single-threaded, no model lock is needed. Stage 2 replaces the per-frame loop
with a real batched detector+embed across cameras while KEEPING one tracker per
camera (ByteTrack stop-rule) -- the frame contract here does not change.

Device portability: the extractor is built on the resolved device; detection runs
through ultralytics with `device=`. Nothing hard-codes CUDA.
"""

import threading


class InferenceStage(threading.Thread):
    def __init__(self, detectors, embedders, inference_queue, identity_queue,
                 stop_event, metrics=None):
        super().__init__(name="inference", daemon=True)
        self.detectors = detectors        # {cam: PersonDetector}
        self.embedders = embedders        # {cam: TrackEmbedder} (share one extractor)
        self.inference_queue = inference_queue
        self.identity_queue = identity_queue
        self.stop_event = stop_event
        self.metrics = metrics
        self.frames_done = 0

    def run(self):
        while not self.stop_event.is_set():
            batch = self.inference_queue.get(timeout=0.1)
            if not batch:
                continue
            for frame in batch:
                if self.stop_event.is_set():
                    break
                self._process(frame)
                self.identity_queue.put(frame)
                self.frames_done += 1

    def _process(self, frame):
        """Attach detections (with embeddings on the fresh ones) to `frame`."""
        detector = self.detectors.get(frame.cam)
        embedder = self.embedders.get(frame.cam)
        if detector is None:
            frame.detections = []
            return
        # Detect + track (per-camera ByteTrack state, persist=True inside).
        detections = detector.track(frame.image)
        # Attach embeddings: TrackEmbedder sets det.embedding (+ det.crop_quality)
        # on the tracks it (re-)embeds this frame; others keep the cached vector.
        if embedder is not None:
            embedder.process(frame.image, detections, frame.frame_index)
            frame.meta["fresh_track_ids"] = {
                d.track_id for d in embedder.last_embedded if d.track_id is not None
            }
        else:
            frame.meta["fresh_track_ids"] = set()
        frame.detections = detections
