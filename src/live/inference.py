"""
inference.py  --  STAGE: detect + track + embed.

Stage 1 ran this SERIALLY (one frame at a time). Stage 2 processes the cameras in
a batch CONCURRENTLY: the A100 GPU sat at ~22% utilisation because a single
Python thread issued CUDA calls one after another with CPU work (NMS, ByteTrack
update, crop prep) in between. Running one worker per camera overlaps that idle
time -- while camera A does its CPU work, camera B's kernels run -- which is what
lets us keep up with N live streams (fewer dropped frames -> richer prototypes).

ByteTrack stop-rule (FIXED): each camera keeps its OWN detector + tracker, and a
given camera's frames are processed SEQUENTIALLY on a single worker (so its
ByteTrack state is never touched by two threads). We do NOT batch detection across
cameras or touch ultralytics internals. Different cameras run on different
detector instances -> parallel-safe. The ReID extractor is SHARED across cameras,
so its forward passes are serialised by a lock (detection, the dominant cost with
separate models per camera, still overlaps freely). The frame contract into the
identity stage is unchanged, and frames are forwarded in batch order so each
camera's per-frame order is preserved for rendering.
"""

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


class InferenceStage(threading.Thread):
    def __init__(self, detectors, embedders, inference_queue, identity_queue,
                 stop_event, metrics=None, max_workers=None):
        super().__init__(name="inference", daemon=True)
        self.detectors = detectors        # {cam: PersonDetector} (own ByteTrack each)
        self.embedders = embedders        # {cam: TrackEmbedder} (share one extractor)
        self.inference_queue = inference_queue
        self.identity_queue = identity_queue
        self.stop_event = stop_event
        self.metrics = metrics
        self.frames_done = 0

        n = max(1, len(detectors))
        self.max_workers = n if not max_workers else min(int(max_workers), n)
        self._embed_lock = threading.Lock()   # guards the SHARED ReID extractor
        self._pool = None

    def run(self):
        # One reusable pool sized to the camera count. daemon threads.
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers,
                                        thread_name_prefix="infer")
        try:
            while not self.stop_event.is_set():
                batch = self.inference_queue.get(timeout=0.1)
                if not batch:
                    continue
                # Group by camera, preserving per-camera order (ByteTrack needs a
                # camera's frames in sequence, on ONE worker).
                groups = defaultdict(list)
                for frame in batch:
                    groups[frame.cam].append(frame)
                # Process cameras concurrently; wait for all before forwarding.
                futures = [self._pool.submit(self._process_group, frames)
                           for frames in groups.values()]
                for fut in futures:
                    fut.result()
                # Forward in the original batch order -> each camera's frames keep
                # their relative order downstream.
                for frame in batch:
                    self.identity_queue.put(frame)
                    self.frames_done += 1
        finally:
            self._pool.shutdown(wait=False)

    def _process_group(self, frames):
        """Process ONE camera's frames sequentially (its ByteTrack state is only
        ever touched by this single worker)."""
        for frame in frames:
            if self.stop_event.is_set():
                return
            self._process(frame)

    def _process(self, frame):
        """Attach detections (with embeddings on the fresh ones) to `frame`."""
        detector = self.detectors.get(frame.cam)
        embedder = self.embedders.get(frame.cam)
        if detector is None:
            frame.detections = []
            frame.meta["fresh_track_ids"] = set()
            return
        # Detect + track: per-camera model + ByteTrack state (persist=True inside).
        # Separate model instances per camera -> safe to run concurrently.
        detections = detector.track(frame.image)
        if embedder is not None:
            # The extractor is shared across cameras; serialise its forward pass.
            with self._embed_lock:
                embedder.process(frame.image, detections, frame.frame_index)
                fresh = {d.track_id for d in embedder.last_embedded
                         if d.track_id is not None}
            frame.meta["fresh_track_ids"] = fresh
        else:
            frame.meta["fresh_track_ids"] = set()
        frame.detections = detections
