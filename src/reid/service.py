"""
service.py  --  STAGE 6:  attach ReID embeddings to TRACKS (throttled).

============================ WHAT THIS DOES ================================
Given a frame and the tracked detections for it (each already carrying a
ByteTrack track_id from the detector), attach a 512-d appearance embedding to
each detection:  det.embedding = <unit vector>.

That is the WHOLE job. TrackEmbedder deliberately does NOT:
  * assign global ids            (that's the Identity Service, Stage 7)
  * search a gallery / Qdrant    (Stage 7)
  * merge tracks across cameras  (Stage 7)
  * average / bank features      (a gallery concern, Stage 7)
The name is "TrackEmbedder", not "ReIDService", precisely so nobody is tempted
to grow identity logic in here. An embedding is a measurement; deciding WHO it
is belongs to a separate, stateful service with its own constraints.

------------------------------- THROTTLING --------------------------------
Re-embedding every track on every frame is wasteful: a person's appearance
barely changes between consecutive frames, and each forward pass costs real CPU
time (the deployment target has hundreds of cameras). So per track we recompute
the embedding only every `interval` frames and REUSE the cached vector in
between. Tracks that are due this frame are batched into ONE extract_batch call.

`interval` and `ttl` are in FRAMES, not seconds -- the service has no clock and
no notion of fps (keeping it dependency-free). At 25 fps, interval=10 ~= every
0.4 s, ttl=300 ~= 12 s. Tune per deployment.

-------------------------------- CACHE / TTL ------------------------------
The per-track cache would grow without bound in a long-running system as
ByteTrack mints new ids, so entries not refreshed within `ttl` frames (i.e. the
track has left) are evicted. This keeps memory flat over days/weeks of uptime.
============================================================================
"""

import cv2
import numpy as np

from detector import crop_person   # shared "box -> safe crop" primitive


class TrackEmbedder:
    def __init__(self, extractor, interval=10, ttl=300, quality=None):
        """
        extractor : a ReIDExtractor (loaded once, reused).
        interval  : recompute a track's embedding every N frames (>=1).
        ttl       : evict a track from the cache after it's been unseen for this
                    many frames.
        quality   : optional crop-quality gate config. Bad crops are not
                    embedded or persisted, which keeps the gallery cleaner.
        """
        self.extractor = extractor
        self.interval = max(1, interval)
        self.ttl = ttl
        quality = quality or {}
        self.quality_enabled = quality.get("enabled", False)
        self.min_crop_width = quality.get("min_width", 24)
        self.min_crop_height = quality.get("min_height", 64)
        self.min_crop_area = quality.get("min_area", 2500)
        self.min_box_area_ratio = quality.get("min_box_area_ratio", 0.002)
        self.min_blur = quality.get("min_blur", 20.0)
        self.min_brightness = quality.get("min_brightness", 20.0)
        self.max_brightness = quality.get("max_brightness", 235.0)
        self.min_aspect = quality.get("min_aspect", 0.20)
        self.max_aspect = quality.get("max_aspect", 1.20)

        # track_id -> {"embedding": (512,) np.ndarray, "frame": int last computed}
        self._cache = {}

        # How many forward-pass embeddings we actually computed on the last
        # process() call. Exposed for monitoring/logging (forward passes per
        # frame is the cost knob in production); not part of the core contract.
        self.last_num_embedded = 0

        # The detections that got a FRESH embedding on the last process() call
        # (i.e. were recomputed, not served from cache). The live pipeline uses
        # this to know exactly which observations are new and worth persisting to
        # the vector store -- so we store at the throttle rate, not every frame.
        self.last_embedded = []

        # Quality-gate monitoring for the last process() call.
        self.last_quality_rejected = []

    def process(self, frame, detections, frame_index):
        """
        Attach det.embedding for every tracked detection in `detections`, in
        place, and return the same list.

        Untracked detections (track_id is None -- e.g. a box ByteTrack hasn't
        confirmed yet) are skipped: with no stable id we have nowhere to cache
        and nothing to attach embeddings to across frames.
        """
        due = []   # (detection, crop) pairs that need a fresh forward pass
        rejected = []

        for det in detections:
            if det.track_id is None:
                continue

            entry = self._cache.get(det.track_id)
            is_due = entry is None or (frame_index - entry["frame"]) >= self.interval

            if not is_due:
                det.embedding = entry["embedding"]        # cache hit: reuse
                continue

            crop = crop_person(frame, det)
            if crop is None:
                # Degenerate box this frame. Keep the last good vector if we have
                # one; otherwise this track stays unembedded until a valid crop.
                if entry is not None:
                    det.embedding = entry["embedding"]
                continue
            quality = self._crop_quality(crop, frame.shape)
            det.crop_quality = quality
            if not quality["accepted"]:
                rejected.append(det)
                if entry is not None:
                    det.embedding = entry["embedding"]
                continue
            due.append((det, crop))

        # One batched forward pass for every track due this frame.
        if due:
            embeddings = self.extractor.extract_batch([c for _, c in due])
            for (det, _), emb in zip(due, embeddings):
                det.embedding = emb
                self._cache[det.track_id] = {"embedding": emb, "frame": frame_index}
        self.last_num_embedded = len(due)
        self.last_embedded = [det for det, _ in due]
        self.last_quality_rejected = rejected

        self._evict_stale(frame_index)
        return detections

    def _crop_quality(self, crop, frame_shape):
        """Return quality metrics plus an accepted/reason decision."""
        h, w = crop.shape[:2]
        area = h * w
        frame_h, frame_w = frame_shape[:2]
        frame_area = max(1, frame_h * frame_w)
        aspect = w / max(1, h)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        metrics = {
            "accepted": True,
            "reason": "ok",
            "width": w,
            "height": h,
            "area": area,
            "box_area_ratio": area / frame_area,
            "aspect": aspect,
            "blur": blur,
            "brightness": brightness,
        }
        if not self.quality_enabled:
            return metrics

        checks = [
            (w >= self.min_crop_width, "too_narrow"),
            (h >= self.min_crop_height, "too_short"),
            (area >= self.min_crop_area, "too_small"),
            (metrics["box_area_ratio"] >= self.min_box_area_ratio, "too_tiny_in_frame"),
            (self.min_aspect <= aspect <= self.max_aspect, "bad_aspect"),
            (blur >= self.min_blur, "blurry"),
            (self.min_brightness <= brightness <= self.max_brightness, "bad_brightness"),
        ]
        for ok, reason in checks:
            if not ok:
                metrics["accepted"] = False
                metrics["reason"] = reason
                break
        return metrics

    def _evict_stale(self, frame_index):
        """Drop tracks not refreshed within `ttl` frames (they've left view)."""
        stale = [tid for tid, e in self._cache.items()
                 if frame_index - e["frame"] > self.ttl]
        for tid in stale:
            del self._cache[tid]
