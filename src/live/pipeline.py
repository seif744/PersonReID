"""
pipeline.py  --  LivePipeline: wire the stages, run, shut down cleanly.

Stage 1 backbone (headless; the deliverable is output_<cam>.mp4):

  per camera:  DecodeBackend -> CaptureThread -> NewestSlot            \
  shared:      BatchScheduler -> inference_queue -> InferenceStage ->   > identity_queue
  shared:      IdentityStage -> per-cam render_queue                   /
  per camera:  RenderStage -> writer_queue -> WriterStage -> output_<cam>.mp4

Startup order (v5 §12): capability report -> WARM-UP (dummy batch through
detector + ReID, BEFORE opening any source, so first real frames don't pay init
latency) -> start consumer stages -> start captures last.

Shutdown (v5 §10, race-free, in try/finally): stop -> join capture -> drain/join
scheduler+inference+identity -> join render -> flush+release writers (finalize
every MP4) -> report. A stop (Ctrl-C, all files ended, or max_duration) never
corrupts an output. N-camera generic: everything is per-camera + over the active
set; adding a camera is just another source entry.
"""

import os
import sys
import threading
import time

import numpy as np

# src/ is already on sys.path (main.py inserts it); import the reused components.
from detector import PersonDetector
from reid.extractor import ReIDExtractor
from reid.service import TrackEmbedder

from live.capabilities import capability_report
from live.decode_backend import make_decode_backend
from live.queues import NewestSlot, DropOldestQueue
from live.capture import CaptureThread
from live.scheduler import BatchScheduler
from live.inference import InferenceStage
from live.identity_stage import IdentityStage
from live.render import RenderStage
from live.writer import WriterStage


class LivePipeline:
    def __init__(self, sources, cfg):
        self.sources = sources            # [(cam_name, url_or_path), ...]  any N
        self.cfg = cfg
        self.live_cfg = cfg.get("live", {}) or {}
        self.stop_event = threading.Event()
        self.threads = []                 # (name, thread) in start order
        self.captures = []                # CaptureThread refs (for finished/reconnect)
        self.writers = []                 # WriterStage refs (finalize on shutdown)

    # ---- config helpers ----------------------------------------------------
    def _g(self, group, key, default):
        return (self.live_cfg.get(group, {}) or {}).get(key, default)

    def run(self):
        cap = capability_report(self._g("run", "device", "auto"),
                                self._g("capture", "nvdec", "auto"))
        device = cap["device"]
        print(f"[live] Starting LivePipeline on {len(self.sources)} source(s): "
              f"{', '.join(n for n, _ in self.sources)} | device={device}")

        # ---- shared model (one extractor; per-camera detectors/embedders) ----
        reid_cfg = self.cfg.get("reid", {}) or {}
        det_cfg = self.cfg.get("detector", {}) or {}
        trk_cfg = self.cfg.get("tracker", {}) or {}
        extractor = ReIDExtractor(weights=reid_cfg["weights"], device=device)

        detectors, embedders = {}, {}
        for name, _ in self.sources:
            detectors[name] = PersonDetector(
                model_path=det_cfg["model"],
                confidence_threshold=det_cfg["confidence_threshold"],
                person_class_id=det_cfg["person_class_id"],
                tracker_config=trk_cfg.get("config", "bytetrack.yaml"),
                pose_ensemble=det_cfg.get("pose_ensemble"),
            )
            embedders[name] = TrackEmbedder(
                extractor,
                interval=reid_cfg.get("interval", 10),
                ttl=reid_cfg.get("ttl", 300),
                quality=reid_cfg.get("quality"),
                max_embeddings_per_track=reid_cfg.get("max_embeddings_per_track", 0),
                warmup_embeddings=reid_cfg.get("warmup_embeddings", 3),
            )

        # ---- WARM-UP before opening any source (v5 §12) ----------------------
        self._warmup(detectors, extractor)

        # ---- build queues + per-camera stages --------------------------------
        bs = int(self._g("inference", "max_batch_size", 8)) if device.startswith("cuda") else 1
        out_cfg = self.live_cfg.get("output", {}) or {}
        max_writer_q = int(out_cfg.get("max_writer_queue", 16))
        fps = float(out_cfg.get("fps_default", 20))

        slots, render_queues = {}, {}
        inference_queue = DropOldestQueue(int(self._g("inference", "max_inference_queue", 2)))
        identity_queue = DropOldestQueue(int(self._g("identity", "max_queue", 64)))

        for name, url in self.sources:
            slot = NewestSlot()
            slots[name] = slot
            backend = make_decode_backend(
                url, capabilities=cap, stop_event=self.stop_event,
                reconnect_attempts=int(self._g("capture", "reconnect_attempts", 5)),
                reconnect_backoff=float(self._g("capture", "reconnect_base_delay", 1)),
            )
            capt = CaptureThread(
                name, backend, slot, self.stop_event,
                reconnect_base_delay=float(self._g("capture", "reconnect_base_delay", 1)),
                reconnect_max_delay=float(self._g("capture", "reconnect_max_delay", 30)),
                reconnect_attempts=int(self._g("capture", "reconnect_attempts", 5)),
                device=device,
            )
            self.captures.append(capt)

            render_q = DropOldestQueue(max_writer_q)
            writer_q = DropOldestQueue(max_writer_q)
            render_queues[name] = render_q
            renderer = RenderStage(name, render_q, writer_q, self.stop_event)
            writer = WriterStage(
                name, writer_q, self.stop_event,
                out_path=f"output_{name}.mp4", fps=fps,
                codec=out_cfg.get("codec", "h264"),
                offline_after_sec=float(out_cfg.get("offline_overlay_after_sec", 3)),
                reconnect_ref=(lambda c=capt: c.reconnects),
            )
            self.writers.append(writer)
            # register in start order: consumers first (writer, render), captures last
            self.threads.append((f"writer-{name}", writer))
            self.threads.append((f"render-{name}", renderer))

        scheduler = BatchScheduler(
            slots, inference_queue, self.stop_event,
            max_batch_size=bs,
            t_batch_ms=float(self._g("inference", "t_batch_ms", 15)),
            max_frame_staleness_ms=float(self._g("capture", "max_frame_staleness_ms", 100)),
        )
        inference = InferenceStage(detectors, embedders, inference_queue,
                                   identity_queue, self.stop_event)
        identity = IdentityStage(
            identity_queue, render_queues, self.stop_event,
            min_evidence_obs=int(self._g("identity", "min_evidence_obs", 3)),
            same_camera_threshold=float(self._g("identity", "same_camera_threshold", 0.90)),
            cross_camera_threshold=float(self._g("identity", "cross_camera_threshold", 0.63)),
            accept_margin=float(self._g("identity", "accept_margin", 0.03)),
        )
        # shared consumers start before captures
        self.threads.append(("identity", identity))
        self.threads.append(("inference", inference))
        self.threads.append(("scheduler", scheduler))
        for capt in self.captures:
            self.threads.append((capt.name, capt))

        # ---- start (consumers already ordered before captures) --------------
        for _, t in self.threads:
            t.start()

        # ---- supervise until stop -------------------------------------------
        max_dur = float(self._g("run", "max_duration_sec", 0) or 0)
        started = time.monotonic()
        try:
            while not self.stop_event.is_set():
                if all(c.finished for c in self.captures):
                    print("[live] all sources ended -> draining and finalizing...")
                    time.sleep(1.0)     # let in-flight frames flush through
                    break
                if max_dur > 0 and (time.monotonic() - started) >= max_dur:
                    print(f"[live] max_duration_sec={max_dur:g} reached -> stopping.")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[live] Ctrl-C -> stopping and finalizing outputs...")
        finally:
            self._shutdown()

    def _warmup(self, detectors, extractor):
        """Run a dummy frame through detection + ReID so the first real frames
        don't pay model-init latency. Guarded; failure is non-fatal."""
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            for det in detectors.values():
                det.track(dummy)
            extractor.extract_batch([np.zeros((256, 128, 3), dtype=np.uint8)])
            print("[live] warm-up complete (detector + ReID initialised).")
        except Exception as e:
            print(f"[live] warm-up skipped ({e}).")

    def _shutdown(self):
        """Race-free shutdown (v5 §10): stop, join in pipeline order, writers
        finalize their MP4s in their own finally-blocks."""
        self.stop_event.set()
        # Join in the order data flows so downstream flushes what upstream sent.
        order = ["scheduler", "inference", "identity"]
        by_name = {}
        for nm, t in self.threads:
            by_name.setdefault(nm, t)
        # captures first (stop producing), then core, then render, then writers
        for capt in self.captures:
            capt.join(timeout=5)
        for nm in order:
            t = by_name.get(nm)
            if t:
                t.join(timeout=5)
        for nm, t in self.threads:
            if nm.startswith("render-"):
                t.join(timeout=5)
        for w in self.writers:
            w.join(timeout=10)     # writer flushes queue + releases the MP4
        print("[live] shutdown complete.")
