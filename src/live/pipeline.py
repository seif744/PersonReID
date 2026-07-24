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
from live.topology import FailOpenTopology, GraphTopology
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
        self.renderers = []               # RenderStage refs (metrics)
        # metrics wiring (populated in run(); read by the reporter)
        self._m = {}                      # {slots, inference_queue, identity_queue,
                                          #  render_queues, writer_queues, scheduler,
                                          #  inference, identity, max_batch}
        self._peaks = {"inference_q": 0, "identity_q": 0, "writer_q": 0}
        self._t_start = None

    # ---- config helpers ----------------------------------------------------
    def _g(self, group, key, default):
        return (self.live_cfg.get(group, {}) or {}).get(key, default)

    def _build_topology(self):
        """Build the cross-camera veto from the live.topology config block.
        Returns a GraphTopology when enabled with edges, else FailOpenTopology
        (the fail-open invariant: no data -> nothing is ever blocked)."""
        tcfg = self.live_cfg.get("topology", {}) or {}
        edges = tcfg.get("edges") or []
        if not tcfg.get("enabled", False) or not edges:
            print("[live] cross-camera topology veto: OFF (appearance-only; intended, "
                  "not an error -- enable live.topology with measured transit times to use it).")
            return FailOpenTopology()
        parsed = []
        for e in edges:
            try:
                a, b, sec = e[0], e[1], float(e[2])
                parsed.append((str(a), str(b), sec))
            except (TypeError, IndexError, ValueError):
                print(f"[live] topology: skipping malformed edge {e!r} "
                      f"(expected [cam_a, cam_b, min_transit_sec]).")
        if not parsed:
            return FailOpenTopology()
        topo = GraphTopology(edges=parsed)
        cams = ", ".join(sorted(topo.known()))
        print(f"[live] topology veto ACTIVE: {len(parsed)} edge(s) over "
              f"{len(topo.known())} camera(s) [{cams}]; any other camera is fail-open.")
        return topo

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

        # LIVE-only pose toggle: the pose ensemble is a SECOND model per frame.
        # Skipping it ~halves detection cost -> far fewer dropped frames -> stabler
        # ByteTrack -> less identity fragmentation. Scoped to the live path only via
        # live.inference.pose_ensemble; the file-batch path (main.py) still reads
        # detector.pose_ensemble unchanged (regression gate intact).
        live_pose = self._g("inference", "pose_ensemble", True)
        pose_cfg = det_cfg.get("pose_ensemble") if live_pose else None
        if not live_pose:
            print("[live] pose ensemble DISABLED for the live path "
                  "(throughput: 1 detection model/frame instead of 2).")

        detectors, embedders = {}, {}
        for name, _ in self.sources:
            detectors[name] = PersonDetector(
                model_path=det_cfg["model"],
                confidence_threshold=det_cfg["confidence_threshold"],
                person_class_id=det_cfg["person_class_id"],
                tracker_config=trk_cfg.get("config", "bytetrack.yaml"),
                pose_ensemble=pose_cfg,
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

        slots, render_queues, writer_queues = {}, {}, {}
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
            writer_queues[name] = writer_q
            renderer = RenderStage(name, render_q, writer_q, self.stop_event)
            self.renderers.append(renderer)
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
                                   identity_queue, self.stop_event,
                                   max_workers=int(self._g("inference", "max_workers", 0)))
        limits_cfg = self.live_cfg.get("limits", {}) or {}
        identity = IdentityStage(
            identity_queue, render_queues, self.stop_event,
            min_evidence_obs=int(self._g("identity", "min_evidence_obs", 3)),
            same_camera_threshold=float(self._g("identity", "same_camera_threshold", 0.90)),
            cross_camera_threshold=float(self._g("identity", "cross_camera_threshold", 0.63)),
            accept_margin=float(self._g("identity", "accept_margin", 0.03)),
            bank_size=int(self._g("identity", "bank_size", 20)),
            active_ttl_sec=float(limits_cfg.get("active_ttl", 300)),
            max_active_identities=int(limits_cfg.get("max_active_identities", 200)),
            max_per_lane=int(self._g("identity", "max_queue", 64)),
            # Cross-camera physical-impossibility veto, built from the
            # live.topology config block (fail-open if absent/disabled/empty).
            topology=self._build_topology(),
        )
        # shared consumers start before captures
        self.threads.append(("identity", identity))
        self.threads.append(("inference", inference))
        self.threads.append(("scheduler", scheduler))
        for capt in self.captures:
            self.threads.append((capt.name, capt))

        # ---- stash refs for the metrics reporter ----------------------------
        self._m = {
            "slots": slots, "inference_queue": inference_queue,
            "identity_queue": identity_queue, "render_queues": render_queues,
            "writer_queues": writer_queues, "scheduler": scheduler,
            "inference": inference, "identity": identity, "max_batch": bs,
        }

        # ---- start (consumers already ordered before captures) --------------
        for _, t in self.threads:
            t.start()

        # ---- supervise until stop -------------------------------------------
        max_dur = float(self._g("run", "max_duration_sec", 0) or 0)
        log_interval = float((self.live_cfg.get("metrics", {}) or {}).get("log_interval_sec", 10) or 0)
        self._t_start = time.monotonic()
        last_log = self._t_start
        try:
            while not self.stop_event.is_set():
                self._track_peaks()
                if all(c.finished for c in self.captures):
                    print("[live] all sources ended -> draining and finalizing...")
                    time.sleep(1.0)     # let in-flight frames flush through
                    break
                if max_dur > 0 and (time.monotonic() - self._t_start) >= max_dur:
                    print(f"[live] max_duration_sec={max_dur:g} reached -> stopping.")
                    break
                now = time.monotonic()
                if log_interval > 0 and (now - last_log) >= log_interval:
                    self._report(now - self._t_start, final=False)
                    last_log = now
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
        if self._t_start is not None:
            self._report(time.monotonic() - self._t_start, final=True)
        print("[live] shutdown complete.")

    # ---- metrics -----------------------------------------------------------
    def _track_peaks(self):
        m = self._m
        if not m:
            return
        self._peaks["inference_q"] = max(self._peaks["inference_q"], len(m["inference_queue"]))
        self._peaks["identity_q"] = max(self._peaks["identity_q"], len(m["identity_queue"]))
        wq = max((len(q) for q in m["writer_queues"].values()), default=0)
        self._peaks["writer_q"] = max(self._peaks["writer_q"], wq)

    def _report(self, elapsed, final=False):
        """Print a metrics snapshot. The numbers that answer 'is it keeping up?':
        read-fps vs written-fps (the gap is dropped frames), the drop counters at
        each shedding point, and the scheduler's batch UTILISATION (how full each
        inference batch is -- low util on GPU = headroom Stage 2 batching would
        use). Identity counters show whether the engine is minting/reacquiring/
        linking sensibly."""
        m = self._m
        if not m or elapsed <= 0:
            return
        self._track_peaks()
        sched = m["scheduler"]
        avg_batch = sched.dispatched_frames / max(1, sched.dispatched_batches)
        util = 100.0 * avg_batch / max(1, m["max_batch"])
        st = m["identity"].stats()
        head = "final SUMMARY" if final else f"t=+{elapsed:.0f}s"
        print(f"[live:metrics] {head}  (elapsed={elapsed:.1f}s, max_batch={m['max_batch']})")
        for capt in self.captures:
            cam = capt.cam
            rd = capt.frame_index
            rendered = next((r.rendered for r in self.renderers if r.cam == cam), 0)
            written = next((w.frames_written for w in self.writers if w.cam == cam), 0)
            slot = m["slots"].get(cam)
            rq = m["render_queues"].get(cam)
            wq = m["writer_queues"].get(cam)
            print(f"    {cam}: read={rd} ({rd / elapsed:.1f}fps)  rendered={rendered}  "
                  f"written={written} ({written / elapsed:.1f}fps)  "
                  f"slot_drop={slot.dropped if slot else 0}  "
                  f"rq_drop={rq.dropped if rq else 0}  wq_drop={wq.dropped if wq else 0}  "
                  f"reconnects={capt.reconnects}")
        print(f"    scheduler: batches={sched.dispatched_batches} frames={sched.dispatched_frames} "
              f"avg_batch={avg_batch:.2f} util={util:.0f}% stale_skipped={sched.stale_skipped}")
        print(f"    drops/peaks: infer_q={m['inference_queue'].dropped}/{self._peaks['inference_q']} "
              f"identity_in={m['identity_queue'].dropped}/{self._peaks['identity_q']} "
              f"identity_fair_drop={st['fair_dropped']} writer_q_peak={self._peaks['writer_q']} "
              f"| inference_done={m['inference'].frames_done}")
        print(f"    identity: minted={st['minted']} reacquired={st['reacquired']} "
              f"linked={st['linked']} active={st['active_identities']} "
              f"resolved_frames={st['frames_done']}")
        print(f"    x-camera: attempts={st['xcam_attempts']} linked={st['linked']} "
              f"rejected[thresh={st['xcam_rej_threshold']} margin={st['xcam_rej_margin']} "
              f"recip={st['xcam_rej_reciprocal']} topology={st['xcam_rej_topology']}] "
              f"max_subthreshold_score={st['xcam_max_subthreshold']:.3f}")
        print(f"    same-cam reacquire: attempts={st['recam_attempts']} "
              f"ok={st['reacquired']} rejected_below_thr={st['recam_rej_below']} "
              f"max_rejected_score={st['recam_max_rej']:.3f}  "
              f"(high rejected_below_thr => fragmentation; same_camera_threshold too strict)")
        print(f"    coactive_vetoes={st['coactive_vetoes']}  "
              f"(co-present same-camera false-merges prevented)")
        from live.identity_engine import HIST_LABELS
        lab = " ".join(HIST_LABELS)
        rh = " ".join(f"{n:>4}" for n in st['recam_hist'])
        xh = " ".join(f"{n:>4}" for n in st['xcam_hist'])
        print(f"    score histograms (best candidate per attempt)   bins: {lab}")
        print(f"      same-cam reacquire: {rh}   (pick same_camera_threshold below the same-person cluster)")
        print(f"      cross-camera:       {xh}   (pick cross_camera_threshold below the true-match cluster)")
