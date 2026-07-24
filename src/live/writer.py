"""
writer.py  --  STAGE: encode annotated frames -> output_<cam>.mp4 (per camera).

Software H.264 (x264) via OpenCV/FFmpeg, `mp4v` fallback. Runs on its own thread
per camera so encoding (CPU-bound; no NVENC on A100) never stalls the GPU stage.

WALL-CLOCK PACING (v5 §7): the output must play back in real time regardless of
how fast frames arrive. The writer ticks at `fps`; each tick it writes the NEWEST
annotated frame available (draining/dropping older queued frames -> real-time
length), or DUPLICATES the last frame when none arrived (fills processing gaps).

OFFLINE OVERLAY: after `offline_overlay_after_sec` with no new frame (camera
down / reconnecting), it writes a "CAMERA OFFLINE" frame -- with camera id,
wall-clock timestamp, and reconnect count -- once per fps tick so the timeline
stays wall-clock-continuous (never an infinite freeze, never sporadic frames).

DEGRADE, NEVER CRASH (v5 §14): if the writer can't open (codec/disk), it logs
and drops recording for that camera; the pipeline keeps running.
"""

import threading
import time
from datetime import datetime

import cv2


class WriterStage(threading.Thread):
    def __init__(self, cam, writer_queue, stop_event, out_path, fps=20.0,
                 codec="h264", offline_after_sec=3.0, reconnect_ref=None):
        super().__init__(name=f"writer-{cam}", daemon=True)
        self.cam = cam
        self.in_q = writer_queue
        self.stop_event = stop_event
        self.out_path = out_path
        self.fps = float(fps) if fps and fps > 0 else 20.0
        self.codec = codec
        self.offline_after = float(offline_after_sec)
        self.reconnect_ref = reconnect_ref     # optional callable -> reconnect count
        self.writer = None
        self.disabled = False                  # True if we gave up opening a writer
        self.frames_written = 0

    def _open(self, w, h):
        # Prefer software H.264; fall back to mp4v (matches the file path).
        candidates = (["avc1", "h264", "H264", "mp4v"] if self.codec == "h264"
                      else ["mp4v"])
        for cc in candidates:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))
            if writer.isOpened():
                print(f"[writer:{self.cam}] {self.out_path} ({cc}, {self.fps:g} fps, {w}x{h})")
                return writer
            writer.release()
        print(f"[writer:{self.cam}] ERROR: could not open a VideoWriter "
              f"(codecs tried: {candidates}); recording DISABLED for this camera.")
        self.disabled = True
        return None

    def _offline_frame(self, template):
        img = template.copy()
        img[:] = (20, 20, 20)
        rc = ""
        if self.reconnect_ref is not None:
            try:
                rc = f"  reconnect #{self.reconnect_ref()}"
            except Exception:
                rc = ""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"CAMERA OFFLINE: {self.cam}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(img, f"{now}{rc}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img

    def run(self):
        interval = 1.0 / self.fps
        next_tick = time.monotonic()
        last_img = None
        last_real = None        # monotonic time we last received a real frame

        def drain_newest(block_timeout):
            """Return the newest (img, ts) available, blocking up to
            block_timeout for the first, then draining the rest non-blocking."""
            item = self.in_q.get(timeout=max(0.0, block_timeout))
            if item is None:
                return None
            nxt = self.in_q.get(timeout=0.0)
            while nxt is not None:
                item = nxt
                nxt = self.in_q.get(timeout=0.0)
            return item

        while not self.stop_event.is_set():
            now = time.monotonic()
            item = drain_newest(next_tick - now)
            if item is not None:
                img, _cap_ts = item
                last_img = img
                last_real = time.monotonic()
                if self.writer is None and not self.disabled:
                    h, w = img.shape[:2]
                    self.writer = self._open(w, h)

            now = time.monotonic()
            if now < next_tick:
                continue
            # A tick is due. Write newest, or duplicate, or offline overlay.
            if last_img is not None and self.writer is not None:
                gap = now - (last_real or now)
                frame = (self._offline_frame(last_img)
                         if gap > self.offline_after else last_img)
                self.writer.write(frame)
                self.frames_written += 1
            # Advance the tick; if we fell far behind, resync (no spiral).
            next_tick += interval
            if next_tick < now - interval:
                next_tick = now

        # ---- shutdown: flush a few remaining frames, then finalize the MP4 ----
        try:
            item = self.in_q.get(timeout=0.0)
            while item is not None and self.writer is not None:
                self.writer.write(item[0])
                self.frames_written += 1
                item = self.in_q.get(timeout=0.0)
        finally:
            if self.writer is not None:
                self.writer.release()
                print(f"[writer:{self.cam}] finalized {self.out_path} "
                      f"({self.frames_written} frames).")
