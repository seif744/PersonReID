"""
Stage 2 validation: parallel per-camera inference.

Verifies the concurrency contract WITHOUT real models (deterministic, no GPU):
  * Every frame in a batch is processed and forwarded (none lost).
  * A single camera's frames are processed SEQUENTIALLY on one worker -- its
    detector/ByteTrack is never touched by two threads at once (stop-rule).
  * The SHARED ReID extractor is never used concurrently (the _embed_lock holds).
  * Different cameras DO overlap (real parallelism, not accidental serialisation).
  * Per-camera frame order is preserved downstream (batch order out).
"""

import threading
import time
from types import SimpleNamespace

from _synth import Check
from live.frame import Frame
from live.inference import InferenceStage
from live.queues import DropOldestQueue


class FakeDetector:
    """Records call order; trips if the SAME detector is entered concurrently."""
    def __init__(self, cam):
        self.cam = cam
        self.calls = []
        self._active = 0
        self.overlap_violation = False

    def track(self, image):
        self._active += 1
        if self._active != 1:
            self.overlap_violation = True
        time.sleep(0.01)                       # widen any race window
        self.calls.append(image)               # image carries the frame_index
        self._active -= 1
        return [SimpleNamespace(track_id=int(image))]


class FakeEmbedder:
    """Shares a counter across cameras; trips if the shared extractor is entered
    concurrently (i.e. the _embed_lock failed)."""
    def __init__(self, shared):
        self.shared = shared
        self.last_embedded = []

    def process(self, image, detections, frame_index):
        self.shared["active"] += 1
        if self.shared["active"] != 1:
            self.shared["violation"] = True
        self.shared["max_parallel_cams"] = max(
            self.shared["max_parallel_cams"], self.shared["detect_active"])
        time.sleep(0.01)
        self.last_embedded = list(detections)
        self.shared["active"] -= 1


def main():
    c = Check("Stage2 parallel inference (per-camera concurrency + shared-extractor lock)")

    shared = {"active": 0, "violation": False, "detect_active": 0, "max_parallel_cams": 0}
    cams = ["A", "B", "C"]
    detectors = {cam: FakeDetector(cam) for cam in cams}

    # Wrap track() to count how many cameras detect at the same time (proves overlap).
    for cam, det in detectors.items():
        orig = det.track
        def tracked(image, _orig=orig):
            shared["detect_active"] += 1
            shared["max_parallel_cams"] = max(shared["max_parallel_cams"], shared["detect_active"])
            try:
                return _orig(image)
            finally:
                shared["detect_active"] -= 1
        det.track = tracked

    embedders = {cam: FakeEmbedder(shared) for cam in cams}

    inq = DropOldestQueue(4)
    outq = DropOldestQueue(64)
    stop = threading.Event()
    stage = InferenceStage(detectors, embedders, inq, outq, stop, max_workers=3)

    # Batch: A twice (out of order among other cams), B once, C twice.
    ts = time.time()
    batch = [Frame("A", ts, 1, image=1), Frame("B", ts, 5, image=5),
             Frame("A", ts, 2, image=2), Frame("C", ts, 7, image=7),
             Frame("C", ts, 8, image=8)]
    inq.put(batch)
    stage.start()

    deadline = time.time() + 5
    while stage.frames_done < len(batch) and time.time() < deadline:
        time.sleep(0.01)
    stop.set()
    stage.join(timeout=2)

    c.eq(stage.frames_done, len(batch), "every frame processed and forwarded")
    out = []
    while True:
        f = outq.get(timeout=0)
        if f is None:
            break
        out.append(f)
    c.eq([f.frame_index for f in out], [1, 5, 2, 7, 8], "forwarded in batch order (per-camera order preserved)")
    c.eq(detectors["A"].calls, [1, 2], "camera A frames processed sequentially, in order")
    c.eq(detectors["C"].calls, [7, 8], "camera C frames processed sequentially, in order")
    c.ok(not any(d.overlap_violation for d in detectors.values()),
         "no camera's detector was entered concurrently (ByteTrack race-free)")
    c.ok(not shared["violation"], "shared ReID extractor never used concurrently (_embed_lock holds)")
    c.ok(shared["max_parallel_cams"] >= 2,
         f"cameras actually ran in parallel (max concurrent detect = {shared['max_parallel_cams']})")

    c.done()


if __name__ == "__main__":
    main()
