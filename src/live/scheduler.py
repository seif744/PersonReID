"""
scheduler.py  --  STAGE: batch scheduler (Stage 1 = trivial; Stage 2 = fair).

Reads each active camera's NEWEST slot, keeps only frames that are still FRESH
(age from the wall-clock `ts` <= max_frame_staleness_ms -- so a frame that sat
in a slow decode is judged stale and skipped, v5 §5), and dispatches a batch to
the inference stage when it reaches `max_batch_size` OR `t_batch_ms` elapses.

Stage 1 keeps the policy trivial (grab fresh frames, dispatch; bs is usually 1
on CPU). It ALREADY emits a list ("batch"), so Stage 2 only has to add fairness
(round-robin, starvation bound) without changing the interface. It never waits
for an individual camera beyond one cycle -- a stalled camera simply contributes
nothing that cycle (no head-of-line blocking).
"""

import threading
import time


class BatchScheduler(threading.Thread):
    def __init__(self, slots, inference_queue, stop_event,
                 max_batch_size=1, t_batch_ms=15, max_frame_staleness_ms=100,
                 cycle_sleep_ms=2):
        super().__init__(name="scheduler", daemon=True)
        self.slots = slots                      # {cam: NewestSlot}
        self.inference_queue = inference_queue  # DropOldestQueue of batches (list[Frame])
        self.stop_event = stop_event
        self.max_batch_size = max(1, int(max_batch_size))
        self.t_batch = float(t_batch_ms) / 1000.0
        self.max_staleness = float(max_frame_staleness_ms) / 1000.0
        self.cycle_sleep = float(cycle_sleep_ms) / 1000.0
        self.dispatched_batches = 0
        self.dispatched_frames = 0
        self.stale_skipped = 0

    def run(self):
        while not self.stop_event.is_set():
            batch = []
            deadline = time.monotonic() + self.t_batch
            # Accumulate fresh frames until the batch is full or the timer fires.
            while (len(batch) < self.max_batch_size
                   and time.monotonic() < deadline
                   and not self.stop_event.is_set()):
                got_any = False
                for cam, slot in self.slots.items():
                    frame = slot.get()
                    if frame is None:
                        continue
                    got_any = True
                    if frame.age_ms(time.time()) > self.max_staleness * 1000.0:
                        self.stale_skipped += 1     # too old -> skip, don't process late
                        continue
                    batch.append(frame)
                    if len(batch) >= self.max_batch_size:
                        break
                if not got_any:
                    time.sleep(self.cycle_sleep)     # nothing ready; yield briefly

            if batch:
                self.inference_queue.put(batch)
                self.dispatched_batches += 1
                self.dispatched_frames += len(batch)
