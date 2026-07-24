"""
queues.py  --  the bounded, load-shedding primitives every live stage uses.

The v5 architecture's core discipline (Execution rules): **bound latency, shed
load** rather than let work pile up. Queues are deliberately small; when full
they DROP (and count the drop) instead of blocking, so overload surfaces as a
bounded, logged loss instead of ever-growing lag. Two primitives:

  * NewestSlot     -- size 1, keep only the freshest item (capture -> scheduler).
                      A slow consumer always sees the latest frame, never a backlog.
  * DropOldestQueue-- bounded FIFO; when full, drop the OLDEST to make room
                      (identity / render / writer queues).

Neither ever blocks a producer. `get()` blocks the CONSUMER briefly (with a
timeout) so threads can poll a stop flag between items.
"""

import collections
import threading


class NewestSlot:
    """A one-item mailbox that keeps only the newest value (drop-oldest, size 1)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self.dropped = 0

    def put(self, item):
        with self._lock:
            if self._item is not None:
                self.dropped += 1     # overwrote an unconsumed frame
            self._item = item

    def get(self):
        """Return the current item and clear the slot (None if empty)."""
        with self._lock:
            item, self._item = self._item, None
            return item

    def peek(self):
        with self._lock:
            return self._item


class DropOldestQueue:
    """
    Bounded FIFO. `put` never blocks: at capacity it drops the OLDEST item to
    make room for the new one (and increments `dropped`). `get` blocks the
    consumer up to `timeout` seconds so it can re-check a stop flag.
    """

    def __init__(self, maxsize):
        self.maxsize = max(1, int(maxsize))
        self._dq = collections.deque()
        self._cv = threading.Condition()
        self.dropped = 0

    def put(self, item):
        with self._cv:
            if len(self._dq) >= self.maxsize:
                self._dq.popleft()
                self.dropped += 1
            self._dq.append(item)
            self._cv.notify()

    def get(self, timeout=0.1):
        with self._cv:
            if not self._dq:
                self._cv.wait(timeout)
            if not self._dq:
                return None
            return self._dq.popleft()

    def __len__(self):
        with self._cv:
            return len(self._dq)
