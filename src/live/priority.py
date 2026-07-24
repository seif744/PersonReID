"""
priority.py  --  the anti-starvation priority queue for the identity stage
(Stage 3, §5).

The identity stage reads ONE shared input queue that carries whole frames from
every camera (topology fixed -- see queues.py; we do NOT merge or re-route the
pipeline's queues). Under a burst, one busy camera can dominate that FIFO and
delay every other camera's identity resolution -- so its boxes sit on "REID ..."
and its output_<cam>.mp4 looks bad. That is starvation.

CameraFairQueue is an INTERNAL re-ordering buffer the stage drains its input into
each cycle. It keeps a separate FIFO lane per camera and hands them out
ROUND-ROBIN, so:

  * per-camera order is preserved (each lane is FIFO -> a camera's frames render
    in capture order);
  * cross-camera service is fair -- a camera with pending work is served within
    one rotation, so no camera waits behind another's backlog. STARVATION BOUND:
    a non-empty lane is served within `n_active_cameras` pops (provable, tested);
  * it stays load-shedding: each lane is bounded and drops its OLDEST frame when
    full (counted), consistent with the pipeline's bounded-latency discipline.

This changes NOTHING about the external queue topology or drop policy; it only
decides the ORDER the identity engine processes already-delivered frames.
"""

import collections


class CameraFairQueue:
    def __init__(self, max_per_lane=64):
        self.max_per_lane = max(1, int(max_per_lane))
        self._lanes = {}          # cam -> deque
        self._order = []          # cameras in first-seen order (rotation ring)
        self._rr = 0              # index into _order of the next lane to consider
        self.dropped = 0          # frames shed by a full lane (overload protection)

    def push(self, cam, item):
        lane = self._lanes.get(cam)
        if lane is None:
            lane = collections.deque()
            self._lanes[cam] = lane
            self._order.append(cam)
        if len(lane) >= self.max_per_lane:
            lane.popleft()        # shed the OLDEST for this camera
            self.dropped += 1
        lane.append(item)

    def pop(self):
        """Return the next item round-robin across non-empty lanes, or None if
        the queue is empty. Advances the rotation so a *different* camera is
        preferred next -- bounding any camera's wait to one full rotation."""
        n = len(self._order)
        if n == 0:
            return None
        for step in range(n):
            idx = (self._rr + step) % n
            cam = self._order[idx]
            lane = self._lanes[cam]
            if lane:
                item = lane.popleft()
                self._rr = (idx + 1) % n     # next pop starts past the one just served
                return item
        return None                          # all lanes empty

    def lane_len(self, cam):
        lane = self._lanes.get(cam)
        return len(lane) if lane else 0

    def __len__(self):
        return sum(len(lane) for lane in self._lanes.values())
