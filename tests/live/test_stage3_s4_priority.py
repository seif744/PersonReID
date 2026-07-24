"""
Stage 3 -- Substage 4 validation: the anti-starvation priority (fair) queue.

Acceptance (this substage):
  * Per-camera FIFO is preserved (a camera's frames come out in push order).
  * Round-robin fairness: a camera with a single frame is served promptly, not
    stuck behind another camera's large backlog.
  * Starvation bound: any non-empty lane is served within n_active_cameras pops.
  * Load-shedding: a lane over its bound drops its OLDEST item (counted), keeping
    the newest -- consistent with the pipeline's bounded-latency discipline.
  * Empty queue pops None; __len__ is accurate.
"""

from _synth import Check
from live.priority import CameraFairQueue


def _drain(q):
    out = []
    while True:
        item = q.pop()
        if item is None:
            break
        out.append(item)
    return out


def main():
    c = Check("Stage3-S4 anti-starvation fair queue")

    # empty
    q = CameraFairQueue()
    c.ok(q.pop() is None, "empty queue pops None")
    c.eq(len(q), 0, "empty queue has length 0")

    # per-camera FIFO + round-robin fairness
    q = CameraFairQueue()
    for x in ["a1", "a2", "a3"]:
        q.push("A", x)
    q.push("B", "b1")               # B arrives with a single frame behind A's backlog
    c.eq(len(q), 4, "length counts all lanes")
    order = _drain(q)
    c.eq(order, ["a1", "b1", "a2", "a3"], "round-robin interleaves; B not stuck behind A")
    # per-camera order within the interleaving is preserved
    c.eq([o for o in order if o.startswith("a")], ["a1", "a2", "a3"], "camera A stays FIFO")

    # starvation bound: with 3 cameras each holding work, each is served within 3 pops
    q = CameraFairQueue()
    for k in range(10):
        q.push("A", f"a{k}")
    q.push("B", "b0")
    q.push("C", "c0")
    first3 = [q.pop() for _ in range(3)]
    c.ok("b0" in first3 and "c0" in first3,
         f"B and C served within n_cameras(=3) pops (first3={first3})")

    # load-shedding: lane bound drops OLDEST, keeps NEWEST
    q = CameraFairQueue(max_per_lane=3)
    for k in range(5):              # push 5 into a 3-slot lane
        q.push("A", f"a{k}")
    c.eq(q.lane_len("A"), 3, "lane capped at max_per_lane")
    c.eq(q.dropped, 2, "two oldest frames dropped (counted)")
    kept = _drain(q)
    c.eq(kept, ["a2", "a3", "a4"], "oldest dropped, newest retained (drop-oldest)")

    c.done()


if __name__ == "__main__":
    main()
