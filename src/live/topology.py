"""
topology.py  --  the pluggable cross-camera transition veto (Stage 3, §5).

The cross-camera lane of the identity engine can consult a TOPOLOGY model to
reject *physically impossible* jumps: a person last seen in one camera cannot be
the person now in another if not enough time has elapsed to physically get there.
This is historically the single biggest cross-camera false-merge cut.

FAIL-OPEN is the fixed invariant (handoff Execution rule 5 + Stage-3 spec): the
veto only ever *rejects* a pair it has data for. An unknown camera, or no
topology at all -> the pair is ALLOWED (appearance-only), never blocked. So
plugging in a partial map can only ADD safe vetoes; it can never starve a camera
we lack data for.

`FailOpenTopology` is the do-nothing default. `GraphTopology` is the real veto:
an adjacency map with per-edge MINIMUM transit times, from which it derives the
shortest-path minimum time between ANY two cameras and vetoes a link only when
LESS than that has elapsed (a teleport). It is wired from the `live.topology`
config block when present (see pipeline.py).
"""

from abc import ABC, abstractmethod


class TopologyVeto(ABC):
    """Interface the identity engine calls before accepting a cross-camera link.

    `allowed(...)` answers: could a person last seen in `src_cam` at `src_ts`
    plausibly be the person now in `dst_cam` at `dst_ts`? Return True to ALLOW
    (appearance decides), False to VETO. Implementations MUST fail open: if they
    lack data for this pair, return True.
    """

    @abstractmethod
    def allowed(self, src_cam, dst_cam, src_ts, dst_ts):  # pragma: no cover - interface
        raise NotImplementedError


class FailOpenTopology(TopologyVeto):
    """The default: no topology data -> every transition is allowed.

    Used whenever the `live.topology` config block is absent or disabled. It is
    intentionally trivial and total: same camera is always fine, and any
    cross-camera pair is allowed (appearance-only).
    """

    def allowed(self, src_cam, dst_cam, src_ts, dst_ts):
        return True


class GraphTopology(TopologyVeto):
    """Physically-grounded veto from a camera adjacency map + per-edge MINIMUM
    transit times.

    Edges are `(cam_a, cam_b, min_transit_sec)`; UNDIRECTED by default because
    walking between two cameras is bidirectional. From the edges it computes the
    all-pairs shortest-path minimum transit time (Floyd-Warshall over the handful
    of cameras), so a non-adjacent pair still gets a real minimum through the
    chokepoints (e.g. cam_206 <-> cam_224 sums cam_206<->cam_213<->cam_224).

    A cross-camera link is VETOED only when strictly LESS time than that minimum
    has elapsed -- i.e. the person could not physically have covered the distance
    yet. A LONGER gap is always allowed (loitering / an indirect route / a
    missed intermediate camera), so this NEVER causes a false SPLIT of a genuine
    slow re-appearance -- it only removes impossible-fast merges. This is the
    conservative direction (handoff §1: false-merge-conservative, and never trade
    correctness for a veto).

    Fail-open on any camera the map doesn't know (invariant §2.5): an unknown
    camera -> allowed. Same camera -> always allowed (same-camera logic is the
    engine's overlap veto, not topology).
    """

    _INF = float("inf")

    def __init__(self, edges=None, directed=False):
        self._adj = {}                 # node -> {neighbour: min_transit_sec}
        self._known = set()
        self._dist = None              # cached all-pairs min transit
        for e in (edges or []):
            self.add_edge(*e, directed=directed)

    def add_edge(self, cam_a, cam_b, min_transit_sec, directed=False):
        w = max(0.0, float(min_transit_sec))
        self._adj.setdefault(cam_a, {})
        self._adj.setdefault(cam_b, {})
        self._adj[cam_a][cam_b] = min(w, self._adj[cam_a].get(cam_b, self._INF))
        if not directed:
            self._adj[cam_b][cam_a] = min(w, self._adj[cam_b].get(cam_a, self._INF))
        self._known.add(cam_a)
        self._known.add(cam_b)
        self._dist = None              # invalidate the cache

    def known(self):
        return set(self._known)

    def _compute(self):
        nodes = list(self._known)
        INF = self._INF
        dist = {u: {v: (0.0 if u == v else INF) for v in nodes} for u in nodes}
        for u in nodes:
            for v, w in self._adj.get(u, {}).items():
                if w < dist[u][v]:
                    dist[u][v] = w
        for k in nodes:                # Floyd-Warshall (cameras are few)
            dk = dist[k]
            for i in nodes:
                dik = dist[i][k]
                if dik == INF:
                    continue
                di = dist[i]
                for j in nodes:
                    nd = dik + dk[j]
                    if nd < di[j]:
                        di[j] = nd
        self._dist = dist

    def min_transit(self, src_cam, dst_cam):
        """Shortest-path minimum transit seconds between two KNOWN cameras
        (0 for the same camera, inf if they are disconnected)."""
        if self._dist is None:
            self._compute()
        return self._dist.get(src_cam, {}).get(dst_cam, self._INF)

    def allowed(self, src_cam, dst_cam, src_ts, dst_ts):
        if src_cam == dst_cam:
            return True
        if src_cam not in self._known or dst_cam not in self._known:
            return True                            # fail-open on unknown cameras
        dt = dst_ts - src_ts
        if dt < 0:
            dt = 0.0                               # clock-skew guard
        return dt >= self.min_transit(src_cam, dst_cam)
