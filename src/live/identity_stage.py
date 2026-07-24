"""
identity_stage.py  --  STAGE: assign a stable reid to every detection.

FRESH engine (per handoff §4), but with the PROVEN decision policies ported from
the existing tuned system (Execution rules pt.2) -- fresh implementation, NOT
fresh decision logic. Serial single thread so the id space stays consistent.

Stage 1 scope (the evidence-gate CORE, pt.3):
  * A new track is shown IMMEDIATELY with a PROVISIONAL (negative) id and commits
    to no identity yet.
  * It accumulates a QUALITY-WEIGHTED rolling prototype; only after
    `min_evidence_obs` good observations does it RESOLVE -- match-or-mint ON THE
    AGGREGATE (never the first frame).
  * Match uses the ported invariants: same-camera time-overlap HARD veto, a
    same-camera lane (>= same_camera_threshold) and a cross-camera lane
    (>= cross_camera_threshold), a runner-up margin, and mint-when-uncertain
    (false-merge-conservative).
  * Active set is in-memory prototypes.

Deferred to Stage 3 (the FULL §4 engine): medoids, TTL/LRU eviction, cold
reactivation via Qdrant, reciprocal-best, anti-starvation priority queue, async
persistence, and the pluggable topology veto. Provisional ids are NEGATIVE and
never leave this process (drawing.py renders them as "REID ..." pending).
"""

import threading

import numpy as np

from identity.verifier import bbox_quality_scalar   # reuse the proven quality scalar


class IdentityStage(threading.Thread):
    def __init__(self, identity_queue, render_queues, stop_event,
                 min_evidence_obs=3, same_camera_threshold=0.90,
                 cross_camera_threshold=0.63, accept_margin=0.03):
        super().__init__(name="identity", daemon=True)
        self.in_q = identity_queue
        self.render_queues = render_queues     # {cam: DropOldestQueue}
        self.stop_event = stop_event
        self.min_obs = max(1, int(min_evidence_obs))
        self.same_thr = float(same_camera_threshold)
        self.cross_thr = float(cross_camera_threshold)
        self.margin = float(accept_margin)

        self._tracks = {}       # (cam, track_id) -> track state dict
        self._active = {}       # gid -> {"sum","n","proto","cams":{cam:[[t0,t1],..]}, "last_ts"}
        self._next_gid = 1
        self._next_prov = -1
        self.minted = 0

    # ---- thread loop -------------------------------------------------------
    def run(self):
        while not self.stop_event.is_set():
            frame = self.in_q.get(timeout=0.1)
            if frame is None:
                continue
            self._resolve_frame(frame)
            rq = self.render_queues.get(frame.cam)
            if rq is not None:
                rq.put(frame)

    def _resolve_frame(self, frame):
        """Stamp reid_id on every detection in the frame."""
        fresh = frame.meta.get("fresh_track_ids", set())
        for det in (frame.detections or []):
            if det.track_id is None:
                det.reid_id = None
                continue
            has_fresh_emb = (det.track_id in fresh) and (det.embedding is not None)
            reid = self._resolve_track(frame.cam, det, frame.ts, has_fresh_emb)
            det.reid_id = reid
            det.global_id = reid if (reid is not None and reid > 0) else None

    # ---- per-track evidence gate + match-or-mint ---------------------------
    def _resolve_track(self, cam, det, ts, has_fresh_emb):
        key = (cam, det.track_id)
        st = self._tracks.get(key)
        if st is None:
            st = {"gid": self._mint_provisional(), "status": "provisional",
                  "sum": None, "w": 0.0, "obs": 0, "first_ts": ts, "last_ts": ts,
                  "span": None}
            self._tracks[key] = st

        st["last_ts"] = ts
        if not has_fresh_emb:
            # No new evidence this frame -> keep the track's current id (sticky).
            if st["status"] == "assigned" and st["span"] is not None:
                st["span"][1] = ts     # extend its active window
            return st["gid"]

        emb = np.asarray(det.embedding, dtype=np.float32).ravel()
        n = np.linalg.norm(emb)
        if n > 0:
            w = max(bbox_quality_scalar(det.crop_quality), 0.05)
            unit = emb / n
            st["sum"] = (unit * w) if st["sum"] is None else (st["sum"] + unit * w)
            st["w"] += w
        st["obs"] += 1

        if st["status"] == "provisional" and st["obs"] >= self.min_obs:
            self._resolve(cam, det.track_id, st, ts)
        elif st["status"] == "assigned":
            self._reinforce(cam, st, ts)      # keep prototype fresh; sticky id
        return st["gid"]

    def _aggregate(self, st):
        if st["sum"] is None or st["w"] <= 0:
            return None
        v = st["sum"] / st["w"]
        n = np.linalg.norm(v)
        return None if n <= 0 else (v / n)

    def _resolve(self, cam, track_id, st, ts):
        agg = self._aggregate(st)
        if agg is None:
            st["obs"] = self.min_obs - 1      # degenerate; wait for more
            return
        gid = self._match(cam, agg, ts)
        if gid is None:
            gid = self._mint()
        st["gid"] = gid
        st["status"] = "assigned"
        # Add this track's evidence to the identity's active-set prototype + span.
        a = self._active.setdefault(
            gid, {"sum": np.zeros_like(agg), "n": 0, "proto": None, "cams": {}, "last_ts": ts})
        a["sum"] = a["sum"] + agg
        a["n"] += 1
        a["proto"] = a["sum"] / max(1e-12, np.linalg.norm(a["sum"]))
        a["last_ts"] = ts
        span = [st["first_ts"], ts]
        a["cams"].setdefault(cam, []).append(span)
        st["span"] = span

    def _reinforce(self, cam, st, ts):
        """An already-assigned track keeps its id (sticky) but refreshes the
        prototype and its active-camera window with new evidence."""
        agg = self._aggregate(st)
        a = self._active.get(st["gid"])
        if agg is None or a is None:
            return
        a["sum"] = a["sum"] + agg
        a["n"] += 1
        a["proto"] = a["sum"] / max(1e-12, np.linalg.norm(a["sum"]))
        a["last_ts"] = ts
        if st["span"] is not None:
            st["span"][1] = ts

    def _match(self, cam, agg, ts):
        """Two-lane match on the aggregate against in-memory prototypes.
        Returns a gid to reuse, or None to mint. Ported invariants: same-camera
        overlap HARD veto, lane thresholds, runner-up margin, mint-when-uncertain."""
        scored = []
        for gid, a in self._active.items():
            if a["proto"] is None:
                continue
            if self._same_camera_overlap(gid, cam, ts):
                continue                        # provably a different person
            s = float(a["proto"] @ agg)
            same_cam = cam in a["cams"]
            scored.append((s, gid, same_cam))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=True)
        best_s, best_gid, best_same = scored[0]
        runner = scored[1][0] if len(scored) > 1 else -1.0
        thr = self.same_thr if best_same else self.cross_thr
        if best_s >= thr and (best_s - runner) >= self.margin:
            return best_gid
        return None                             # uncertain -> mint (conservative)

    def _same_camera_overlap(self, gid, cam, ts):
        """True if this identity already has a span in `cam` that strictly
        contains `ts` -- i.e. it is co-present here via another track, so it
        cannot also be this new track (two bodies)."""
        for (t0, t1) in self._active.get(gid, {}).get("cams", {}).get(cam, []):
            if t0 < ts < t1:
                return True
        return False

    def _mint(self):
        gid = self._next_gid
        self._next_gid += 1
        self.minted += 1
        return gid

    def _mint_provisional(self):
        pid = self._next_prov
        self._next_prov -= 1
        return pid
