"""
identity_engine.py  --  Stage 3: the full in-memory identity engine.

This is the FRESH live active-set engine (handoff §5) that turns per-frame
appearance embeddings into stable reids WITHOUT Qdrant and WITHOUT the offline
reconcile the file path uses. It ports the PROVEN decision policies from the
tuned file system (`identity/service.py::_assign_online` / `_two_lane_match` /
`_reciprocal_best_ok` and `reconcile.py`) -- fresh implementation, same logic --
so the online path *approximates* (never equals) offline quality.

Two objects, deliberately thread-free and deterministic so they can be
logic-tested on synthetic embeddings with no GPU/video/threads:

  * ActiveIdentitySet  -- the in-memory gallery: per-identity exemplar bank
    (prototype mean + medoid-style best exemplar), per-camera time spans, and
    last-seen bookkeeping, with TTL + LRU eviction to bound memory.
  * IdentityEngine     -- the per-observation decision core: evidence gate ->
    two-lane match (same-camera cold-reactivation lane, then cross-camera
    reciprocal-best lane with a pluggable topology veto) -> mint-when-uncertain.

Ported invariants (false-merge-conservative, handoff §1):
  * A new track shows a PROVISIONAL (negative) id and commits to nothing until it
    has `min_evidence_obs` good views -- match-or-mint runs on the AGGREGATE.
  * Same-camera time-overlap is a HARD veto (one body can't be two tracks at once).
  * Same-camera lane re-acquires a dropped track at a strict threshold (cold
    reactivation). Cross-camera lane links a person across cams only past a
    lower threshold AND a runner-up margin AND reciprocal-best AND topology.
  * When uncertain -> MINT (never guess a merge that no reconcile can undo).

Timestamps are the frame's wall-clock `ts` (frame.py's single clock); spans,
TTLs and topology all reason in seconds, which is what makes cross-camera
temporal logic comparable across cameras (frame indices are per-camera and
frames get dropped, so they are NOT comparable in the live path).
"""

import collections

import numpy as np

from identity.verifier import bbox_quality_scalar     # proven crop-quality scalar
from live.topology import FailOpenTopology


def _unit(vec):
    """L2-normalize a vector; return None if degenerate (zero / non-finite)."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        return None
    return v / n


class ActiveIdentitySet:
    """In-memory gallery of live identities (handoff §5: prototypes/medoids).

    Per gid we keep a bounded bank of recent UNIT exemplar embeddings. Scoring a
    query blends the prototype (bank mean) with the strongest single exemplar
    (medoid-style) -- ported verbatim from service.py::_bank_score, which is more
    robust than a mean alone when a person flips front/back or pose-shifts.
    Per-camera time spans back the same-camera overlap veto and lane routing.
    """

    def __init__(self, bank_size=20):
        self.bank_size = max(1, int(bank_size))
        self._banks = {}       # gid -> deque[unit np.ndarray] (maxlen bank_size)
        self._spans = {}       # gid -> {cam: [[t0, t1], ...]} wall-clock seconds
        self._obs = {}         # gid -> int observation count
        self._last_ts = {}     # gid -> float last time seen (any camera)
        self._last_cam = {}    # gid -> str camera of the most recent observation

    # ---- membership / read ------------------------------------------------
    def gids(self):
        return list(self._banks.keys())

    def has(self, gid):
        return gid in self._banks

    def cameras(self, gid):
        return set(self._spans.get(gid, {}).keys())

    def obs_count(self, gid):
        return self._obs.get(gid, 0)

    def last_seen(self, gid):
        """(camera, ts) of this identity's most recent observation, or None."""
        if gid not in self._last_ts:
            return None
        return self._last_cam.get(gid), self._last_ts[gid]

    # ---- write ------------------------------------------------------------
    def add_observation(self, gid, cam, unit_emb, ts, ts_start=None):
        """Record one UNIT embedding + its (cam, ts) against `gid`, extending the
        camera span over [ts_start or ts, ts]. `unit_emb` must already be
        L2-normalized (or None to only extend the span/bookkeeping without a bank
        entry). `ts_start` lets a track that just crossed the evidence gate open
        its span back at its FIRST observation, so the overlap veto also covers
        the provisional window (a co-present second body is still vetoed)."""
        if gid not in self._banks:
            self._banks[gid] = collections.deque(maxlen=self.bank_size)
            self._spans[gid] = {}
            self._obs[gid] = 0
        if unit_emb is not None:
            self._banks[gid].append(np.asarray(unit_emb, dtype=np.float32))
        self._obs[gid] += 1
        self._last_ts[gid] = ts
        self._last_cam[gid] = cam
        self._record_span(gid, cam, ts if ts_start is None else ts_start, ts)

    def _record_span(self, gid, cam, t0, t1):
        spans = self._spans[gid].setdefault(cam, [])
        # A start within 1s of the last span end is treated as the same continuous
        # visit (mirrors the file path's +1 frame tolerance) so a handoff between
        # two track ids in one camera stays one span rather than fragmenting.
        if spans and t0 <= spans[-1][1] + 1.0:
            spans[-1][1] = max(spans[-1][1], t1)
        else:
            spans.append([t0, t1])

    def extend_span(self, gid, cam, ts):
        """Extend the current camera span to `ts` without adding a bank entry
        (an already-assigned track seen again with no fresh embedding)."""
        if gid in self._spans:
            self._record_span(gid, cam, ts, ts)
            self._last_ts[gid] = ts
            self._last_cam[gid] = cam

    # ---- scoring (prototype + medoid-style exemplar) ----------------------
    def score(self, gid, unit_query):
        """max(prototype-mean score, strongest-exemplar score). Ported from
        service.py::_bank_score. `unit_query` must be L2-normalized. None if the
        bank is empty."""
        bank = self._banks.get(gid)
        if not bank:
            return None
        vecs = np.stack(list(bank)).astype(np.float32)     # already unit vectors
        q = np.asarray(unit_query, dtype=np.float32).ravel()
        proto = np.mean(vecs, axis=0)
        pn = np.linalg.norm(proto)
        proto_score = float((proto / pn) @ q) if pn > 0 else None
        exemplar_score = float(np.max(vecs @ q))            # medoid-style nearest
        if proto_score is None:
            return exemplar_score
        return max(proto_score, exemplar_score)

    def prototype(self, gid):
        """L2-normalized bank mean (this identity's representative vector)."""
        bank = self._banks.get(gid)
        if not bank:
            return None
        return _unit(np.mean(np.stack(list(bank)), axis=0))

    # ---- physical veto ----------------------------------------------------
    def same_camera_overlap(self, gid, cam, ts):
        """True if `gid` already has a span in `cam` that STRICTLY contains `ts`
        -- it is co-present here via another track, so it cannot also be this new
        track (two bodies). Boundary touches are allowed so a track-id handoff
        can reuse the identity."""
        for (t0, t1) in self._spans.get(gid, {}).get(cam, []):
            if t0 < ts < t1:
                return True
        return False

    # ---- eviction (bound memory; handoff limits block) --------------------
    def sweep(self, now, ttl_sec, max_active):
        """Evict identities unseen for `ttl_sec` seconds, then LRU-cap the set at
        `max_active`. Returns the number evicted. TTL is generous by design: an
        identity must survive a person's absence for cold reactivation to work,
        so eviction only bounds memory, it is not part of the match logic."""
        evicted = 0
        if ttl_sec and ttl_sec > 0:
            stale = [g for g, t in self._last_ts.items() if (now - t) > ttl_sec]
            for g in stale:
                self._drop(g)
                evicted += 1
        if max_active and len(self._banks) > max_active:
            # drop the least-recently-seen down to the cap
            order = sorted(self._last_ts.items(), key=lambda kv: kv[1])
            for g, _ in order[:len(self._banks) - int(max_active)]:
                self._drop(g)
                evicted += 1
        return evicted

    def _drop(self, gid):
        self._banks.pop(gid, None)
        self._spans.pop(gid, None)
        self._obs.pop(gid, None)
        self._last_ts.pop(gid, None)
        self._last_cam.pop(gid, None)


class IdentityEngine:
    """Per-observation decision core (thread-free, deterministic).

    Call `assign(cam, track_id, embedding, crop_quality, ts, has_fresh_emb)` once
    per detection per frame; it returns the reid to display now (a NEGATIVE
    provisional id while gathering evidence, a POSITIVE stable gid once resolved).
    """

    def __init__(self, min_evidence_obs=3, same_camera_threshold=0.90,
                 cross_camera_threshold=0.63, accept_margin=0.03,
                 bank_size=20, active_ttl_sec=300.0, max_active_identities=200,
                 topology=None):
        self.min_obs = max(1, int(min_evidence_obs))
        self.same_thr = float(same_camera_threshold)
        self.cross_thr = float(cross_camera_threshold)
        self.margin = float(accept_margin)
        self.active_ttl_sec = float(active_ttl_sec)
        self.max_active = int(max_active_identities)
        self.topology = topology or FailOpenTopology()

        self.store = ActiveIdentitySet(bank_size=bank_size)
        self._tracks = {}          # (cam, track_id) -> track state dict
        self._next_gid = 1
        self._next_prov = -1
        # counters (surfaced by the stage for metrics)
        self.minted = 0
        self.reacquired = 0        # same-camera cold reactivations
        self.linked = 0            # cross-camera links

    # ---- public API -------------------------------------------------------
    def assign(self, cam, track_id, embedding, crop_quality, ts, has_fresh_emb):
        """Resolve one detection to a reid. Sticky once assigned."""
        key = (cam, track_id)
        st = self._tracks.get(key)
        if st is None:
            st = {"gid": self._mint_provisional(), "status": "provisional",
                  "sum": None, "w": 0.0, "obs": 0, "first_ts": ts, "last_ts": ts}
            self._tracks[key] = st
        st["last_ts"] = ts

        if not has_fresh_emb or embedding is None:
            # No new evidence this frame -> keep the current (sticky) id and, if
            # already assigned, extend its active-camera window so the overlap
            # veto stays correct while the person is still on screen.
            if st["status"] == "assigned":
                self.store.extend_span(st["gid"], cam, ts)
            return st["gid"]

        unit = _unit(embedding)
        if unit is not None:
            w = max(bbox_quality_scalar(crop_quality), 0.05)   # floor: never zero
            st["sum"] = (unit * w) if st["sum"] is None else (st["sum"] + unit * w)
            st["w"] += w
        st["obs"] += 1

        if st["status"] == "provisional":
            if st["obs"] >= self.min_obs:
                self._resolve(cam, st, ts)
        else:                                   # assigned -> reinforce, stay sticky
            self._reinforce(cam, st, unit, ts)
        return st["gid"]

    def sweep(self, now):
        """Evict cold identities + stale track state (call periodically)."""
        self.store.sweep(now, self.active_ttl_sec, self.max_active)
        if self.active_ttl_sec and self.active_ttl_sec > 0:
            stale = [k for k, s in self._tracks.items()
                     if (now - s["last_ts"]) > self.active_ttl_sec]
            for k in stale:
                del self._tracks[k]

    # ---- aggregate / resolve / reinforce ----------------------------------
    def _aggregate(self, st):
        if st["sum"] is None or st["w"] <= 0:
            return None
        return _unit(st["sum"] / st["w"])

    def _resolve(self, cam, st, ts):
        """Evidence gate reached: match-or-mint on the quality-weighted aggregate,
        then seed the active set with this track's evidence + span."""
        agg = self._aggregate(st)
        if agg is None:
            st["obs"] = self.min_obs - 1        # degenerate; wait for more
            return
        gid = self._match(cam, agg, ts)
        if gid is None:
            gid = self._mint()
        st["gid"] = gid
        st["status"] = "assigned"
        # Open the span back at the track's first observation so the overlap veto
        # covers the whole time it has been on screen, not just from resolve on.
        self.store.add_observation(gid, cam, agg, ts, ts_start=st["first_ts"])

    def _reinforce(self, cam, st, unit, ts):
        """An assigned track keeps its id (sticky) but feeds the identity's bank +
        span with fresh evidence so the prototype/medoid stays current."""
        self.store.add_observation(st["gid"], cam, unit, ts)

    # ---- matching (two-lane, false-merge-conservative) --------------------
    def _match(self, cam, agg, ts):
        """Return a gid to reuse, or None to mint. Mirrors the proven
        service.py::_two_lane_match on the in-memory active set.

        Order matters: the SAME-camera reactivation lane is tried FIRST (a
        dropped track re-entering its own camera is the common case and gets a
        strict threshold), then the CROSS-camera lane (lower threshold, but
        guarded by a runner-up margin + reciprocal-best + topology). Uncertain ->
        None (mint). The same-camera time-overlap HARD veto is applied while
        gathering candidates, so a co-present identity is never a candidate.
        """
        scored = {}
        for gid in self.store.gids():
            if self.store.same_camera_overlap(gid, cam, ts):
                continue                      # co-present here -> a different body
            s = self.store.score(gid, agg)
            if s is not None:
                scored[gid] = s
        if not scored:
            return None

        # --- SAME-CAMERA lane: cold reactivation of a dropped-then-reacquired
        # track (its prior span in this camera is time-disjoint -- it passed the
        # overlap veto above). Strict threshold; no cross-camera margin needed.
        same_cam = [g for g in scored if cam in self.store.cameras(g)]
        if same_cam:
            best = max(same_cam, key=lambda g: scored[g])
            if scored[best] >= self.same_thr:
                self.reacquired += 1
                return best

        # --- CROSS-CAMERA lane: link the same person across cameras. Lower
        # threshold than same-camera, but a merge only survives if it clears
        # cross_thr, BEATS the runner-up by `margin` (no ambiguous crowd),
        # is RECIPROCAL-best (we are the candidate's best partner too), and the
        # topology veto ALLOWS the transition. Any failure -> mint (conservative).
        cross_cam = [g for g in scored if any(cc != cam for cc in self.store.cameras(g))]
        if cross_cam:
            ranked = sorted(cross_cam, key=lambda g: scored[g], reverse=True)
            best = ranked[0]
            best_s = scored[best]
            runner = scored[ranked[1]] if len(ranked) > 1 else -1.0
            if (best_s >= self.cross_thr
                    and (best_s - runner) >= self.margin
                    and self._reciprocal_best_ok(best, agg, cam, ts, best_s)
                    and self._topology_ok(best, cam, ts)):
                self.linked += 1
                return best

        return None

    def _reciprocal_best_ok(self, gid_star, agg, cam, ts, our_score):
        """Reciprocal-best guard (ported from service.py::_reciprocal_best_ok):
        score gid_star's OWN prototype against every other active identity (minus
        those co-present in this camera) and confirm none is a better partner for
        it than we are. Stops one track absorbing a look-alike crowd when scores
        are compressed. Fail-safe: no prototype -> allow."""
        proto = self.store.prototype(gid_star)
        if proto is None:
            return True
        best_other = -1.0
        for gid in self.store.gids():
            if gid == gid_star or self.store.same_camera_overlap(gid, cam, ts):
                continue
            s = self.store.score(gid, proto)
            if s is not None:
                best_other = max(best_other, s)
        return our_score >= best_other

    def _topology_ok(self, gid, cam, ts):
        """Cross-camera transition veto (pluggable; fail-open by default). Asks
        the topology model whether a person last seen in gid's most-recent camera
        at that time could plausibly be here now. FailOpenTopology allows all."""
        ls = self.store.last_seen(gid)
        if ls is None:
            return True
        src_cam, src_ts = ls
        return self.topology.allowed(src_cam, cam, src_ts, ts)

    # ---- id minting -------------------------------------------------------
    def _mint(self):
        gid = self._next_gid
        self._next_gid += 1
        self.minted += 1
        return gid

    def _mint_provisional(self):
        pid = self._next_prov
        self._next_prov -= 1
        return pid
