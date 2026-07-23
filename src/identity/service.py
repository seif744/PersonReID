"""
service.py  --  STAGE 7:  Identity Service. Assigns stable global_ids.

============================ WHAT THIS DOES ================================
This is the ONE component allowed to decide WHO a person is. Given a track's
appearance embedding + metadata (camera, track_id, frame), it returns a
global_id: a number that is the SAME for the same real person across cameras and
across time (even after they leave and reappear as a new track_id).

It is the "candidate -> decide -> commit" loop from DESIGN.md:
    1. candidate : ask the vector store for the nearest stored embeddings
    2. decide    : best candidate above threshold -> reuse its global_id;
                   otherwise MINT a new global_id (a person we haven't matched)
    3. commit    : store this observation under the chosen global_id

WHY the model and the store must NOT do this (recap): the model only sees one
crop (no context); the store only knows vector distances (no camera map, no
clock). Only here do we combine appearance with WHERE and WHEN -- so only here
can identity be decided. See DESIGN.md.

------------------------------- KEY CHOICE --------------------------------
Decision is made ONCE PER TRACK, on its first observation, then cached. A track
already has continuity (ByteTrack follows it frame-to-frame); re-deciding every
frame would let the id flicker. So (camera, track_id) -> global_id is sticky.

------------------------------- V1 SCOPE ----------------------------------
This version decides on APPEARANCE + a threshold only. We proved in Phase 4
that appearance alone has overlapping tails (same-min 0.51 < diff-max 0.85), so
this WILL make mistakes on hard cases; the durable fix is a stronger, domain-
appropriate embedding model, not more heuristics here. One physical guard is
kept: an identity cannot be reused for a track that overlaps it in time within
the same camera (two co-present tracks are two different people).

Threadsafety: this class is NOT thread-safe (mints ids, mutates a dict). When
wired into the multi-camera main.py, guard assign() with a lock -- like the
model and store locks already there.
============================================================================
"""

from collections import defaultdict, deque

import numpy as np

from identity.reranking import CameraAwareReranker
from identity.verifier import Verifier, bbox_quality_scalar


class IdentityService:
    def __init__(self, store, threshold: float = 0.85, top_k: int = 10,
                 bank_size: int = 20, min_score_gap: float = 0.03,
                 rerank_cfg: dict = None, verification_cfg: dict = None):
        """
        store     : a PersonVectorStore used as the identity gallery.
        threshold : minimum cosine score to accept a candidate as the SAME
                    person. Too high -> the same person gets split into new ids;
                    too low -> different people get MERGED. There is no perfect
                    value (the distributions overlap) -- this is why the
                    constraint hooks exist.

                    This default of 0.85 is a conservative generic fallback,
                    biased HIGH on purpose: a merge (two people fused into one
                    id) corrupts both their histories and is the worst error;
                    a split (one person given a spare id) is recoverable. So we
                    prefer to mint a new id when unsure. This value is
                    DATA-AND-MODEL-DEPENDENT -- calibrate it per deployment
                    with the threshold sweep in demo_identity.py, and re-measure
                    whenever the ReID checkpoint changes. config.yaml overrides
                    this default with the value actually calibrated for the
                    checkpoint currently in use (see identity.threshold there,
                    and ARCHITECTURE.md §6/§7).
        top_k     : how many nearest candidates to pull before deciding.
        bank_size : how many recent/good embeddings to keep per global_id for
                    prototype scoring.
        min_score_gap : require the best identity to beat the second-best by
                        this margin, reducing ambiguous look-alike merges.
        rerank_cfg : optional dict enabling ADR-002 Upgrade 1 (camera-aware
                     k-reciprocal + Jaccard re-ranking, see reranking.py).
                     {enabled, k1, cross_camera_k1_boost, lambda_}
        verification_cfg : optional dict enabling ADR-002 Upgrade 2 (a scored
                     verification layer replacing the plain threshold/gap gate,
                     see verifier.py). {enabled, accept_threshold, min_prob_gap,
                     log_path}. When disabled, decisions fall back to the plain
                     threshold/min_score_gap check above -- the original,
                     already-validated behaviour.
        """
        self.store = store
        self.threshold = threshold
        self.top_k = top_k
        self.bank_size = max(1, int(bank_size))
        self.min_score_gap = max(0.0, float(min_score_gap))
        self._track_reassign_warmup = 5

        self._track_to_global = {}   # (camera, track_id) -> global_id  (sticky)
        self._track_points = defaultdict(list)   # (camera, run_id, track_id) -> point ids
        self._track_embeddings = defaultdict(list) # same key -> embeddings for possible reassignment
        self._track_frames = defaultdict(list)    # same key -> frame indices
        # global_id -> camera -> list of [start_frame, end_frame] spans.
        # Live matching uses this to reject same-camera overlap before a merge.
        self._spans = defaultdict(lambda: defaultdict(list))
        self._banks = defaultdict(lambda: deque(maxlen=self.bank_size))
        self._obs_count = defaultdict(int)          # gid -> total commits ever
        self._last_seen = {}                        # gid -> (camera, frame_index)

        rerank_cfg = rerank_cfg or {}
        self._reranker = None
        if rerank_cfg.get("enabled"):
            self._reranker = CameraAwareReranker(
                k1=rerank_cfg.get("k1", 8),
                cross_camera_k1_boost=rerank_cfg.get("cross_camera_k1_boost", 4),
                lambda_=rerank_cfg.get("lambda_", 0.7),
            )

        verification_cfg = verification_cfg or {}
        self._verifier = None
        if verification_cfg.get("enabled"):
            self._verifier = Verifier(
                accept_threshold=verification_cfg.get("accept_threshold", 0.55),
                min_prob_gap=verification_cfg.get("min_prob_gap", 0.05),
                log_path=verification_cfg.get("log_path"),
            )

        self._load_existing_state()
        self._next_global_id = self._initial_next_global_id()

    def assign(self, camera: str, track_id: int, embedding, frame_index: int,
               run_id=None, crop_quality=None) -> int:
        """
        Return the global_id for this observation, deciding it if the track is
        new. Also commits the observation to the gallery under that global_id.
        """
        key = (camera, track_id)

        # Track already identified -> keep its id (stability). We still commit
        # this fresh observation so the gallery accumulates more views of them.
        if key in self._track_to_global:
            gid = self._track_to_global[key]
            gid = self._commit(camera, track_id, embedding, frame_index, gid,
                               run_id, crop_quality)
            gid = self._maybe_reassign_track(
                camera, track_id, embedding, frame_index, run_id, crop_quality, gid)
            return gid

        # New track -> match against the gallery, or mint a new identity.
        gid = self._match_or_mint(camera, embedding, frame_index, run_id, crop_quality)
        self._track_to_global[key] = gid
        gid = self._commit(camera, track_id, embedding, frame_index, gid, run_id,
                           crop_quality)
        return gid

    def _match_or_mint(self, camera: str, embedding, frame_index: int,
                        run_id=None, crop_quality=None) -> int:
        """Find the best already-identified candidate; accept it or mint new."""
        candidates = self.store.search(embedding, limit=self.top_k)

        candidate_gids = set()
        fallback_scores = defaultdict(lambda: -1.0)
        for hit in candidates:
            payload = hit.payload or {}
            # Never let detections from the exact same camera frame vote on
            # each other. They are simultaneous observations, so one fresh
            # track cannot be used to identify another track in that same frame.
            if payload.get("camera") == camera and payload.get("frame") == frame_index:
                continue

            gid = payload.get("reid_id", payload.get("global_id"))
            if gid is None:
                continue   # a raw observation not yet identified -- skip
            gid = int(gid)

            # Never let an identity win if it was already present in the same
            # camera at an overlapping time. That would collapse two simultaneous
            # people into one global id. This is a HARD physical exclusion --
            # kept in front of both the plain and verifier-based decision paths,
            # never overridden by a soft signal (ADR-002 Principle 1).
            if self._same_camera_overlap(gid, camera, frame_index):
                continue

            candidate_gids.add(gid)
            fallback_scores[gid] = max(fallback_scores[gid], float(hit.score))

        cosine_scores = {}
        for gid in candidate_gids:
            bank_score = self._bank_score(gid, embedding)
            cosine_scores[gid] = bank_score if bank_score is not None else fallback_scores[gid]

        if self._verifier is not None:
            return self._decide_with_verifier(camera, embedding, frame_index,
                                               candidate_gids, cosine_scores,
                                               run_id, crop_quality)

        scored = sorted(((score, gid) for gid, score in cosine_scores.items()), reverse=True)
        if scored:
            best_score, best_gid = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else -1.0
            clear_enough = (best_score - second_score) >= self.min_score_gap
            if best_score >= self.threshold and clear_enough:
                return best_gid        # matched an existing person
        return self._mint()            # new (or unmatchable) person

    def _decide_with_verifier(self, camera, embedding, frame_index,
                               candidate_gids, cosine_scores, run_id, crop_quality):
        """ADR-002 Upgrade 2 path: score each candidate with the verification
        layer (optionally fed by Upgrade 1's re-ranked score) instead of the
        plain threshold/gap check."""
        rerank_scores = {}
        if self._reranker is not None and candidate_gids:
            prototypes = {
                gid: self._prototype_vector(gid)
                for gid in self._banks
                if self._prototype_vector(gid) is not None
            }
            cameras = {gid: self._primary_camera(gid) for gid in prototypes}
            rerank_scores = self._reranker.rerank(
                embedding, camera, candidate_gids, prototypes, cameras)

        quality_scalar = bbox_quality_scalar(crop_quality)

        probs = {}
        for gid in candidate_gids:
            cosine = cosine_scores.get(gid, -1.0)
            last_seen_camera, last_seen_run, last_seen_frame = self._last_seen.get(
                gid, (None, None, None))
            # frame_index only means "how long ago" within the SAME run of the
            # SAME camera -- a fresh run restarts frame numbering from 0, so
            # comparing across runs (or camera names reused for a different
            # video) would produce meaningless/negative gaps. Only compute a
            # real gap when both camera AND run_id match; otherwise treat it
            # as unknown (neutral 0), same as if we'd never seen this gid.
            time_since_last = (
                frame_index - last_seen_frame
                if last_seen_camera == camera and last_seen_run == run_id
                and last_seen_frame is not None
                else 0
            )
            features = {
                "gid": gid,
                "cosine": cosine,
                "rerank_score": rerank_scores.get(gid, cosine),
                "observation_count": self._obs_count.get(gid, 0),
                "time_since_last_obs": time_since_last,
                "bbox_quality": quality_scalar,
            }
            probs[gid] = self._verifier.score(features)

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        if ranked:
            best_gid, best_prob = ranked[0]
            second_prob = ranked[1][1] if len(ranked) > 1 else -1.0
            clear_enough = (best_prob - second_prob) >= self._verifier.min_prob_gap
            if best_prob >= self._verifier.accept_threshold and clear_enough:
                return best_gid
        return self._mint()

    def _mint(self) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    def _commit(self, camera, track_id, embedding, frame_index, gid, run_id=None,
                crop_quality=None):
        """Store this observation under its global_id (grows the gallery)."""
        key = (camera, run_id, track_id)
        payload = {
            "camera": camera,
            "track_id": track_id,
            "frame": frame_index,
            "global_id": gid,
            "reid_id": gid,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        if crop_quality is not None:
            payload["crop_quality"] = crop_quality
        point_id = self.store.add(embedding, payload)
        self._track_points[key].append(point_id)
        self._track_embeddings[key].append(np.asarray(embedding, dtype=np.float32).ravel())
        self._track_frames[key].append(int(frame_index))
        self._add_to_bank(gid, embedding)
        self._record_span(gid, camera, frame_index)
        self._obs_count[int(gid)] += 1
        self._last_seen[int(gid)] = (camera, run_id, frame_index)
        return gid

    def _add_to_bank(self, gid, embedding):
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        if norm <= 0:
            return
        self._banks[int(gid)].append(emb / norm)

    def _bank_score(self, gid, embedding):
        """
        Score this query against one identity's feature bank.

        The old prototype-only score can blur front/back or pose-shifted views
        into a mean vector that is less similar than any real exemplar. To keep
        the live matcher robust when a track flips from front to back view (or
        changes track ids and comes back looking different), blend the
        prototype score with the strongest exemplar match.
        """
        bank = self._banks.get(int(gid))
        if not bank:
            return None
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        if norm <= 0:
            return None
        emb = emb / norm
        bank_vecs = np.stack(list(bank)).astype(np.float32)
        proto = np.mean(bank_vecs, axis=0)
        proto_norm = np.linalg.norm(proto)
        proto_score = None
        if proto_norm > 0:
            proto = proto / proto_norm
            proto_score = float(proto @ emb)

        exemplar_norms = np.linalg.norm(bank_vecs, axis=1, keepdims=True)
        bank_vecs = bank_vecs / np.clip(exemplar_norms, 1e-12, None)
        exemplar_score = float(np.max(bank_vecs @ emb))

        if proto_score is None:
            return exemplar_score
        return max(proto_score, exemplar_score)

    def _prototype_vector(self, gid):
        """L2-normalized mean of this identity's bank -- used by the reranker
        as its "gallery" representation. None if the bank is empty."""
        bank = self._banks.get(int(gid))
        if not bank:
            return None
        proto = np.mean(np.stack(list(bank)), axis=0)
        norm = np.linalg.norm(proto)
        if norm <= 0:
            return None
        return proto / norm

    def _primary_camera(self, gid):
        """The camera where this identity has the most observed frames --
        used by the reranker to decide same-camera vs cross-camera pairs."""
        cams = self._spans.get(int(gid))
        if not cams:
            return None
        return max(cams, key=lambda cam: sum(end - start + 1 for start, end in cams[cam]))

    def _load_existing_state(self):
        """Warm banks, span caches, observation counts, and last-seen frames
        from a persistent store."""
        try:
            offset = None
            while True:
                pts, offset = self.store.client.scroll(
                    self.store.collection, limit=1000, offset=offset,
                    with_payload=True, with_vectors=True)
                for p in pts:
                    payload = p.payload or {}
                    gid = payload.get("reid_id", payload.get("global_id"))
                    if gid is None:
                        continue
                    gid = int(gid)
                    vector = p.vector
                    if isinstance(vector, dict):
                        vector = next(iter(vector.values()), None)
                    if vector is not None:
                        self._add_to_bank(gid, vector)
                    camera = payload.get("camera")
                    frame = payload.get("frame")
                    run_id = payload.get("run_id")
                    self._obs_count[gid] += 1
                    if camera is not None and frame is not None:
                        frame = int(frame)
                        self._record_span(gid, camera, frame)
                        # run_id is a sortable "YYYYMMDD_HHMMSS" timestamp, so a
                        # later run_id IS a later observation regardless of its
                        # (restarted-from-0) frame numbering. Prefer the latest
                        # run_id; within the same run, prefer the latest frame.
                        prev = self._last_seen.get(gid)
                        prev_run = prev[1] if prev else None
                        prev_frame = prev[2] if prev else -1
                        is_newer_run = (run_id or "") > (prev_run or "")
                        is_same_run_later_frame = run_id == prev_run and frame > prev_frame
                        if prev is None or is_newer_run or is_same_run_later_frame:
                            self._last_seen[gid] = (camera, run_id, frame)
                if offset is None:
                    break
        except Exception:
            self._banks.clear()
            self._spans.clear()
            self._obs_count.clear()
            self._last_seen.clear()

    def _record_span(self, gid, camera, frame_index):
        """Extend this identity's per-camera time spans with one observation."""
        spans = self._spans[int(gid)][camera]
        if spans and frame_index <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], frame_index)
        else:
            spans.append([frame_index, frame_index])

    def _same_camera_overlap(self, gid, camera, frame_index):
        """
        True if this identity already has any observation span in `camera` that
        strictly contains `frame_index`. Boundary touches are allowed so a
        handoff between two track ids can still reuse the same identity.
        """
        spans = self._spans.get(int(gid), {}).get(camera, [])
        for start, end in spans:
            if start < frame_index < end:
                return True
        return False

    def _maybe_reassign_track(self, camera, track_id, embedding, frame_index,
                               run_id, crop_quality, current_gid):
        """
        Re-check a young track against the gallery before it hardens into a bad
        new identity.

        If the current gid has only a tiny amount of support, we allow a better
        existing identity to replace it once a later embedding makes the match
        clear. This catches the common failure mode where the first crop for a
        new ByteTrack id is a bad back-view or partial occlusion.
        """
        if current_gid is None:
            return None

        current_obs = self._obs_count.get(int(current_gid), 0)
        if current_obs > self._track_reassign_warmup:
            return current_gid

        candidates = self.store.search(embedding, limit=self.top_k)
        fallback_scores = defaultdict(lambda: -1.0)
        candidate_gids = set()
        for hit in candidates:
            payload = hit.payload or {}
            if payload.get("camera") == camera and payload.get("frame") == frame_index:
                continue
            gid = payload.get("reid_id", payload.get("global_id"))
            if gid is None:
                continue
            gid = int(gid)
            if gid == current_gid:
                continue
            if self._same_camera_overlap(gid, camera, frame_index):
                continue
            candidate_gids.add(gid)
            fallback_scores[gid] = max(fallback_scores[gid], float(hit.score))

        if not candidate_gids:
            return current_gid

        scores = {}
        for gid in candidate_gids:
            bank_score = self._bank_score(gid, embedding)
            scores[gid] = bank_score if bank_score is not None else fallback_scores[gid]

        if not scores:
            return current_gid

        best_score, best_gid = max((score, gid) for gid, score in scores.items())
        current_score = self._bank_score(current_gid, embedding)
        if current_score is None:
            current_score = -1.0

        # Only switch if the alternative is clearly better.
        if best_score < self.threshold:
            return current_gid
        if best_score < (current_score + self.min_score_gap):
            return current_gid

        key = (camera, run_id, track_id)
        point_ids = self._track_points.get(key, [])
        if point_ids:
            self.store.set_global_id(point_ids, best_gid)

        # Move the track's observations over to the new identity state so future
        # matching sees the corrected id, not the mistaken mint.
        embeddings = self._track_embeddings.get(key, [])
        frames = self._track_frames.get(key, [])
        for emb, frame in zip(embeddings, frames):
            self._add_to_bank(best_gid, emb)
            self._record_span(best_gid, camera, frame)
        self._obs_count[int(best_gid)] += len(embeddings)
        if frames:
            self._last_seen[int(best_gid)] = (camera, run_id, max(frames))

        self._track_to_global[key] = best_gid
        self._drop_identity_if_unreferenced(current_gid)
        return best_gid

    def _drop_identity_if_unreferenced(self, gid):
        gid = int(gid)
        if gid <= 0:
            return
        if gid in self._track_to_global.values():
            return
        self._banks.pop(gid, None)
        self._spans.pop(gid, None)
        self._obs_count.pop(gid, None)
        self._last_seen.pop(gid, None)

    def _initial_next_global_id(self):
        """
        Continue after existing ids when using a persistent store. Otherwise a
        new run without --reset could mint GID 1 again for a different person.
        """
        max_gid = 0
        try:
            offset = None
            while True:
                pts, offset = self.store.client.scroll(
                    self.store.collection, limit=1000, offset=offset,
                    with_payload=True, with_vectors=False)
                for p in pts:
                    payload = p.payload or {}
                    gid = payload.get("reid_id", payload.get("global_id"))
                    if gid is not None:
                        max_gid = max(max_gid, int(gid))
                if offset is None:
                    break
        except Exception:
            max_gid = 0
        return max_gid + 1
