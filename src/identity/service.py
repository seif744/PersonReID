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
This first version decides on APPEARANCE + a threshold only. We proved in Phase 4
that appearance alone has overlapping tails (same-min 0.51 < diff-max 0.85), so
this WILL make mistakes on hard cases. The camera-topology / temporal / motion
constraints that fix those are marked with CONSTRAINT HOOK below -- structured
in, implemented later, exactly as DESIGN.md specifies.

Threadsafety: this class is NOT thread-safe (mints ids, mutates a dict). When
wired into the multi-camera main.py, guard assign() with a lock -- like the
model and store locks already there.
============================================================================
"""

from collections import defaultdict, deque

import numpy as np


class IdentityService:
    def __init__(self, store, threshold: float = 0.85, top_k: int = 10,
                 bank_size: int = 20, min_score_gap: float = 0.03):
        """
        store     : a PersonVectorStore used as the identity gallery.
        threshold : minimum cosine score to accept a candidate as the SAME
                    person. Too high -> the same person gets split into new ids;
                    too low -> different people get MERGED. There is no perfect
                    value (the distributions overlap) -- this is why the
                    constraint hooks exist.

                    Default 0.85 is biased HIGH on purpose: a merge (two people
                    fused into one id) corrupts both their histories and is the
                    worst error; a split (one person given a spare id) is
                    recoverable. So we prefer to mint a new id when unsure. This
                    value is DATA-DEPENDENT -- calibrate it per deployment with
                    the threshold sweep in demo_identity.py (for our dev crops,
                    0.85 gave 0 merges).
        top_k     : how many nearest candidates to pull before deciding.
        bank_size : how many recent/good embeddings to keep per global_id for
                    prototype scoring.
        min_score_gap : require the best identity to beat the second-best by
                        this margin, reducing ambiguous look-alike merges.
        """
        self.store = store
        self.threshold = threshold
        self.top_k = top_k
        self.bank_size = max(1, int(bank_size))
        self.min_score_gap = max(0.0, float(min_score_gap))

        self._track_to_global = {}   # (camera, track_id) -> global_id  (sticky)
        self._banks = defaultdict(lambda: deque(maxlen=self.bank_size))
        self._load_existing_banks()
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
            self._commit(camera, track_id, embedding, frame_index, gid, run_id,
                         crop_quality)
            return gid

        # New track -> match against the gallery, or mint a new identity.
        gid = self._match_or_mint(camera, embedding, frame_index)
        self._track_to_global[key] = gid
        self._commit(camera, track_id, embedding, frame_index, gid, run_id,
                     crop_quality)
        return gid

    def _match_or_mint(self, camera: str, embedding, frame_index: int) -> int:
        """Find the best already-identified candidate; accept it or mint new."""
        candidates = self.store.search(embedding, limit=self.top_k)

        candidate_gids = set()
        fallback_scores = defaultdict(lambda: -1.0)
        for hit in candidates:
            gid = (hit.payload or {}).get("global_id")
            if gid is None:
                continue   # a raw observation not yet identified -- skip
            gid = int(gid)
            candidate_gids.add(gid)
            fallback_scores[gid] = max(fallback_scores[gid], float(hit.score))

            # ---- CONSTRAINT HOOK (DESIGN.md) -- not yet implemented ----------
            # Before trusting appearance, a real system would GATE this candidate:
            #   * temporal:  reject if this global_id was seen elsewhere at a time
            #                that makes being here now impossible.
            #   * topology:  reject if the candidate's last camera has no path to
            #                `camera` within the elapsed frames/time.
            #   * motion:    down-weight if the trajectory is inconsistent.
            # These turn "closest vector" into "closest PLAUSIBLE person". Until
            # they exist, v1 trusts appearance alone and will err on look-alikes.
            # ------------------------------------------------------------------

        scored = []
        for gid in candidate_gids:
            bank_score = self._bank_score(gid, embedding)
            score = bank_score if bank_score is not None else fallback_scores[gid]
            scored.append((score, gid))
        scored.sort(reverse=True)

        if scored:
            best_score, best_gid = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else -1.0
            clear_enough = (best_score - second_score) >= self.min_score_gap
            if best_score >= self.threshold and clear_enough:
                return best_gid        # matched an existing person
        return self._mint()            # new (or unmatchable) person

    def _mint(self) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    def _commit(self, camera, track_id, embedding, frame_index, gid, run_id=None,
                crop_quality=None):
        """Store this observation under its global_id (grows the gallery)."""
        payload = {
            "camera": camera,
            "track_id": track_id,
            "frame": frame_index,
            "global_id": gid,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        if crop_quality is not None:
            payload["crop_quality"] = crop_quality
        self.store.add(embedding, payload)
        self._add_to_bank(gid, embedding)

    def _add_to_bank(self, gid, embedding):
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        if norm <= 0:
            return
        self._banks[int(gid)].append(emb / norm)

    def _bank_score(self, gid, embedding):
        bank = self._banks.get(int(gid))
        if not bank:
            return None
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        if norm <= 0:
            return None
        emb = emb / norm
        proto = np.mean(np.stack(list(bank)), axis=0)
        proto_norm = np.linalg.norm(proto)
        if proto_norm <= 0:
            return None
        proto = proto / proto_norm
        return float(proto @ emb)

    def _load_existing_banks(self):
        """Warm banks from a persistent store, capped per global_id."""
        try:
            offset = None
            while True:
                pts, offset = self.store.client.scroll(
                    self.store.collection, limit=1000, offset=offset,
                    with_payload=True, with_vectors=True)
                for p in pts:
                    gid = (p.payload or {}).get("global_id")
                    if gid is None:
                        continue
                    vector = p.vector
                    if isinstance(vector, dict):
                        vector = next(iter(vector.values()), None)
                    if vector is not None:
                        self._add_to_bank(gid, vector)
                if offset is None:
                    break
        except Exception:
            self._banks.clear()

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
                    gid = (p.payload or {}).get("global_id")
                    if gid is not None:
                        max_gid = max(max_gid, int(gid))
                if offset is None:
                    break
        except Exception:
            max_gid = 0
        return max_gid + 1
