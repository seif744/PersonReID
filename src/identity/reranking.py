"""
reranking.py -- ADR-002 Upgrade 1: camera-aware k-reciprocal + Jaccard re-ranking.

Standard k-reciprocal re-ranking (Zhong et al., "Re-ranking Person
Re-identification with k-reciprocal Encoding") judges two embeddings as similar
not just by raw cosine distance, but by how much their NEIGHBOURHOODS overlap:
if x and y keep showing up in each other's nearest-neighbour lists, and share
many of the same neighbours, that's a stronger signal than raw cosine alone --
it catches cases the raw metric misses under domain shift.

CA-Jaccard (CVPR'24) adds a camera-aware adjustment: cross-camera cosine
similarity runs systematically lower than same-camera similarity (lighting,
angle, domain gap), so a strict top-k neighbourhood check unfairly excludes
genuine cross-camera matches. The fix here is a WIDER neighbourhood window for
cross-camera reciprocity checks (`cross_camera_k1_boost`) -- NOT a score
multiplier. Multiplying cosine by a "camera weight" would just rescale scores
without changing which points count as neighbours, which is a different (and
explicitly rejected) algorithm.

This module re-ranks over per-identity PROTOTYPES (the mean-pooled,
re-normalized bank vectors IdentityService already maintains in memory),
because the underlying Qdrant collection only holds raw per-observation
points, not one point per identity. With a small number of identities (tens,
not millions), the full pairwise prototype similarity matrix is cheap on CPU
and is cached, rebuilt only when the identity roster/bank contents actually
change (see `_rebuild_if_needed`).
"""

import numpy as np


class CameraAwareReranker:
    def __init__(self, k1=8, cross_camera_k1_boost=4, lambda_=0.7):
        self.k1 = max(1, int(k1))
        self.cross_camera_k1_boost = max(0, int(cross_camera_k1_boost))
        self.lambda_ = float(lambda_)
        self._cache_sig = None
        self._neighbor_lists = {}   # gid -> [(score, other_gid), ...] sorted desc, gallery-only

    def _rebuild_if_needed(self, prototypes):
        # Cheap signature of "did the gallery change since last call" -- avoids
        # recomputing the O(N^2) neighbour graph every single detection.
        sig = tuple(sorted(
            (gid, round(float(np.sum(vec)), 5), round(float(vec[0]), 5))
            for gid, vec in prototypes.items()
        ))
        if sig == self._cache_sig:
            return
        self._cache_sig = sig

        gids = list(prototypes.keys())
        self._neighbor_lists = {}
        for gi in gids:
            sims = [
                (float(prototypes[gi] @ prototypes[gj]), gj)
                for gj in gids if gj != gi
            ]
            sims.sort(reverse=True)
            self._neighbor_lists[gi] = sims

    def _effective_k(self, gid, other_camera, cameras):
        """Widen k1 when the pair spans two different cameras."""
        return self.k1 if cameras.get(gid) == other_camera else self.k1 + self.cross_camera_k1_boost

    def _in_neighbor_set(self, gid, score, query_camera, cameras):
        """True if something scoring `score` against `gid` would fall within
        gid's (possibly camera-widened) top-k neighbours."""
        sims = self._neighbor_lists.get(gid, [])
        k = self._effective_k(gid, query_camera, cameras)
        if len(sims) < k:
            return True   # small gallery -- everyone counts as "close enough"
        kth_score = sims[k - 1][0]
        return score >= kth_score

    def _reciprocal_set_gallery(self, gid, cameras):
        """R(gid, k1), computed purely within the gallery (no query involved) --
        used for local query expansion and for scoring candidates' own sets."""
        camera = cameras.get(gid)
        sims = self._neighbor_lists.get(gid, [])
        window = self.k1 + self.cross_camera_k1_boost
        result = set()
        for score, other in sims[:window]:
            k = self._effective_k(other, camera, cameras)
            other_sims = self._neighbor_lists.get(other, [])
            if len(other_sims) < k or any(g == gid for _, g in other_sims[:k]):
                result.add(other)
        return result

    def rerank(self, query_embedding, query_camera, candidate_gids, prototypes, cameras):
        """
        prototypes : {gid: L2-normalized prototype vector}, ALL known identities
                     with a non-empty bank (not just candidates) -- needed to
                     build the neighbourhood graph.
        cameras    : {gid: primary camera string}
        candidate_gids : the subset we actually need re-ranked scores for.

        Returns {gid: blended_score} for each candidate present in prototypes.
        """
        if not candidate_gids or not prototypes:
            return {}

        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        qn = np.linalg.norm(q)
        if qn <= 0:
            return {}
        q = q / qn

        # With too few known identities, reciprocal-neighbour sets degenerate
        # (a candidate's own neighbourhood can be empty or just itself), which
        # makes the Jaccard overlap trivially "perfect" regardless of whether
        # it's a good match -- that inflates confidence exactly when there's
        # the LEAST evidence to justify it. Below this floor, skip the Jaccard
        # blend entirely and fall back to plain cosine.
        min_gallery_for_jaccard = self.k1 + 1
        if len(prototypes) < min_gallery_for_jaccard:
            return {
                gid: float(prototypes[gid] @ q)
                for gid in candidate_gids if gid in prototypes
            }

        self._rebuild_if_needed(prototypes)

        query_scores = {gid: float(prototypes[gid] @ q) for gid in prototypes}
        ranked_gallery = sorted(query_scores.items(), key=lambda kv: kv[1], reverse=True)

        # R(query, k1): gallery members that rank in the query's own top-k AND
        # that the query would fall into the neighbourhood of (two-sided
        # reciprocity), with camera-aware widening on both checks.
        query_top = set()
        for rank, (gid, _score) in enumerate(ranked_gallery):
            if rank < self._effective_k(gid, query_camera, cameras):
                query_top.add(gid)
        r_query = {
            gid for gid in query_top
            if self._in_neighbor_set(gid, query_scores[gid], query_camera, cameras)
        }

        # Local query expansion (standard k-reciprocal encoding): pull in each
        # reciprocal neighbour's own reciprocal set too, so the query's
        # neighbourhood signature isn't just its immediate matches.
        expanded = set(r_query)
        for gid in r_query:
            expanded |= self._reciprocal_set_gallery(gid, cameras)

        results = {}
        for gid in candidate_gids:
            if gid not in prototypes:
                continue
            r_cand = self._reciprocal_set_gallery(gid, cameras)
            r_cand.add(gid)   # a node is trivially in its own neighbourhood
            union = expanded | r_cand
            inter = expanded & r_cand
            jaccard_dist = 1.0 - (len(inter) / len(union) if union else 0.0)
            cosine = query_scores[gid]
            results[gid] = self.lambda_ * cosine + (1.0 - self.lambda_) * (1.0 - jaccard_dist)
        return results
