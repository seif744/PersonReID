"""
reconcile.py  --  STAGE 7b:  OFFLINE identity reconciliation.

============================ WHY THIS EXISTS ==============================
The live IdentityService (service.py) decides a track's global_id ONCE, on the
track's first observation, and never revisits it (stickiness prevents id flicker
frame-to-frame). That is correct for a single stream, but it has a blind spot the
moment two cameras run at once:

    A person who enters BOTH cameras at the same instant is embedded and decided
    in each camera before the other camera has committed anything to the gallery.
    Both cameras therefore MINT -- and, because the decision is sticky, they can
    never later discover they are the same person. The match window never existed.

That is exactly the failure this pass repairs. AFTER all camera workers finish
and the gallery is fully populated, we look back over every global_id, build an
appearance prototype for each, and MERGE ids that are the same person. This is a
batch, whole-gallery view the live path structurally cannot have.

------------------------------- SAFETY --------------------------------------
Merging is the most destructive identity error (it fuses two people's histories),
so this pass is conservative, matching service.py's "prefer a split" bias:

  * APPEARANCE GATE: two ids merge only if their prototype cosine >= threshold
    (default: the same threshold the live service uses).
  * PHYSICAL EXCLUSION: two ids are NEVER merged if they are co-present in the
    SAME camera (their tracks overlap in time). One body cannot be two
    simultaneous tracks in one view, so such a pair is provably two people --
    regardless of how alike they look (twins, uniforms). This also correctly
    still ALLOWS merging disjoint fragments of one person in a single camera
    (a dropped-then-reacquired track), which fixes over-counting too.
  * TRANSITIVE-SAFE: clusters grow by union only when NO member of one side
    conflicts with ANY member of the other, so a chain of merges can never
    sneak two co-present ids into the same identity.
  * RECIPROCAL BEST MATCH (optional, on by default): a pair merges only if each
    id is the OTHER's single best above-threshold partner. On out-of-domain
    footage the appearance model compresses scores -- many different people land
    in the same 0.8-0.9 band as the true match -- and a lone threshold then
    over-merges. Requiring mutual nearest-neighbour keeps the one real pair and
    rejects the look-alike crowd. Turn off only with a model whose same/different
    scores are cleanly separated (see the sweep in demo_identity.py).

Deterministic: the surviving id of a merged cluster is its smallest global_id,
and vectors/point-ids are never touched -- only the global_id payload is
re-stamped. Safe to re-run (idempotent): a reconciled gallery has no pairs left
to merge.
============================================================================
"""

from collections import defaultdict

import numpy as np

from identity.evidence import (
    EvidenceConfig,
    FrameObservation,
    TrackEvidence,
    color_similarity,
)


def _proto_sim(pa, pb):
    """
    Similarity between two tracklets (or clusters) represented by their MEDOID
    prototype sets. Each argument is a (P, D) array of L2-normalized prototypes;
    the score is the best match across every prototype pair -- so a front-view
    query matches a front-view exemplar even when the tracks' *mean* vectors
    (blurred across viewpoints) would not. This is the multi-prototype upgrade
    over the old single-mean cosine. Returns -1.0 when either side has no
    appearance evidence.
    """
    if pa.size == 0 or pb.size == 0:
        return -1.0
    return float(np.max(pa @ pb.T))


def _cluster_prototype(members, protos):
    """A cluster's prototype set is the STACK of its members' prototypes, so
    cross-cluster similarity (via _proto_sim) is the best match over every
    exemplar either side has seen. Empty -> (0, 0)."""
    mats = [protos[m] for m in members if protos[m].size]
    if not mats:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(mats)


def _set_color(members, ev_map):
    """Observation-weighted mean of members' colour descriptors, renormalized to
    a histogram. None if no member carries colour (or their sizes disagree) --
    treated as 'no colour evidence', never as a mismatch."""
    vecs, weights = [], []
    for m in members:
        ev = ev_map.get(m)
        if ev is None or ev.color_descriptor is None:
            continue
        vecs.append(np.asarray(ev.color_descriptor, dtype=np.float32).ravel())
        weights.append(max(1, ev.observation_count))
    if not vecs or len({v.shape[0] for v in vecs}) != 1:
        return None
    mat = np.stack(vecs)
    w = np.asarray(weights, dtype=np.float32)
    agg = (mat * w[:, None]).sum(axis=0)
    total = float(agg.sum())
    if total <= 0:
        return None
    return agg / total


def _spans_disjoint(span_a, span_b):
    """
    Cheap pre-filter only: True when two tracklets' (min, max) frame
    envelopes don't even overlap. NOT sufficient on its own to prove they
    coexisted -- two tracks can have overlapping envelopes while never
    sharing an actual frame (e.g. the tracker drops an id and reacquires the
    same person under a new one; the old track's tail and the new track's
    head can straddle without ever being simultaneous). Use
    `_tracks_overlap` for the real physical-exclusion decision.
    """
    return span_a[1] < span_b[0] or span_b[1] < span_a[0]


def _tracks_overlap(frames_a, frames_b):
    """True when two tracklets were actually observed in a shared frame --
    i.e. they provably coexisted in the same camera at the same instant."""
    return not frames_a.isdisjoint(frames_b)


def _gather_tracklets(store, run_id):
    """
    Read gallery observations grouped by camera-local tracklet.

    Returns tracklet -> {
        vectors: [embedding],
        points: [point ids],
        span: (min_frame, max_frame),
        frames: frozenset of observed frame numbers,
        gids: {original global ids},
        quals: [crop_quality dict|None],       # per observation, aligned to vectors
        colors: [color_descriptor list|None],  # per observation
        heights: [normalized_height float|None],# per observation
    }.
    """
    data = defaultdict(lambda: {
        "vectors": [],
        "points": [],
        "frames": [],
        "gids": set(),
        "quals": [],
        "colors": [],
        "heights": [],
    })

    offset = None
    while True:
        pts, offset = store.client.scroll(
            store.collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=True)
        for p in pts:
            pl = p.payload or {}
            if run_id is not None and pl.get("run_id") != run_id:
                continue
            camera = pl.get("camera")
            track_id = pl.get("track_id")
            frame = pl.get("frame")
            if camera is None or track_id is None or frame is None:
                continue

            vector = p.vector
            if isinstance(vector, dict):
                vector = next(iter(vector.values()), None)
            if vector is None:
                continue

            key = (camera, int(track_id))
            data[key]["vectors"].append(np.asarray(vector, dtype=np.float32))
            data[key]["points"].append(p.id)
            data[key]["frames"].append(int(frame))
            data[key]["quals"].append(pl.get("crop_quality"))
            data[key]["colors"].append(pl.get("color_descriptor"))
            data[key]["heights"].append(pl.get("normalized_height"))
            gid = pl.get("reid_id", pl.get("global_id"))
            if gid is not None:
                data[key]["gids"].add(int(gid))

        if offset is None:
            break

    out = {}
    for key, info in data.items():
        if not info["vectors"]:
            continue
        frames = info["frames"]
        out[key] = {
            "vectors": info["vectors"],
            "points": info["points"],
            "span": (min(frames), max(frames)),
            "frames": frozenset(frames),
            "frames_list": list(frames),   # per-observation, aligned to vectors
            "gids": info["gids"],
            "quals": info["quals"],
            "colors": info["colors"],
            "heights": info["heights"],
        }
    return out


def _build_evidence(key, info, cfg):
    """Turn one gathered tracklet into a TrackEvidence (the Track Evidence Layer
    entry point). Reconstructs per-observation FrameObservations from the stored
    payload fields, so the reconciler reasons over cleaned, multi-prototype
    track evidence instead of a bag of raw frame vectors."""
    camera, track_id = key
    obs = []
    for i, vec in enumerate(info["vectors"]):
        obs.append(FrameObservation(
            frame_index=info["frames_list"][i],
            camera_id=camera, track_id=track_id,
            embedding=vec,
            crop_quality=info["quals"][i],
            color_descriptor=info["colors"][i],
            normalized_height=info["heights"][i],
            point_id=info["points"][i],
        ))
    return TrackEvidence.from_observations(obs, cfg)


def _otsu_split(scores, bins=256):
    """
    Otsu's method: the score that maximizes between-class variance of a
    histogram split into two groups. Standard for finding the boundary
    between two modes in a distribution without assuming their shape.
    Returns None if the scores don't vary enough to bin meaningfully.
    """
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return None
    hist, edges = np.histogram(scores, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    idx = np.arange(bins)
    sum_total = float(np.sum(hist * idx))

    sum_below = 0.0
    weight_below = 0.0
    best_variance = -1.0
    best_bin = None
    for i in range(bins):
        weight_below += hist[i]
        if weight_below == 0:
            continue
        weight_above = total - weight_below
        if weight_above == 0:
            break
        sum_below += i * hist[i]
        mean_below = sum_below / weight_below
        mean_above = (sum_total - sum_below) / weight_above
        variance = weight_below * weight_above * (mean_below - mean_above) ** 2
        if variance > best_variance:
            best_variance = variance
            best_bin = i
    if best_bin is None:
        return None
    return float(edges[best_bin + 1])


def _calibrate_threshold(pairwise_scores, fallback, min_pairs=6,
                         lo_bound=0.35, hi_bound=0.92, log=print):
    """
    Derive this run's cross-camera match threshold from its OWN score
    distribution instead of a value hand-tuned on different footage.

    WHY: a fixed cosine threshold only works as long as new video stays in
    the domain (lighting, camera, compression, ReID checkpoint) the number
    was measured on -- see verifier.py's w0/w_cosine comment and this
    project's config.yaml. New footage shifts the whole score distribution,
    so the fixed number silently stops separating "same person" from
    "different person" and either misses real matches or over-merges.

    A gallery large enough to have cross-camera pairs at all is a mix of
    "different people" (low cosine) and "same person, two views" (high
    cosine) pairs -- Otsu's method finds the boundary between those two
    modes directly from the data, so it re-anchors itself every run.

    Falls back to `fallback` (the configured/default threshold) when there
    isn't enough evidence yet (too few pairs) or the split lands somewhere
    implausible for a ReID cosine score (outside [lo_bound, hi_bound]),
    which usually means the pairs are all one mode (nothing to split).
    """
    if len(pairwise_scores) < min_pairs:
        log(f"  tracklet reconcile: only {len(pairwise_scores)} cross-camera "
            f"pairs (< {min_pairs}), using fallback threshold {fallback:.3f}")
        return fallback

    split = _otsu_split(pairwise_scores)
    if split is None or not (lo_bound <= split <= hi_bound):
        log(f"  tracklet reconcile: auto-calibration split "
            f"({split}) outside plausible range, using fallback "
            f"threshold {fallback:.3f}")
        return fallback

    log(f"  tracklet reconcile: auto-calibrated threshold {split:.3f} "
        f"from {len(pairwise_scores)} cross-camera pairs "
        f"(fallback would have been {fallback:.3f})")
    return split


def reconcile_tracklets(store, threshold=None, run_id=None,
                        same_camera_threshold=0.90,
                        require_reciprocal_best=True,
                        min_tracklet_observations=1, log=print,
                        fallback_threshold=0.63,
                        color_min_similarity=0.15,
                        evidence_config=None):
    """
    Rebuild global ids from camera-local tracklets.

    It does not trust the existing global_id buckets, because a bad live
    assignment can already contain several people. The tracklet -- one camera's
    view of one track -- is the unit of evidence, now expressed as a
    `TrackEvidence` object (the Track Evidence Layer): each tracklet is cleaned
    (quality-weighted, outlier-pruned) and summarized as a small set of MEDOID
    prototypes, so matching compares the best of one track's real viewpoints
    against the best of another's rather than two blurred means.

    threshold : cross-camera match threshold. Pass a float to pin it (the old
        behaviour). Pass None (the default) to auto-calibrate it from this
        run's own cross-camera score distribution instead -- see
        `_calibrate_threshold` -- so a new video's different lighting/camera/
        compression doesn't require hand-editing a config value.

    min_tracklet_observations : a real identity needs sustained observation. A
        tracklet with fewer stored observations than this is treated as detector
        noise / an unusable fragment: it is marked UNIDENTIFIED (its global_id is
        cleared) rather than becoming its own person, so it does not inflate the
        head-count. Its vectors stay in the gallery. 1 = keep everything.

    color_min_similarity : CNN-INDEPENDENT veto. Two tracklets are never merged
        if BOTH carry a torso-colour descriptor (reid/color.py) and their colour
        histogram intersection is below this -- someone in a red shirt is not
        someone in a blue one, no matter how the embedding scores them. Colour
        can only BLOCK a merge, never create one (respecting the false-merge
        bias); it is skipped when either side lacks colour. 0.0 disables it.

    evidence_config : optional EvidenceConfig controlling how each tracklet is
        cleaned/summarized (quality floor, outlier strictness, medoid count).
    """
    cfg = evidence_config or EvidenceConfig()
    auto_calibrate = threshold is None
    tracklets = _gather_tracklets(store, run_id)
    all_keys = sorted(tracklets)

    # Suppress spurious tracklets before clustering. One- or two-frame tracks are
    # almost always a missed/duplicate detection, too short to embed reliably;
    # letting each become an identity is what inflates the person count.
    suppressed = [k for k in all_keys
                  if len(tracklets[k]["vectors"]) < min_tracklet_observations]
    for k in suppressed:
        store.clear_global_id(tracklets[k]["points"])
        log(f"  tracklet reconcile: suppressed {k} "
            f"({len(tracklets[k]['vectors'])} obs < {min_tracklet_observations})")

    keys = [k for k in all_keys if k not in set(suppressed)]
    if len(keys) < 2:
        return {}

    # Build the Track Evidence Layer's view of each surviving tracklet.
    ev_map = {k: _build_evidence(k, tracklets[k], cfg) for k in keys}
    protos = {k: (ev_map[k].prototypes if ev_map[k] is not None
                  else np.empty((0, 0), dtype=np.float32)) for k in keys}
    keys = [k for k in keys if protos[k].size]

    parent = {k: k for k in keys}
    members = {k: {k} for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def conflict(set_a, set_b):
        for a in set_a:
            for b in set_b:
                if a[0] != b[0]:
                    continue
                if _spans_disjoint(tracklets[a]["span"], tracklets[b]["span"]):
                    continue
                if _tracks_overlap(tracklets[a]["frames"], tracklets[b]["frames"]):
                    return True
        return False

    def color_conflict(set_a, set_b):
        """CNN-independent veto: block a merge when both sides have colour
        evidence and it clearly disagrees. Missing colour -> no veto."""
        if color_min_similarity <= 0:
            return False
        sim = color_similarity(_set_color(set_a, ev_map), _set_color(set_b, ev_map))
        return sim != -1.0 and sim < color_min_similarity

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        ma, mb = members[ra], members[rb]
        if conflict(ma, mb) or color_conflict(ma, mb):
            return False
        winner, loser = (ra, rb) if ra < rb else (rb, ra)
        parent[loser] = winner
        members[winner] = ma | mb
        members.pop(loser, None)
        return True

    if auto_calibrate:
        cross_camera_scores = [
            _proto_sim(protos[a], protos[b])
            for i, a in enumerate(keys)
            for b in keys[i + 1:]
            if a[0] != b[0] and not conflict({a}, {b})
        ]
        threshold = _calibrate_threshold(
            cross_camera_scores, fallback_threshold, log=log)
    else:
        threshold = fallback_threshold

    # Phase 1: repair same-camera track fragmentation with a higher threshold.
    same_pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] != b[0]:
                continue
            if _tracks_overlap(tracklets[a]["frames"], tracklets[b]["frames"]):
                continue
            s = _proto_sim(protos[a], protos[b])
            if s >= same_camera_threshold:
                same_pairs.append((s, a, b))
    for s, a, b in sorted(same_pairs, reverse=True):
        if union(a, b):
            log(f"  tracklet reconcile: same-camera merge {a} + {b} "
                f"(cosine {s:.3f})")

    def current_roots():
        roots = defaultdict(set)
        for k in keys:
            roots[find(k)].add(k)
        return roots

    # Phase 2 runs in ROUNDS, re-scoring from scratch each round, so a chain of
    # 3+ mutually-similar fragments (e.g. A's best match is B, but B's own best
    # match is C, not A) can fully consolidate instead of stalling after one
    # pass. Each round still requires reciprocal-best (when enabled) using the
    # CURRENT cluster prototypes -- merging never gets easier than the safety
    # rule allows, it just gets to re-check after each merge updates the
    # clusters. A round that merges nothing ends the loop; every merge strictly
    # reduces the number of roots by one, so this always terminates.
    while True:
        roots = current_roots()
        root_protos = {r: _cluster_prototype(ms, protos) for r, ms in roots.items()}
        root_keys = sorted(r for r, p in root_protos.items() if p.size)

        def mergeable_cross(a, b):
            if conflict(roots[a], roots[b]) or color_conflict(roots[a], roots[b]):
                return False
            return any(x[0] != y[0] for x in roots[a] for y in roots[b])

        root_scores = {}
        for i, a in enumerate(root_keys):
            for b in root_keys[i + 1:]:
                if not mergeable_cross(a, b):
                    continue
                root_scores[(a, b)] = _proto_sim(root_protos[a], root_protos[b])

        def root_score(a, b):
            return root_scores[(a, b)] if a < b else root_scores[(b, a)]

        best_partner = {}
        if require_reciprocal_best:
            for a in root_keys:
                candidates = [
                    (root_score(a, b), b) for b in root_keys
                    if b != a
                    and ((a, b) in root_scores or (b, a) in root_scores)
                    and root_score(a, b) >= threshold
                ]
                if candidates:
                    best_partner[a] = max(candidates)[1]

        cross_pairs = []
        for (a, b), s in root_scores.items():
            if s < threshold:
                continue
            if require_reciprocal_best and not (
                    best_partner.get(a) == b and best_partner.get(b) == a):
                continue
            cross_pairs.append((s, a, b))

        merged_this_round = False
        for s, a, b in sorted(cross_pairs, reverse=True):
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if union(ra, rb):
                log(f"  tracklet reconcile: cross-camera merge {a} + {b} "
                    f"(cosine {s:.3f})")
                merged_this_round = True

        if not merged_this_round:
            break

    final_roots = current_roots()
    all_gids = sorted(
        gid for info in tracklets.values() for gid in info["gids"])
    next_gid = (max(all_gids) + 1) if all_gids else 1
    used_survivors = set()
    remap = {}
    for root, cluster in sorted(final_roots.items()):
        existing = sorted(
            gid for k in cluster for gid in tracklets[k]["gids"])
        survivor = None
        for gid in existing:
            if gid not in used_survivors:
                survivor = gid
                break
        if survivor is None:
            survivor = next_gid
            next_gid += 1
        used_survivors.add(survivor)
        for k in sorted(cluster):
            point_ids = tracklets[k]["points"]
            store.set_global_id(point_ids, survivor)
            remap[k] = survivor

    log(f"  tracklet reconcile: {len(keys)} tracklets -> "
        f"{len(set(remap.values()))} identities.")
    return remap
