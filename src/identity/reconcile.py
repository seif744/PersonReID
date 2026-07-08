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


def _prototype(vectors):
    """Mean of L2-normalized vectors, renormalized -- the id's appearance center."""
    mat = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    proto = mat.mean(axis=0)
    n = np.linalg.norm(proto)
    if n <= 0:
        return None
    return proto / n


def _spans_overlap(a, b):
    """Do two (min_frame, max_frame) inclusive intervals intersect?"""
    return a[0] <= b[1] and b[0] <= a[1]


def _co_present(spans_x, spans_y):
    """
    True if ids x and y are ever in the SAME camera at overlapping times.
    spans_* : {camera: [(min_frame, max_frame), ...]}  (one span per track).
    """
    for camera, xs in spans_x.items():
        ys = spans_y.get(camera)
        if not ys:
            continue
        for sx in xs:
            for sy in ys:
                if _spans_overlap(sx, sy):
                    return True
    return False


def _spans_disjoint(span_a, span_b):
    """True when two same-camera tracklets do not overlap in time."""
    return span_a[1] < span_b[0] or span_b[1] < span_a[0]


def _gather_tracklets(store, run_id):
    """
    Read gallery observations grouped by camera-local tracklet.

    Returns tracklet -> {
        vectors: [embedding],
        points: [point ids],
        span: (min_frame, max_frame),
        gids: {original global ids},
    }.
    """
    data = defaultdict(lambda: {
        "vectors": [],
        "points": [],
        "frames": [],
        "gids": set(),
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
            gid = pl.get("global_id")
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
            "gids": info["gids"],
        }
    return out


def _cluster_prototype(members, protos):
    return _prototype([protos[m] for m in members])


def reconcile_tracklets(store, threshold, run_id=None,
                        same_camera_threshold=0.90,
                        require_reciprocal_best=True,
                        min_tracklet_observations=1, log=print):
    """
    Rebuild global ids from camera-local tracklets.

    This is stricter than reconcile_global_ids(): it does not trust existing
    global_id buckets, because a bad live assignment can already contain several
    people. The tracklet is the unit of evidence.

    min_tracklet_observations : a real identity needs sustained observation. A
        tracklet with fewer stored observations than this is treated as detector
        noise / an unusable fragment: it is marked UNIDENTIFIED (its global_id is
        cleared) rather than becoming its own person, so it does not inflate the
        head-count. Its vectors stay in the gallery. 1 = keep everything.
    """
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

    protos = {k: _prototype(tracklets[k]["vectors"]) for k in keys}
    keys = [k for k in keys if protos[k] is not None]

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
                if not _spans_disjoint(tracklets[a]["span"], tracklets[b]["span"]):
                    return True
        return False

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        ma, mb = members[ra], members[rb]
        if conflict(ma, mb):
            return False
        winner, loser = (ra, rb) if ra < rb else (rb, ra)
        parent[loser] = winner
        members[winner] = ma | mb
        members.pop(loser, None)
        return True

    # Phase 1: repair same-camera track fragmentation with a higher threshold.
    same_pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] != b[0]:
                continue
            if not _spans_disjoint(tracklets[a]["span"], tracklets[b]["span"]):
                continue
            s = float(protos[a] @ protos[b])
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

    roots = current_roots()
    root_protos = {r: _cluster_prototype(ms, protos) for r, ms in roots.items()}
    root_keys = sorted(r for r, p in root_protos.items() if p is not None)

    def mergeable_cross(a, b):
        if conflict(roots[a], roots[b]):
            return False
        return any(x[0] != y[0] for x in roots[a] for y in roots[b])

    root_scores = {}
    for i, a in enumerate(root_keys):
        for b in root_keys[i + 1:]:
            if not mergeable_cross(a, b):
                continue
            root_scores[(a, b)] = float(root_protos[a] @ root_protos[b])

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

    for s, a, b in sorted(cross_pairs, reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if union(ra, rb):
            log(f"  tracklet reconcile: cross-camera merge {a} + {b} "
                f"(cosine {s:.3f})")

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


def _gather(store, run_id):
    """
    Read the gallery once. Returns per global_id:
        vectors : list of stored embeddings
        points  : list of point ids
        spans   : {camera: [(min_frame, max_frame) per track]}
        cameras : set of cameras the id appears in
    Scoped to run_id when given, so a persistent multi-run store is untouched.
    """
    vectors = defaultdict(list)
    points = defaultdict(list)
    # gid -> camera -> track_id -> [min_frame, max_frame]
    track_span = defaultdict(lambda: defaultdict(dict))
    cameras = defaultdict(set)

    offset = None
    while True:
        pts, offset = store.client.scroll(
            store.collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=True)
        for p in pts:
            pl = p.payload or {}
            if run_id is not None and pl.get("run_id") != run_id:
                continue
            gid = pl.get("global_id")
            if gid is None:
                continue
            gid = int(gid)

            vector = p.vector
            if isinstance(vector, dict):          # named-vector collections
                vector = next(iter(vector.values()), None)
            if vector is None:
                continue
            vectors[gid].append(np.asarray(vector, dtype=np.float32))
            points[gid].append(p.id)

            camera = pl.get("camera")
            cameras[gid].add(camera)
            frame = pl.get("frame")
            if camera is not None and frame is not None:
                tid = pl.get("track_id")
                span = track_span[gid][camera].get(tid)
                if span is None:
                    track_span[gid][camera][tid] = [frame, frame]
                else:
                    span[0] = min(span[0], frame)
                    span[1] = max(span[1], frame)
        if offset is None:
            break

    spans = {
        gid: {cam: list(tracks.values()) for cam, tracks in cams.items()}
        for gid, cams in track_span.items()
    }
    return vectors, points, spans, cameras


def reconcile_global_ids(store, threshold, run_id=None,
                         block_same_camera_overlap=True,
                         require_reciprocal_best=True, log=print):
    """
    Merge global_ids that are the same real person across the whole gallery.

    store      : PersonVectorStore (the gallery).
    threshold  : prototype cosine at/above which two ids are the same person.
    run_id     : restrict to one run's points (recommended for persistent stores).
    block_same_camera_overlap : enforce the physical-exclusion guard (keep True in
                 production; expose only for testing).
    require_reciprocal_best : only merge a pair that are each other's best
                 above-threshold partner (guards against look-alike over-merging
                 on weak/out-of-domain models). Keep True unless the model's
                 same/different score distributions are cleanly separated.
    log        : where progress lines go (print, or a logger's .info).

    Returns {losing_gid: surviving_gid} for every id that was remapped (empty if
    nothing merged). Only payloads are rewritten; vectors/point-ids are untouched.
    """
    vectors, points, spans, cameras = _gather(store, run_id)
    gids = sorted(vectors)
    if len(gids) < 2:
        return {}

    protos = {g: _prototype(vectors[g]) for g in gids}
    gids = [g for g in gids if protos[g] is not None]

    sim = {(gi, gj): float(protos[gi] @ protos[gj])
           for i, gi in enumerate(gids) for gj in gids[i + 1:]}

    def score(a, b):
        return sim[(a, b)] if a < b else sim[(b, a)]

    # Each id's single best above-threshold partner (for reciprocal matching).
    best_partner = {}
    for g in gids:
        others = [(score(g, o), o) for o in gids if o != g and score(g, o) >= threshold]
        if others:
            best_partner[g] = max(others)[1]

    # Every candidate pair, most-similar first, so the strongest evidence drives
    # the clustering order.
    pairs = []
    for (gi, gj), s in sim.items():
        if s < threshold:
            continue
        if require_reciprocal_best and not (
                best_partner.get(gi) == gj and best_partner.get(gj) == gi):
            continue
        pairs.append((s, gi, gj))
    pairs.sort(reverse=True)

    # Agglomerate with a union-find that respects the exclusion constraint at the
    # CLUSTER level: only union if no member conflicts with any member.
    parent = {g: g for g in gids}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    members = {g: {g} for g in gids}

    def cluster_conflict(ma, mb):
        if not block_same_camera_overlap:
            return False
        for a in ma:
            for b in mb:
                if _co_present(spans.get(a, {}), spans.get(b, {})):
                    return True
        return False

    for s, gi, gj in pairs:
        ra, rb = find(gi), find(gj)
        if ra == rb:
            continue
        ma, mb = members[ra], members[rb]
        if cluster_conflict(ma, mb):
            log(f"  reconcile: kept {gi} and {gj} apart "
                f"(cosine {s:.3f} >= {threshold} but co-present in a camera)")
            continue
        # Merge the smaller-id root wins, for deterministic surviving ids.
        winner, loser = (ra, rb) if ra < rb else (rb, ra)
        parent[loser] = winner
        members[winner] = ma | mb
        members.pop(loser, None)

    # Build the remap: every gid whose root differs from itself.
    remap = {}
    for g in gids:
        root = find(g)
        if root != g:
            remap[g] = root

    if not remap:
        log(f"  reconcile: {len(gids)} identities, no merges.")
        return {}

    # Apply: re-stamp the losing ids' points with the surviving id, batched by
    # surviving id (one set_payload call per cluster).
    survivor_to_losers = defaultdict(list)
    for loser, survivor in remap.items():
        survivor_to_losers[survivor].append(loser)

    for survivor, losers in sorted(survivor_to_losers.items()):
        point_ids = []
        for loser in losers:
            point_ids.extend(points[loser])
        store.set_global_id(point_ids, survivor)
        cams = sorted(
            c for g in [survivor, *losers] for c in cameras.get(g, set())
            if c is not None
        )
        log(f"  reconcile: merged {sorted(losers)} -> {survivor} "
            f"({len(point_ids)} observations; cameras {cams})")

    log(f"  reconcile: {len(gids)} -> {len(gids) - len(remap)} identities "
        f"after merging {len(remap)}.")
    return remap
