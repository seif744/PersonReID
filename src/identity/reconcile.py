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
