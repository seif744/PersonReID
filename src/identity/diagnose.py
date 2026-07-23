"""
diagnose.py  --  READ-ONLY cross-camera ReID failure investigation.

This script CHANGES NOTHING. It only reads the Qdrant gallery and prints
evidence, to answer: *which stage is losing the cross-camera match?* It does not
assign identities, write payloads, or touch config. Run:

    python src/identity/diagnose.py                 # latest run, threshold from config
    python src/identity/diagnose.py <run_id> <thr>  # pin a run and threshold

Ground truth (who is REALLY the same person across cameras) is not available, so
the script does not guess correctness. Instead it measures the *structure* that
determines whether a correct match COULD be found and accepted:

  - within-tracklet consistency        (upper bound: same person, ~same view)
  - cross-camera centroid similarity   (mostly different people; the sea)
  - for each cross-camera MUTUAL nearest-neighbour tracklet pair (the system's
    own best candidates for "same person"):
        mean-prototype cosine   vs   best-exemplar cosine   vs   threshold
    The gap (best_exemplar - mean) crossing the threshold is the decisive test:
      * gap large & straddles threshold -> AVERAGING is discarding the evidence
        (gallery/track representation), a medoid/multi-prototype would recover it.
      * both below threshold            -> the EMBEDDING isn't distinctive enough
        (appearance model / input quality), no downstream logic can fix it.
      * mean already >= threshold       -> retrieval/decision logic (why not merged?)
"""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_env_url():
    """Read QDRANT_URL from .env / environment (server is authoritative)."""
    url = os.environ.get("QDRANT_URL")
    if url:
        return url
    try:
        with open(os.path.join(os.getcwd(), ".env")) as f:
            for line in f:
                line = line.strip()
                if line.startswith("QDRANT_URL") and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return "http://localhost:6333"


def _unit(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _pct(a, ps=(5, 25, 50, 75, 95)):
    if len(a) == 0:
        return "n/a"
    a = np.asarray(a, dtype=np.float64)
    return "  ".join(f"p{p}={np.percentile(a, p):.3f}" for p in ps)


def _mean_proto(vecs):
    m = np.stack(vecs).mean(axis=0)
    return _unit(m)


def _medoid_clusters(vecs, merge=0.75):
    """Greedy viewpoint clustering (same logic idea as a medoid gallery) ->
    number of distinct clusters + the medoid vector of each. Used to measure
    viewpoint diversity and to compute best-exemplar similarity."""
    V = np.stack([_unit(v) for v in vecs])
    n = len(V)
    if n == 1:
        return V, [1]
    sims = V @ V.T
    order = np.argsort(-sims.sum(1))
    seeds = []
    assign = np.full(n, -1)
    for i in order:
        if not seeds:
            seeds.append(i); assign[i] = 0; continue
        ss = [sims[i, s] for s in seeds]
        b = int(np.argmax(ss))
        if ss[b] >= merge or len(seeds) >= 6:
            assign[i] = b
        else:
            assign[i] = len(seeds); seeds.append(i)
    protos, sizes = [], []
    for c in range(len(seeds)):
        idx = np.where(assign == c)[0]
        sub = sims[np.ix_(idx, idx)]
        med = idx[int(np.argmax(sub.sum(1)))]
        protos.append(V[med]); sizes.append(len(idx))
    return np.stack(protos), sizes


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_tracklets(client, collection, run_id):
    tracklets = defaultdict(lambda: {"vecs": [], "frames": [], "quals": [],
                                      "gids": set()})
    run_ids_seen = defaultdict(int)
    offset = None
    while True:
        pts, offset = client.scroll(collection, limit=1000, offset=offset,
                                    with_payload=True, with_vectors=True)
        for p in pts:
            pl = p.payload or {}
            rid = pl.get("run_id")
            run_ids_seen[rid] += 1
            if run_id is not None and rid != run_id:
                continue
            cam, tid, frame = pl.get("camera"), pl.get("track_id"), pl.get("frame")
            if cam is None or tid is None or frame is None:
                continue
            vec = p.vector
            if isinstance(vec, dict):
                vec = next(iter(vec.values()), None)
            if vec is None:
                continue
            key = (cam, int(tid))
            t = tracklets[key]
            t["vecs"].append(np.asarray(vec, dtype=np.float32))
            t["frames"].append(int(frame))
            t["quals"].append(pl.get("crop_quality") or {})
            gid = pl.get("reid_id", pl.get("global_id"))
            if gid is not None:
                t["gids"].add(int(gid))
        if offset is None:
            break
    return tracklets, run_ids_seen


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    from qdrant_client import QdrantClient

    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.63

    client = QdrantClient(url=_load_env_url())
    collection = "persons"

    # Pick the latest run_id if none given.
    tracklets, runs = load_tracklets(client, collection, None)
    all_runs = sorted(r for r in runs if r is not None)
    if run_id is None:
        run_id = all_runs[-1] if all_runs else None
        tracklets, _ = load_tracklets(client, collection, run_id)

    print("=" * 74)
    print("CROSS-CAMERA ReID DIAGNOSTIC  (read-only)")
    print("=" * 74)
    print(f"run_ids in store: {dict(runs)}")
    print(f"analysing run_id = {run_id}   threshold = {threshold}")

    # Precompute per-tracklet summaries.
    T = {}
    for key, t in tracklets.items():
        if not t["vecs"]:
            continue
        V = np.stack([_unit(v) for v in t["vecs"]])
        centroid = _unit(V.mean(0))
        within = float(np.mean((V @ centroid))) if len(V) else 0.0
        variance = float(np.mean(1.0 - (V @ centroid)))
        protos, sizes = _medoid_clusters(t["vecs"])
        blurs = [q.get("blur") for q in t["quals"] if q.get("blur") is not None]
        areas = [q.get("box_area_ratio") for q in t["quals"]
                 if q.get("box_area_ratio") is not None]
        T[key] = {
            "cam": key[0], "tid": key[1], "n": len(V),
            "span": (min(t["frames"]), max(t["frames"])),
            "dur": max(t["frames"]) - min(t["frames"]) + 1,
            "centroid": centroid, "protos": protos, "nproto": len(sizes),
            "consistency": within, "variance": variance,
            "gids": t["gids"],
            "mean_blur": float(np.mean(blurs)) if blurs else float("nan"),
            "mean_area": float(np.mean(areas)) if areas else float("nan"),
            "frames": set(t["frames"]),
        }

    cams = sorted({v["cam"] for v in T.values()})
    print(f"\n--- OVERVIEW ---")
    print(f"tracklets: {len(T)}   cameras: {cams}")
    for c in cams:
        ks = [k for k in T if T[k]["cam"] == c]
        print(f"  {c}: {len(ks)} tracklets, "
              f"{sum(T[k]['n'] for k in ks)} observations")
    gid_cams = defaultdict(set)
    for k, v in T.items():
        for g in v["gids"]:
            gid_cams[g].add(v["cam"])
    cross = sorted(g for g, cs in gid_cams.items() if len(cs) > 1)
    print(f"reid_ids: {len(gid_cams)}   cross-camera reid_ids: {len(cross)} {cross}")

    # ----- Q5 track representation -----
    print(f"\n--- Q5  TRACK REPRESENTATION (per tracklet) ---")
    print(f"{'cam':<16}{'tid':>7}{'n':>5}{'dur':>6}{'views':>7}"
          f"{'consist':>9}{'var':>7}{'blur':>8}{'area%':>8}  gids")
    for k in sorted(T, key=lambda k: (T[k]['cam'], -T[k]['n'])):
        v = T[k]
        print(f"{v['cam']:<16}{v['tid']:>7}{v['n']:>5}{v['dur']:>6}"
              f"{v['nproto']:>7}{v['consistency']:>9.3f}{v['variance']:>7.3f}"
              f"{v['mean_blur']:>8.1f}{100*v['mean_area']:>8.3f}  {sorted(v['gids'])}")
    variances = [v["variance"] for v in T.values()]
    print(f"  embedding variance across tracklets: {_pct(variances)}")

    # ----- Q1 embedding separability -----
    print(f"\n--- Q1  EMBEDDING SEPARABILITY ---")
    within = [v["consistency"] for v in T.values() if v["n"] > 1]
    print(f"within-tracklet consistency (same person, ~same view): {_pct(within)}")
    # cross-camera centroid cosine over ALL cross-camera tracklet pairs.
    keys = list(T)
    cross_pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if T[a]["cam"] == T[b]["cam"]:
                continue
            mean_cos = float(T[a]["centroid"] @ T[b]["centroid"])
            best_ex = float(np.max(T[a]["protos"] @ T[b]["protos"].T))
            cross_pairs.append((mean_cos, best_ex, a, b))
    print(f"cross-camera centroid cosine (all {len(cross_pairs)} pairs): "
          f"{_pct([c for c, _, _, _ in cross_pairs])}")
    print(f"cross-camera BEST-EXEMPLAR cosine (all pairs):           "
          f"{_pct([e for _, e, _, _ in cross_pairs])}")

    # system-believed same-person cross-camera pairs (share a reid_id)
    believed_same = [(c, e, a, b) for c, e, a, b in cross_pairs
                     if T[a]["gids"] & T[b]["gids"]]
    if believed_same:
        print(f"cross-camera pairs the system MERGED (share reid_id): "
              f"n={len(believed_same)}  "
              f"mean_cos {_pct([c for c, _, _, _ in believed_same])}")

    # ----- Q2/Q3/Q4 mutual-NN cross-camera analysis -----
    print(f"\n--- Q2/Q3/Q4  MUTUAL-NN CROSS-CAMERA PAIRS (candidate matches) ---")
    # best cross-camera partner for each tracklet, by mean-cos (what reconcile uses)
    best_partner = {}
    for k in keys:
        cc = [(float(T[k]["centroid"] @ T[o]["centroid"]), o)
              for o in keys if T[o]["cam"] != T[k]["cam"]]
        if cc:
            best_partner[k] = max(cc)
    mutual = set()
    for k, (s, o) in best_partner.items():
        if best_partner.get(o, (None, None))[1] == k:
            mutual.add(frozenset((k, o)))

    modes = defaultdict(int)
    rows = []
    for pair in mutual:
        a, b = tuple(pair)
        mean_cos = float(T[a]["centroid"] @ T[b]["centroid"])
        best_ex = float(np.max(T[a]["protos"] @ T[b]["protos"].T))
        overlap = bool(T[a]["frames"] & T[b]["frames"]) and T[a]["cam"] == T[b]["cam"]
        merged = bool(T[a]["gids"] & T[b]["gids"])
        if merged:
            mode = "MERGED"
        elif mean_cos >= threshold:
            mode = "retrieved>=thr,not merged (decision)"
        elif best_ex >= threshold > mean_cos:
            mode = "AVERAGING loses it (best-exemplar>=thr>mean)"
        else:
            mode = "below thr both (embedding/input)"
        modes[mode] += 1
        rows.append((best_ex - mean_cos, mean_cos, best_ex, mode, a, b))

    for gap, mc, be, mode, a, b in sorted(rows, reverse=True):
        print(f"  {T[a]['cam']}#{T[a]['tid']} <-> {T[b]['cam']}#{T[b]['tid']}: "
              f"mean={mc:.3f} best_ex={be:.3f} gap={gap:+.3f}  -> {mode}")
    print(f"\n  mutual-NN pair failure-mode counts:")
    for m, n in sorted(modes.items(), key=lambda x: -x[1]):
        print(f"    {n:>3}  {m}")

    # decisive aggregate: how many pairs would flip from split->merge if we scored
    # by best-exemplar (medoid) instead of mean, at the SAME threshold.
    flip = sum(1 for _, mc, be, _, _, _ in rows if mc < threshold <= be)
    print(f"\n  pairs that AVERAGING alone blocks (mean<thr<=best_exemplar): {flip}"
          f"  / {len(rows)} mutual-NN pairs")

    print("\n" + "=" * 74)
    print("Interpretation key:")
    print("  many 'AVERAGING loses it' / high flip count -> gallery+track")
    print("     representation is the bottleneck (medoids/multi-proto recover it).")
    print("  many 'below thr both' -> embedding/input quality is the ceiling")
    print("     (evaluate LightMBN / check crop quality); logic can't fix it.")
    print("  many 'retrieved>=thr,not merged' -> decision/reconcile logic.")
    print("=" * 74)


if __name__ == "__main__":
    main()
