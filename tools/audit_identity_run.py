"""
Audit one identity run from Qdrant.

This is a debugging tool, not part of the live pipeline. It answers:
  * which camera-local tracks ended up under each global_id
  * which cross-camera identity pairs are above a similarity threshold
  * whether the reciprocal-best reconciliation rule would accept each pair

Examples:
  python tools/audit_identity_run.py --run-id 20260708_124059
  python tools/audit_identity_run.py --run-id 20260708_124059 --threshold 0.87
"""

import argparse
from collections import defaultdict
import os
import sys

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from database.store import PersonVectorStore  # noqa: E402


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prototype(vectors):
    mat = np.stack(vectors).astype(np.float32)
    mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
    proto = mat.mean(axis=0)
    norm = np.linalg.norm(proto)
    if norm <= 0:
        return None
    return proto / norm


def gather(store, run_id):
    tracks = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    vectors = defaultdict(list)

    offset = None
    while True:
        pts, offset = store.client.scroll(
            store.collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in pts:
            payload = p.payload or {}
            if payload.get("run_id") != run_id:
                continue

            gid = payload.get("global_id")
            if gid is None:
                continue
            gid = int(gid)

            camera = payload.get("camera")
            track_id = payload.get("track_id")
            frame = payload.get("frame")
            if camera is not None and track_id is not None and frame is not None:
                tracks[gid][camera][track_id].append(int(frame))

            vector = p.vector
            if isinstance(vector, dict):
                vector = next(iter(vector.values()), None)
            if vector is not None:
                vectors[gid].append(np.asarray(vector, dtype=np.float32))

        if offset is None:
            break

    return tracks, vectors


def score_lookup(protos):
    gids = sorted(protos)
    scores = {}
    for i, a in enumerate(gids):
        for b in gids[i + 1:]:
            scores[(a, b)] = float(protos[a] @ protos[b])
    return scores


def score(scores, a, b):
    return scores[(a, b)] if a < b else scores[(b, a)]


def best_partners(gids, scores, threshold):
    best = {}
    for gid in gids:
        candidates = [
            (score(scores, gid, other), other)
            for other in gids
            if other != gid and score(scores, gid, other) >= threshold
        ]
        if candidates:
            best[gid] = max(candidates)[1]
    return best


def print_tracks(tracks):
    for gid in sorted(tracks):
        cameras = tracks[gid]
        obs = sum(
            len(frames)
            for cam_tracks in cameras.values()
            for frames in cam_tracks.values()
        )
        print(f"GID {gid}: cameras={sorted(cameras)} observations={obs}")
        for camera in sorted(cameras):
            bits = []
            for track_id, frames in sorted(cameras[camera].items()):
                bits.append(
                    f"track {int(track_id):04d} frames {min(frames)}-{max(frames)} "
                    f"({len(frames)} obs)"
                )
            print(f"  {camera}: " + "; ".join(bits))


def print_cross_camera_candidates(tracks, protos, threshold, top):
    gids = sorted(protos)
    scores = score_lookup(protos)
    best = best_partners(gids, scores, threshold)

    rows = []
    for i, a in enumerate(gids):
        cams_a = set(tracks[a])
        for b in gids[i + 1:]:
            cams_b = set(tracks[b])
            if not cams_a or not cams_b or not cams_a.isdisjoint(cams_b):
                continue
            pair_score = score(scores, a, b)
            rows.append((pair_score, a, b, cams_a, cams_b))

    rows.sort(reverse=True)
    print(f"\nTop cross-camera candidates (threshold={threshold:.3f})")
    for pair_score, a, b, cams_a, cams_b in rows[:top]:
        mutual = best.get(a) == b and best.get(b) == a
        status = "MERGE" if pair_score >= threshold and mutual else "skip"
        print(
            f"{status:5} GID {a} <-> {b}: score={pair_score:.4f} "
            f"mutual_best={mutual} cameras={sorted(cams_a)} vs {sorted(cams_b)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Audit ReID identity assignments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    store_cfg = cfg.get("store", {})
    id_cfg = cfg.get("identity", {})
    url = os.environ.get("QDRANT_URL") or store_cfg.get("url") or None
    api_key = os.environ.get("QDRANT_API_KEY") or None
    path = store_cfg.get("path", "qdrant_data")
    threshold = args.threshold if args.threshold is not None else id_cfg.get("threshold", 0.85)

    store = PersonVectorStore(path=path, url=url, api_key=api_key)
    tracks, vectors = gather(store, args.run_id)
    protos = {
        gid: proto
        for gid, proto in ((gid, prototype(vs)) for gid, vs in vectors.items())
        if proto is not None
    }

    total_obs = sum(
        len(frames)
        for cameras in tracks.values()
        for cam_tracks in cameras.values()
        for frames in cam_tracks.values()
    )
    print(f"run_id={args.run_id} global_ids={len(tracks)} observations={total_obs}")
    print_tracks(tracks)
    print_cross_camera_candidates(tracks, protos, threshold, args.top)


if __name__ == "__main__":
    main()
