"""
test_reconcile.py  --  integration tests for the TrackEvidence-based reconciler.

Uses a throwaway in-memory Qdrant (path=":memory:"), no ReID model -- synthetic
embeddings stand in for appearance. This is the guardrail that the reconcile.py
refactor (single-mean -> multi-prototype TrackEvidence + colour veto) still
produces correct cross-camera merges, per ENGINEERING_HANDOFF.md's acceptance
criterion "offline reconciliation should remain compatible".

Runs under pytest or standalone:
    python src/identity/test_reconcile.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.store import PersonVectorStore
from identity.reconcile import reconcile_tracklets

RNG = np.random.default_rng(1234)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _near(base, jitter=0.01):
    return _unit(base + jitter * RNG.standard_normal(512).astype(np.float32))


def _good_quality():
    return {"accepted": True, "blur": 120.0, "brightness": 128.0,
            "box_area_ratio": 0.05, "occlusion": 0.0}


def _add_tracklet(store, *, camera, track_id, gid, frames, base,
                  color=None, run_id="r0"):
    """Insert one tracklet: N observations near `base`, all under one gid."""
    for f in frames:
        payload = {
            "camera": camera, "track_id": track_id, "frame": f,
            "global_id": gid, "reid_id": gid, "run_id": run_id,
            "crop_quality": _good_quality(), "normalized_height": 0.4,
        }
        if color is not None:
            payload["color_descriptor"] = [float(x) for x in color]
        store.add(_near(base), payload)


def _final_gids(store, run_id="r0"):
    """Map (camera, track_id) -> reid_id after reconciliation."""
    out = {}
    offset = None
    while True:
        pts, offset = store.client.scroll(
            store.collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if pl.get("run_id") != run_id:
                continue
            out[(pl.get("camera"), pl.get("track_id"))] = pl.get("reid_id")
        if offset is None:
            break
    return out


def _color(idx):
    """A 32-d colour histogram concentrated in one bin (distinct idx = distinct
    colour). Kept sharp so two different colours have ~0 histogram intersection,
    isolating the veto logic from fixture noise."""
    c = np.zeros(32, dtype=np.float32)
    c[idx] = 1.0
    return c


# --------------------------------------------------------------------------- #
def test_same_person_two_cameras_merges():
    store = PersonVectorStore(path=":memory:")
    person = _unit(RNG.standard_normal(512))
    other = _unit(RNG.standard_normal(512))
    _add_tracklet(store, camera="A", track_id=1, gid=1, frames=range(0, 10), base=person)
    _add_tracklet(store, camera="B", track_id=1, gid=2, frames=range(0, 10), base=person)
    _add_tracklet(store, camera="A", track_id=2, gid=3, frames=range(20, 30), base=other)

    reconcile_tracklets(store, threshold=0.63, run_id="r0",
                        min_tracklet_observations=1, require_reciprocal_best=True,
                        color_min_similarity=0.0, log=lambda *a: None)

    gids = _final_gids(store)
    # The two views of `person` collapse to one id; `other` keeps its own.
    assert gids[("A", 1)] == gids[("B", 1)]
    assert gids[("A", 2)] != gids[("A", 1)]
    assert len(set(gids.values())) == 2


def test_different_people_do_not_merge():
    store = PersonVectorStore(path=":memory:")
    p = _unit(RNG.standard_normal(512))
    q = _unit(RNG.standard_normal(512))
    _add_tracklet(store, camera="A", track_id=1, gid=1, frames=range(0, 10), base=p)
    _add_tracklet(store, camera="B", track_id=1, gid=2, frames=range(0, 10), base=q)

    reconcile_tracklets(store, threshold=0.63, run_id="r0",
                        min_tracklet_observations=1, color_min_similarity=0.0,
                        log=lambda *a: None)

    gids = _final_gids(store)
    assert gids[("A", 1)] != gids[("B", 1)]


def test_color_veto_blocks_appearance_merge():
    # Same appearance (would merge on embedding alone) but starkly different
    # torso colour -> the CNN-independent veto keeps them apart.
    store = PersonVectorStore(path=":memory:")
    look_alike = _unit(RNG.standard_normal(512))
    _add_tracklet(store, camera="A", track_id=1, gid=1, frames=range(0, 10),
                  base=look_alike, color=_color(0))     # "red"
    _add_tracklet(store, camera="B", track_id=1, gid=2, frames=range(0, 10),
                  base=look_alike, color=_color(20))    # "blue"

    reconcile_tracklets(store, threshold=0.63, run_id="r0",
                        min_tracklet_observations=1, color_min_similarity=0.15,
                        log=lambda *a: None)

    gids = _final_gids(store)
    assert gids[("A", 1)] != gids[("B", 1)]   # colour blocked the merge


def test_color_veto_allows_matching_colors():
    # Same appearance AND same colour -> merge proceeds.
    store = PersonVectorStore(path=":memory:")
    person = _unit(RNG.standard_normal(512))
    _add_tracklet(store, camera="A", track_id=1, gid=1, frames=range(0, 10),
                  base=person, color=_color(5))
    _add_tracklet(store, camera="B", track_id=1, gid=2, frames=range(0, 10),
                  base=person, color=_color(5))

    reconcile_tracklets(store, threshold=0.63, run_id="r0",
                        min_tracklet_observations=1, color_min_similarity=0.15,
                        log=lambda *a: None)

    gids = _final_gids(store)
    assert gids[("A", 1)] == gids[("B", 1)]


def test_short_tracklets_suppressed():
    store = PersonVectorStore(path=":memory:")
    p = _unit(RNG.standard_normal(512))
    q = _unit(RNG.standard_normal(512))
    _add_tracklet(store, camera="A", track_id=1, gid=1, frames=range(0, 10), base=p)
    _add_tracklet(store, camera="B", track_id=1, gid=2, frames=range(0, 10), base=p)
    _add_tracklet(store, camera="A", track_id=9, gid=9, frames=range(50, 51), base=q)  # 1 obs

    reconcile_tracklets(store, threshold=0.63, run_id="r0",
                        min_tracklet_observations=3, color_min_similarity=0.0,
                        log=lambda *a: None)

    gids = _final_gids(store)
    # The 1-observation tracklet is suppressed (its identity cleared -> None).
    assert gids[("A", 9)] is None
    assert gids[("A", 1)] == gids[("B", 1)]


if __name__ == "__main__":
    import traceback

    tests = sorted((n, o) for n, o in globals().items()
                   if n.startswith("test_") and callable(o))
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
