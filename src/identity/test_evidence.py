"""
test_evidence.py  --  unit tests for the Track Evidence Layer (evidence.py).

Pure numpy, no model / no Qdrant, so this runs anywhere in milliseconds. Written
as standard pytest `test_*` functions, but pytest is not currently a project
dependency, so the __main__ block at the bottom also runs them with plain
asserts:  `python src/identity/test_evidence.py`  (or `pytest src/identity`).
"""

import numpy as np

from evidence import (
    EvidenceConfig,
    FrameObservation,
    TrackEvidence,
    build_medoid_prototypes,
    frame_quality_score,
    remove_embedding_outliers,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _rand_unit(rng, dim=512):
    return _unit(rng.standard_normal(dim).astype(np.float32))


def _cluster(rng, base, n, jitter=0.02, dim=512):
    """n unit vectors tightly around `base`."""
    return [_unit(base + jitter * rng.standard_normal(dim).astype(np.float32))
            for _ in range(n)]


def _good_quality(**over):
    q = {"accepted": True, "blur": 120.0, "brightness": 128.0,
         "box_area_ratio": 0.05, "occlusion": 0.0}
    q.update(over)
    return q


def _obs(emb, *, frame, quality=None, cam="A", tid=1, bbox=None,
         frame_shape=None, conf=None, color=None, pid=None):
    return FrameObservation(
        frame_index=frame, camera_id=cam, track_id=tid, run_id="r0",
        embedding=emb, bbox=bbox, frame_shape=frame_shape,
        detector_confidence=conf, crop_quality=quality,
        color_descriptor=color, point_id=pid,
    )


# --------------------------------------------------------------------------- #
# frame_quality_score
# --------------------------------------------------------------------------- #
def test_quality_rejected_is_zero():
    assert frame_quality_score({"accepted": False, "blur": 999}) == 0.0


def test_quality_missing_is_neutral():
    assert frame_quality_score(None) == 0.5


def test_quality_in_unit_range_and_monotonic_in_blur():
    low = frame_quality_score(_good_quality(blur=5.0))
    high = frame_quality_score(_good_quality(blur=200.0))
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high > low


def test_quality_occlusion_drags_score_down():
    clean = frame_quality_score(_good_quality(occlusion=0.0))
    occl = frame_quality_score(_good_quality(occlusion=0.9))
    assert occl < clean


def test_quality_detector_confidence_used():
    lo = frame_quality_score(_good_quality(), detector_confidence=0.1)
    hi = frame_quality_score(_good_quality(), detector_confidence=0.99)
    assert hi > lo


# --------------------------------------------------------------------------- #
# remove_embedding_outliers
# --------------------------------------------------------------------------- #
def test_outlier_removal_skipped_when_too_few():
    rng = np.random.default_rng(0)
    vecs = np.stack(_cluster(rng, _rand_unit(rng), 3))
    keep = remove_embedding_outliers(vecs, EvidenceConfig())
    assert list(keep) == [0, 1, 2]   # below min_obs_for_outlier -> keep all


def test_outlier_removal_drops_injected_outlier():
    rng = np.random.default_rng(1)
    base = _rand_unit(rng)
    inliers = _cluster(rng, base, 9, jitter=0.01)
    outlier = _unit(-base)           # opposite direction -> clear outlier
    vecs = np.stack(inliers + [outlier])
    keep = set(remove_embedding_outliers(vecs, EvidenceConfig()).tolist())
    assert 9 not in keep             # the outlier index
    assert len(keep) == 9


def test_outlier_removal_keeps_tight_cluster_intact():
    rng = np.random.default_rng(2)
    vecs = np.stack(_cluster(rng, _rand_unit(rng), 12, jitter=0.005))
    keep = remove_embedding_outliers(vecs, EvidenceConfig())
    assert len(keep) == 12


# --------------------------------------------------------------------------- #
# build_medoid_prototypes
# --------------------------------------------------------------------------- #
def test_single_view_yields_one_prototype():
    rng = np.random.default_rng(3)
    vecs = np.stack(_cluster(rng, _rand_unit(rng), 10, jitter=0.01))
    protos, sizes, consistency = build_medoid_prototypes(vecs, EvidenceConfig())
    assert protos.shape[0] == 1
    assert sizes == [10]
    assert consistency[0] > 0.95


def test_two_views_yield_two_prototypes_and_are_real_exemplars():
    rng = np.random.default_rng(4)
    front = _rand_unit(rng)
    back = _rand_unit(rng)            # random -> near-orthogonal to front
    front_c = _cluster(rng, front, 6, jitter=0.01)
    back_c = _cluster(rng, back, 6, jitter=0.01)
    vecs = np.stack(front_c + back_c)
    protos, sizes, _ = build_medoid_prototypes(vecs, EvidenceConfig())
    assert protos.shape[0] == 2
    assert sorted(sizes) == [6, 6]
    # Each prototype is an ACTUAL observed vector (a medoid), not an average.
    matches = (np.abs(vecs @ protos.T - 1.0) < 1e-4).any(axis=0)
    assert matches.all()


def test_prototypes_respect_max_cap():
    rng = np.random.default_rng(5)
    clusters = []
    for _ in range(6):               # 6 distinct views...
        clusters += _cluster(rng, _rand_unit(rng), 4, jitter=0.01)
    vecs = np.stack(clusters)
    cfg = EvidenceConfig(max_prototypes=3)
    protos, sizes, _ = build_medoid_prototypes(vecs, cfg)
    assert protos.shape[0] <= 3      # ...collapse to at most the cap
    assert sum(sizes) == len(vecs)   # every vector still assigned somewhere


def test_prototypes_ordered_most_populated_first():
    rng = np.random.default_rng(6)
    big = _cluster(rng, _rand_unit(rng), 9, jitter=0.01)
    small = _cluster(rng, _rand_unit(rng), 3, jitter=0.01)
    vecs = np.stack(small + big)     # small first in input order
    _, sizes, _ = build_medoid_prototypes(vecs, EvidenceConfig())
    assert sizes == sorted(sizes, reverse=True)


# --------------------------------------------------------------------------- #
# TrackEvidence.from_observations
# --------------------------------------------------------------------------- #
def test_from_observations_none_on_empty():
    assert TrackEvidence.from_observations([]) is None


def test_from_observations_basic_stats():
    rng = np.random.default_rng(7)
    base = _rand_unit(rng)
    obs = [_obs(v, frame=i, quality=_good_quality(), pid=f"p{i}")
           for i, v in enumerate(_cluster(rng, base, 8, jitter=0.01))]
    ev = TrackEvidence.from_observations(obs)
    assert ev is not None
    assert ev.key == ("A", 1)
    assert ev.raw_observation_count == 8
    assert ev.observation_count == 8
    assert ev.usable is True
    assert ev.prototypes.shape[0] == 1
    assert ev.global_centroid is not None
    assert ev.embedding_variance < 0.05          # tight cluster
    assert 0.0 <= ev.track_confidence <= 1.0
    assert ev.track_confidence > 0.0
    assert ev.start_frame == 0 and ev.end_frame == 7
    assert ev.frames == frozenset(range(8))
    assert set(ev.kept_point_ids) == {f"p{i}" for i in range(8)}


def test_from_observations_frames_include_quality_rejected():
    # A quality-rejected frame is still PHYSICAL presence -> stays in `frames`
    # (co-presence must not depend on crop quality) but not in kept evidence.
    rng = np.random.default_rng(8)
    base = _rand_unit(rng)
    good = [_obs(v, frame=i, quality=_good_quality())
            for i, v in enumerate(_cluster(rng, base, 6, jitter=0.01))]
    bad = _obs(_rand_unit(rng), frame=99, quality={"accepted": False})
    ev = TrackEvidence.from_observations(good + [bad])
    assert 99 in ev.frames                      # physical presence kept
    assert ev.end_frame == 99
    assert ev.raw_observation_count == 7
    assert ev.observation_count == 6            # the rejected frame is not evidence


def test_from_observations_thin_track_flagged_unusable():
    rng = np.random.default_rng(9)
    base = _rand_unit(rng)
    obs = [_obs(v, frame=i, quality=_good_quality())
           for i, v in enumerate(_cluster(rng, base, 2, jitter=0.01))]
    ev = TrackEvidence.from_observations(obs, EvidenceConfig(min_observations=3))
    assert ev.observation_count == 2
    assert ev.usable is False                   # below min_observations


def test_from_observations_all_bad_quality_keeps_best_not_empty():
    # Even if everything is mediocre, we keep the best few rather than erase the
    # track (never silently delete a whole track).
    rng = np.random.default_rng(10)
    base = _rand_unit(rng)
    obs = [_obs(v, frame=i, quality=_good_quality(blur=6.0, box_area_ratio=0.001))
           for i, v in enumerate(_cluster(rng, base, 5, jitter=0.01))]
    ev = TrackEvidence.from_observations(obs, EvidenceConfig(min_quality=0.99))
    assert ev.observation_count >= 3            # min_keep floor respected
    assert ev.prototypes.shape[0] >= 1


def test_from_observations_no_embeddings_returns_shell():
    obs = [_obs(None, frame=i, quality=_good_quality()) for i in range(4)]
    ev = TrackEvidence.from_observations(obs)
    assert ev is not None
    assert ev.observation_count == 0
    assert ev.prototypes.size == 0
    assert ev.global_centroid is None
    assert ev.usable is False
    assert ev.frames == frozenset(range(4))     # presence still recorded


def test_best_similarity_matches_any_prototype():
    rng = np.random.default_rng(11)
    front = _rand_unit(rng)
    back = _rand_unit(rng)
    obs = [_obs(v, frame=i, quality=_good_quality())
           for i, v in enumerate(_cluster(rng, front, 6, jitter=0.01)
                                 + _cluster(rng, back, 6, jitter=0.01))]
    ev = TrackEvidence.from_observations(obs)
    # A query near the back view should score high against the back prototype,
    # even though the front cluster is (tied) the most-populated.
    assert ev.best_similarity(back) > 0.9
    assert ev.best_similarity(front) > 0.9
    assert ev.best_similarity(_rand_unit(rng)) < 0.5


def test_normalized_height_from_boxes():
    rng = np.random.default_rng(12)
    base = _rand_unit(rng)
    obs = [_obs(v, frame=i, quality=_good_quality(),
                bbox=(0, 100, 50, 300), frame_shape=(800, 600))
           for i, v in enumerate(_cluster(rng, base, 5, jitter=0.01))]
    ev = TrackEvidence.from_observations(obs)
    assert abs(ev.normalized_height - (200.0 / 800.0)) < 1e-6


def test_color_descriptor_quality_weighted_and_normalized():
    rng = np.random.default_rng(13)
    base = _rand_unit(rng)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    obs = [
        _obs(_unit(base), frame=0, quality=_good_quality(), color=red),
        _obs(_unit(base), frame=1, quality=_good_quality(), color=red),
        _obs(_unit(base), frame=2, quality=_good_quality(blur=6.0), color=blue),
    ]
    ev = TrackEvidence.from_observations(obs)
    assert ev.color_descriptor is not None
    assert abs(float(ev.color_descriptor.sum()) - 1.0) < 1e-5   # histogram
    # Red dominates: two high-quality reds vs one low-quality blue.
    assert ev.color_descriptor[0] > ev.color_descriptor[2]


def test_confidence_rises_with_more_and_better_observations():
    rng = np.random.default_rng(14)
    base = _rand_unit(rng)
    few = [_obs(v, frame=i, quality=_good_quality())
           for i, v in enumerate(_cluster(rng, base, 3, jitter=0.01))]
    many = [_obs(v, frame=i, quality=_good_quality())
            for i, v in enumerate(_cluster(rng, base, 8, jitter=0.01))]
    c_few = TrackEvidence.from_observations(few).track_confidence
    c_many = TrackEvidence.from_observations(many).track_confidence
    assert c_many > c_few


# --------------------------------------------------------------------------- #
# standalone runner (works without pytest installed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import traceback

    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
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
