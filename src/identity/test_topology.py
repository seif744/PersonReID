"""
test_topology.py  --  unit tests for the camera transition learner (topology.py).

Pure numpy, no store. Runs under pytest or standalone:
    python src/identity/test_topology.py
"""

import numpy as np

from topology import (
    learn_transitions_from_appearances,
    transition_plausibility,
)


def test_empty_appearances_yield_empty_model():
    assert learn_transitions_from_appearances([]) == {}


def test_single_camera_only_no_transitions():
    apps = [(1, "A", 0, 10), (2, "A", 5, 20)]   # nobody changes camera
    assert learn_transitions_from_appearances(apps) == {}


def test_basic_transition_mean():
    # gid 1: A ends 10 -> B starts 22  (gap 12)
    # gid 2: A ends 30 -> B starts 44  (gap 14)
    apps = [(1, "A", 0, 10), (1, "B", 22, 40),
            (2, "A", 15, 30), (2, "B", 44, 60)]
    model = learn_transitions_from_appearances(apps)
    assert ("A", "B") in model
    stats = model[("A", "B")]
    assert stats.count == 2
    assert abs(stats.mean - 13.0) < 1e-9
    assert stats.min == 12.0 and stats.max == 14.0


def test_direction_matters():
    # A->B for gid 1, B->A for gid 2: distinct ordered pairs.
    apps = [(1, "A", 0, 10), (1, "B", 20, 30),
            (2, "B", 0, 10), (2, "A", 25, 35)]
    model = learn_transitions_from_appearances(apps)
    assert ("A", "B") in model
    assert ("B", "A") in model
    assert model[("A", "B")].mean == 10.0
    assert model[("B", "A")].mean == 15.0


def test_appearances_ordered_by_start_frame():
    # Provided out of order; learner must sort by start frame before pairing.
    apps = [(1, "B", 30, 50), (1, "A", 0, 10)]   # true order A then B, gap 20
    model = learn_transitions_from_appearances(apps)
    assert model[("A", "B")].mean == 20.0


def test_plausibility_neutral_for_unknown_pair():
    model = learn_transitions_from_appearances(
        [(1, "A", 0, 10), (1, "B", 20, 30)])
    assert transition_plausibility(model, "A", 10, "C", 40) == 0.5


def test_plausibility_high_near_mean_low_in_tail():
    apps = [(g, "A", 0, 10) for g in range(1, 6)] + \
           [(g, "B", 10 + 12 + g, 40) for g in range(1, 6)]   # gaps ~12-17
    model = learn_transitions_from_appearances(apps)
    near = transition_plausibility(model, "A", 10, "B", 10 + 14)   # ~mean
    far = transition_plausibility(model, "A", 10, "B", 10 + 500)   # way too slow
    assert near > 0.6
    assert far < 0.05


def test_plausibility_same_camera_is_one():
    assert transition_plausibility({}, "A", 0, "A", 100) == 1.0


if __name__ == "__main__":
    import sys
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
