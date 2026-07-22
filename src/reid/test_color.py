"""
test_color.py  --  unit tests for the torso colour descriptor (reid/color.py)
and the pure colour-similarity helper (identity/evidence.py:color_similarity).

Pure numpy + cv2, no model. Runs under pytest or standalone:
    python src/reid/test_color.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid.color import torso_color_histogram, H_BINS, S_BINS
from identity.evidence import color_similarity


def _solid(color_bgr, h=200, w=100):
    """A solid-colour BGR crop."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def test_none_and_tiny_crops_return_none():
    assert torso_color_histogram(None) is None
    assert torso_color_histogram(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert torso_color_histogram(np.zeros((2, 2, 3), dtype=np.uint8)) is None


def test_all_black_returns_none():
    # HSV of pure black has zero saturation/value everywhere -> a histogram, but
    # calcHist still counts pixels into the S=0 bin, so it is NOT empty; assert it
    # is a valid normalized vector rather than None.
    hist = torso_color_histogram(_solid((0, 0, 0)))
    assert hist is not None
    assert abs(float(hist.sum()) - 1.0) < 1e-5


def test_histogram_shape_and_normalization():
    hist = torso_color_histogram(_solid((0, 0, 255)))   # red
    assert hist is not None
    assert hist.shape == (H_BINS * S_BINS,)
    assert abs(float(hist.sum()) - 1.0) < 1e-5


def test_same_color_high_similarity():
    red_a = torso_color_histogram(_solid((0, 0, 255)))
    red_b = torso_color_histogram(_solid((0, 0, 255)))
    assert color_similarity(red_a, red_b) > 0.95


def test_different_colors_low_similarity():
    red = torso_color_histogram(_solid((0, 0, 255)))
    blue = torso_color_histogram(_solid((255, 0, 0)))
    assert color_similarity(red, blue) < 0.2


def test_similarity_missing_is_sentinel():
    red = torso_color_histogram(_solid((0, 0, 255)))
    assert color_similarity(red, None) == -1.0
    assert color_similarity(None, None) == -1.0


def test_similarity_shape_mismatch_is_sentinel():
    assert color_similarity(np.ones(4), np.ones(8)) == -1.0


def test_value_channel_ignored_lighting_robustness():
    # Same hue/saturation, different brightness (Value) -> should stay similar
    # because Value is intentionally excluded from the descriptor.
    bright_red = torso_color_histogram(_solid((0, 0, 255)))
    dim_red = torso_color_histogram(_solid((0, 0, 128)))
    assert color_similarity(bright_red, dim_red) > 0.9


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
