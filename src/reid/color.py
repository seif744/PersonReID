"""
color.py  --  STAGE 5 (appearance, secondary):  torso colour descriptor.

============================ WHY THIS EXISTS ==============================
The OSNet embedding is the primary appearance signal, but every downstream
identity decision leans on it -- reranking, verifier and reconciliation all
ultimately compare the SAME 512-d vector, so they share its blind spots (pose
flips, domain shift). The handoff (RESEARCH.md) calls for evidence that is
INDEPENDENT of the CNN. A torso colour histogram is exactly that: it is cheap,
CPU-only, and captures "what is this person wearing" in a way orthogonal to the
learned embedding. Two look-alikes the CNN confuses often wear different
colours; a colour mismatch is then a genuine second opinion that can VETO a
false merge.

This module only turns a crop into a colour vector -- a pure measurement, like
the embedding. It assigns no identity and knows nothing about tracks or cameras
(same contract as the ReID extractor; see src/identity/DESIGN.md).

------------------------------- KEY CHOICES -------------------------------
* HSV, hue+saturation only. We drop Value (brightness) so the descriptor is
  robust to lighting/exposure differences BETWEEN cameras -- the whole point of
  a cross-camera cue. Hue+saturation is "which colour, how vivid", not "how
  bright".
* Torso region only. The middle band of the box is dominated by clothing; the
  head (skin/hair, near-constant across people) and legs/feet (often occluded,
  motion-blurred, or cropped at the frame edge) add noise, so we sample a
  central torso window instead of the whole box.
* L1-normalized to a probability histogram, so descriptors are comparable with
  histogram intersection regardless of crop size.
============================================================================
"""

import cv2
import numpy as np

# Default histogram resolution. 8 hue bins x 4 saturation bins = 32-d: enough to
# tell red/blue/green apart and vivid-vs-muted, small enough to stay stable on
# noisy crops (a finer grid just splits the same jacket across many bins).
H_BINS = 8
S_BINS = 4

# Torso window as fractions of the crop (y: top->bottom, x: left->right).
_TORSO_Y = (0.15, 0.55)
_TORSO_X = (0.20, 0.80)


def torso_color_histogram(crop, h_bins: int = H_BINS, s_bins: int = S_BINS):
    """
    BGR crop -> L1-normalized hue-saturation histogram of its torso region, as a
    float32 vector of length h_bins*s_bins. Returns None for a missing/degenerate
    crop or one with no usable colour (all-black), so callers can treat "no colour
    evidence" the same as a missing signal.
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return None

    y0, y1 = int(_TORSO_Y[0] * h), int(_TORSO_Y[1] * h)
    x0, x1 = int(_TORSO_X[0] * w), int(_TORSO_X[1] * w)
    region = crop[y0:y1, x0:x1]
    if region.size == 0:
        region = crop   # tiny box -- fall back to the whole crop rather than fail

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    total = float(hist.sum())
    if total <= 0:
        return None
    return hist / total
