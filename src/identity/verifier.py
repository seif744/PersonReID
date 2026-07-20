"""
verifier.py -- ADR-002 Upgrade 2: verification layer.

Replaces the rigid "cosine >= threshold and gap >= min_score_gap" gate with a
scored combination of signals -- P(same identity) -- so a single noisy cosine
score can't single-handedly decide a merge.

This is a HEURISTIC placeholder: a fixed-weight logistic combination, not a
trained model, because there is no labeled same/different-person dataset for
this deployment yet. The interface -- `Verifier.score(features) -> float` --
is deliberately the same shape a trained classifier would expose
(`model.predict_proba(...)[:, 1]`), and every decision's full feature vector
is logged to `log_path`. Once enough runs accumulate, swap this class's body
for a loaded sklearn/GBT model with NO changes needed at the call site.
"""

import json
import math
import os
import time


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def bbox_quality_scalar(crop_quality):
    """
    Fold the crop_quality dict (produced by reid/service.py's _crop_quality)
    into a single 0-1 goodness score. Missing/unknown quality -> neutral 0.5,
    so an observation we have no quality info for doesn't bias the decision
    either way.
    """
    if not crop_quality:
        return 0.5
    if crop_quality.get("accepted") is False:
        return 0.0
    score = 1.0
    blur = crop_quality.get("blur")
    if blur is not None:
        score *= min(1.0, blur / 60.0)   # saturates comfortably above the min_blur gate
    brightness = crop_quality.get("brightness")
    if brightness is not None:
        mid = 127.5
        score *= max(0.0, 1.0 - abs(brightness - mid) / mid)
    return max(0.0, min(1.0, score))


class Verifier:
    def __init__(self, accept_threshold=0.55, min_prob_gap=0.05, log_path=None):
        self.accept_threshold = float(accept_threshold)
        self.min_prob_gap = float(min_prob_gap)
        self.log_path = log_path
        if self.log_path:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        # Hand-set weights, calibrated so cosine ~0.63 sits near the decision
        # boundary -- measured on this project's own footage with the
        # osnet_x1_0_msmt17 checkpoint: genuinely different people's prototype
        # cosine tops out around 0.48-0.57, while genuine same-person
        # cross-video matches reach 0.70-0.80, a much cleaner gap than the
        # earlier Market1501 checkpoint (which overlapped ~0.70-0.85 for both).
        # Re-measure and re-anchor these weights (or demo_identity.py's sweep)
        # if the ReID weights or camera domain change. Observation count still
        # carries supporting weight: a candidate identity with many accumulated
        # views is stronger corroborating evidence that a borderline cosine
        # score is a real re-appearance, not noise ("track continuity").
        # Cosine/rerank agreement dominates; other signals are supporting
        # nudges, not overrides. PROVISIONAL -- replace with a trained model's
        # coefficients once logged decisions accumulate (see module docstring).
        self.w0 = -13.5             # bias: default toward "not the same person"
        self.w_cosine = 11.0
        self.w_rerank = 7.0
        self.w_observation_count = 0.6
        self.w_time_since_last = -0.01
        self.w_bbox_quality = 1.0

    def score(self, features: dict) -> float:
        cosine = features.get("cosine", 0.0)
        rerank_score = features.get("rerank_score", cosine)
        observation_count = features.get("observation_count", 0)
        time_since_last = features.get("time_since_last_obs", 0)
        bbox_quality = features.get("bbox_quality", 0.5)

        z = (
            self.w0
            + self.w_cosine * cosine
            + self.w_rerank * rerank_score
            + self.w_observation_count * math.log1p(max(0, observation_count))
            + self.w_time_since_last * max(0, time_since_last)
            + self.w_bbox_quality * bbox_quality
        )
        prob = _sigmoid(z)

        if self.log_path:
            self._log(features, prob)
        return prob

    def _log(self, features, prob):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "features": features, "prob": prob}) + "\n")
        except OSError:
            pass
