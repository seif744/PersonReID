"""
topology.py  --  STAGE 7c:  learn camera transition-time distributions.

============================ WHY THIS EXISTS ==============================
The camera layout is FIXED between deployments (a hard constraint, see
RESEARCH.md), so how long people take to walk from one camera's field of view to
another's is a stable, learnable property of the site -- "camera 219 exit ->
camera 224 entry takes ~12 s". The handoff asks us to learn these distributions
automatically and use them as EVIDENCE during identity reasoning: a candidate
re-appearance whose timing matches the learned transition is corroborated; one
that would require teleportation is suspect.

This module LEARNS and REPORTS those distributions from the reconciled gallery.
It is deliberately observational: it does not itself merge or split identities.

------------------------- WHY IT DOES NOT GATE (YET) ----------------------
Using transition timing to VETO a merge inside reconciliation is circular within
a single run -- the transitions are learned FROM the merges the reconciler just
made, so feeding them back to change those same merges is unsound without a
second, carefully-designed pass. It also leans on cross-camera frame numbers
being a shared clock (true only when cameras are recorded synchronously at the
same fps). Given the project's "a false merge is far worse than a false split"
bias, we surface this as measured evidence + a plausibility function that a
future, deliberate refinement pass (or an accumulated multi-run prior) can
consume -- rather than silently rewiring the mature, well-tested reconciler on a
signal we cannot yet trust. See ENGINEERING_HANDOFF.md priority 6.

The pure aggregation (`learn_transitions_from_appearances`) and the plausibility
score (`transition_plausibility`) are unit-tested; `learn_transitions` is the
thin store-reading adapter used by main.py.
============================================================================
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TransitionStats:
    """The learned transit-time distribution for one ORDERED camera pair
    (from_camera -> to_camera), in frames."""
    from_camera: str
    to_camera: str
    count: int
    mean: float
    std: float
    min: float
    max: float


# An "appearance" is one identity's presence on one camera: (gid, camera,
# start_frame, end_frame). This is the unit the pure learner consumes.
Appearance = Tuple[int, str, int, int]


def learn_transitions_from_appearances(
        appearances: List[Appearance]) -> Dict[Tuple[str, str], TransitionStats]:
    """
    Learn ordered camera-pair transit-time distributions from identity
    appearances (pure; no store).

    For each identity, its appearances are ordered by start frame; every
    consecutive pair on DIFFERENT cameras yields one transit sample
    (next.start - current.end). Samples are aggregated per ordered (from, to)
    camera pair into a TransitionStats. Negative samples (the two appearances
    overlap in time -- possible with overlapping camera views) are kept: a
    near-zero/negative transit is itself information about camera adjacency.
    """
    by_gid: Dict[int, List[Appearance]] = defaultdict(list)
    for gid, camera, start, end in appearances:
        by_gid[gid].append((gid, camera, int(start), int(end)))

    samples: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for gid, apps in by_gid.items():
        apps = sorted(apps, key=lambda a: a[2])   # by start frame
        for (_, cam_a, _, end_a), (_, cam_b, start_b, _) in zip(apps, apps[1:]):
            if cam_a == cam_b:
                continue
            samples[(cam_a, cam_b)].append(float(start_b - end_a))

    model: Dict[Tuple[str, str], TransitionStats] = {}
    for (cam_a, cam_b), vals in samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        model[(cam_a, cam_b)] = TransitionStats(
            from_camera=cam_a, to_camera=cam_b, count=int(arr.size),
            mean=float(arr.mean()), std=float(arr.std()),
            min=float(arr.min()), max=float(arr.max()),
        )
    return model


def transition_plausibility(model: Dict[Tuple[str, str], TransitionStats],
                            from_camera: str, exit_frame: int,
                            to_camera: str, entry_frame: int,
                            min_std: float = 1.0) -> float:
    """
    How plausible is a from_camera->to_camera transition with this timing, given
    the learned model? Returns a 0-1 score:

      * unknown camera pair (never observed)        -> 0.5  (neutral; no evidence)
      * gap within the learned distribution         -> near 1.0
      * gap far in the tails (near-teleport / too slow) -> near 0.0

    Gaussian on the z-score of the observed gap against the pair's learned
    mean/std (std floored by `min_std` so a single-sample pair doesn't become
    infinitely strict). Same-camera queries are always plausible (1.0).
    """
    if from_camera == to_camera:
        return 1.0
    stats = model.get((from_camera, to_camera))
    if stats is None:
        return 0.5
    gap = float(entry_frame - exit_frame)
    std = max(min_std, stats.std)
    z = (gap - stats.mean) / std
    return float(np.exp(-0.5 * z * z))


def _appearances_from_store(store, run_id=None) -> List[Appearance]:
    """Read reconciled observations and collapse them to per-(gid, camera)
    appearances (min/max frame). One store scan."""
    spans: Dict[Tuple[int, str], List[int]] = {}   # (gid, camera) -> [min, max]
    offset = None
    while True:
        pts, offset = store.client.scroll(
            store.collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if run_id is not None and pl.get("run_id") != run_id:
                continue
            gid = pl.get("reid_id", pl.get("global_id"))
            camera = pl.get("camera")
            frame = pl.get("frame")
            if gid is None or camera is None or frame is None:
                continue
            key = (int(gid), camera)
            frame = int(frame)
            if key not in spans:
                spans[key] = [frame, frame]
            else:
                spans[key][0] = min(spans[key][0], frame)
                spans[key][1] = max(spans[key][1], frame)
        if offset is None:
            break
    return [(gid, camera, lo, hi) for (gid, camera), (lo, hi) in spans.items()]


def learn_transitions(store, run_id=None) -> Dict[Tuple[str, str], TransitionStats]:
    """Store-reading adapter: read the reconciled gallery and learn the ordered
    camera-pair transit-time distributions for this run."""
    return learn_transitions_from_appearances(_appearances_from_store(store, run_id))


def summarize_transitions(model: Dict[Tuple[str, str], TransitionStats],
                          log=print) -> None:
    """Log the learned transition distributions (frames)."""
    if not model:
        log("  camera topology: no cross-camera transitions observed this run.")
        return
    log("  camera topology: learned transition times (frames):")
    for (cam_a, cam_b), s in sorted(model.items()):
        log(f"    {cam_a} -> {cam_b}: mean={s.mean:.1f} std={s.std:.1f} "
            f"n={s.count} [{s.min:.0f}, {s.max:.0f}]")
