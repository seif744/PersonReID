"""
evidence.py  --  STAGE 7 (evidence):  the Track Evidence Layer.

============================ WHAT THIS IS =================================
This module turns a track's NOISY, per-frame observations into ONE robust,
track-level summary -- a `TrackEvidence` object -- BEFORE any identity decision
is made. It is the "extract & enrich" of the abstraction that already lived,
un-named, inside reconcile.py's `_gather_tracklets` (group by (camera, track_id),
build a prototype, count observations, remember frames/spans). Here it becomes a
first-class thing the reconciler (and, later, the gallery) can consume.

It is deliberately PURE: it takes plain data (embeddings, bounding boxes, the
crop-quality dicts TrackEmbedder already produces, optional colour descriptors)
and returns plain data. It touches NO model, NO Qdrant, NO cameras/clock policy.
That is why it is unit-testable with nothing but numpy, and why it does not
belong in the appearance layer (which must not know about tracks) nor in the
store (which must not own logic). See src/identity/DESIGN.md for the boundaries.

------------------------------ WHAT IT ADDS -------------------------------
Over the old single-mean prototype, TrackEvidence adds the four things the
handoff (RESEARCH.md / ENGINEERING_HANDOFF.md) asks the Track Evidence Layer for:

  1. FRAME QUALITY as a *score*, not just the accept/reject gate TrackEmbedder
     already applies. Quality here down-WEIGHTS observations rather than only
     dropping them, and decides which ones become the prototype.
  2. EMBEDDING OUTLIER removal (robust, MAD-based) so one bad back-view or a
     half-occluded frame can't drag the track's centre off the person.
  3. MULTI-PROTOTYPE MEDOIDS instead of one blurred mean. A person seen front
     AND back produces a mean vector less similar to either real view than the
     views are to each other; keeping a small set of representative *real*
     exemplars (medoids) matches every view instead of none well.
  4. TRACK STATISTICS + UNCERTAINTY (embedding variance, mean quality,
     normalized height, an optional colour descriptor, a bounded confidence)
     so downstream reasoning has evidence INDEPENDENT of raw cosine.

Nothing here decides identity. It produces measurements a decider can fuse.

------------------------------- KEY CHOICES -------------------------------
* "Prefer a split over a merge" carries through: filtering is conservative and
  never silently deletes a whole track -- if quality filtering would empty a
  track, we keep its best few observations and let the caller decide (via the
  `usable`/`observation_count` fields) whether the track is too thin to trust.
* `frames` reflects PHYSICAL presence, so it is built from *every* observation
  (even quality-rejected ones): co-presence in a camera is a physical fact used
  to forbid merges, independent of how good the crop was.
* Everything is L2-normalized internally; cosine == dot product throughout,
  matching the rest of the pipeline (store.py uses COSINE distance).
============================================================================
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class EvidenceConfig:
    """
    Tunables for building TrackEvidence. Defaults are anchored to this project's
    osnet_x1_0_msmt17 checkpoint (same-person prototype cosine ~0.70-0.80,
    different-person ~0.48-0.57; see RESEARCH.md / verifier.py) and to the
    crop-quality gate already configured in config.yaml. They are intentionally
    forgiving: the layer's job is to CLEAN evidence, not to be a second identity
    gate -- that decision stays in the reconciler / verifier.
    """

    # --- quality filtering ---------------------------------------------------
    # Drop observations whose 0-1 quality score is below this. Low, because
    # TrackEmbedder already applied a hard accept/reject gate upstream; this is a
    # second, softer pass that removes the merely-mediocre when better views
    # exist for the same track.
    min_quality: float = 0.15
    # Never let quality filtering drop a track below this many of its best
    # observations, nor below this fraction of them. A track with only poor crops
    # is still a track -- we keep its best evidence and flag it thin rather than
    # erase it (see module docstring).
    min_keep: int = 3
    min_keep_fraction: float = 0.34

    # --- embedding outlier removal (MAD on cosine-to-centre) -----------------
    outlier_mad_k: float = 3.0            # keep obs within k robust-sigmas of the median
    min_obs_for_outlier: int = 5          # too few obs -> no robust centre, skip removal

    # --- multi-prototype medoids ---------------------------------------------
    # Two observations more similar than this join the same prototype; below it,
    # a new prototype (a distinct view: front/back/near/far) is opened, up to the
    # cap. 0.75 sits between msmt17 same-view (~0.8+) and cross-view same-person
    # (~0.7), so it separates genuine viewpoints without shattering one view into
    # many.
    prototype_merge_threshold: float = 0.75
    max_prototypes: int = 4

    # --- statistics / confidence ---------------------------------------------
    # Observation count at which the "coverage" part of confidence saturates.
    target_observations: int = 8
    # A track with fewer kept observations than this is treated as too thin to be
    # a trustworthy identity on its own (mirrors reconcile.min_tracklet_observations).
    min_observations: int = 3


# =============================================================================
# One frame's worth of evidence
# =============================================================================
@dataclass
class FrameObservation:
    """
    One accepted observation of one track in one frame -- the raw input to the
    Track Evidence Layer. Mirrors the fields already flowing through the live
    pipeline (a Detection's embedding + crop_quality, its box, the frame index),
    plus optional independent-evidence channels (colour, detector confidence)
    that cost nothing to carry and everything to have later.

    Only `frame_index` is strictly required. `embedding` is needed for any
    appearance statistic; observations without one still count toward physical
    presence (`frames`) but contribute nothing to prototypes.
    """

    frame_index: int
    camera_id: Optional[str] = None
    track_id: Optional[int] = None
    run_id: Optional[str] = None

    embedding: Optional[np.ndarray] = None
    bbox: Optional[Tuple[float, float, float, float]] = None   # (x1, y1, x2, y2)
    frame_shape: Optional[Tuple[int, int]] = None              # (height, width)
    detector_confidence: Optional[float] = None
    crop_quality: Optional[dict] = None
    color_descriptor: Optional[np.ndarray] = None
    point_id: Optional[str] = None
    # A precomputed box-height / frame-height, when the caller already has it
    # (e.g. TrackEmbedder computed it from the live frame and persisted it).
    # Takes precedence over deriving it from bbox + frame_shape.
    normalized_height: Optional[float] = None


# =============================================================================
# One track's worth of evidence
# =============================================================================
@dataclass
class TrackEvidence:
    """
    A robust, track-level summary produced from many FrameObservations. This is
    the object identity reasoning should consume instead of individual frame
    embeddings.

    Appearance is exposed as BOTH a small set of representative medoid
    `prototypes` (compare a query against every one, take the best -- matches any
    viewpoint) AND a coarse `global_centroid` (a single cheap descriptor, kept as
    an auxiliary per the handoff). The remaining fields are independent evidence
    and uncertainty for a fusion-based decider.
    """

    camera_id: Optional[str]
    track_id: Optional[int]
    run_id: Optional[str]

    start_frame: int
    end_frame: int
    frames: frozenset                       # every observed frame (physical presence)

    observation_count: int                  # observations kept after cleaning
    raw_observation_count: int              # observations seen before cleaning

    prototypes: np.ndarray                  # (P, D) L2-normalized medoids, most-populated first
    prototype_sizes: Tuple[int, ...]        # how many kept obs each prototype represents
    global_centroid: Optional[np.ndarray]   # (D,) L2-normalized mean of kept embeddings

    embedding_variance: float               # mean (1 - cosine to centroid) over kept obs; lower = tighter
    mean_quality: float                     # mean 0-1 quality of kept obs
    track_confidence: float                 # bounded 0-1 uncertainty summary
    viewpoint_diversity: int                # number of distinct prototypes (a viewpoint proxy)

    normalized_height: Optional[float]      # median bbox_height / frame_height, if boxes known
    color_descriptor: Optional[np.ndarray]  # quality-weighted, L1-normalized colour aggregate, if provided

    kept_point_ids: Tuple[str, ...]         # store point-ids of the kept observations
    all_point_ids: Tuple[str, ...]          # store point-ids of every observation

    usable: bool                            # observation_count >= config.min_observations

    @property
    def key(self) -> Tuple[Optional[str], Optional[int]]:
        """The (camera, track) tracklet key -- the natural identity of a track."""
        return (self.camera_id, self.track_id)

    def best_similarity(self, embedding) -> float:
        """
        Best cosine of `embedding` against ANY of this track's prototypes -- the
        multi-prototype replacement for "cosine vs the single mean". Returns -1.0
        when there is no appearance evidence or the query is degenerate.
        """
        if self.prototypes.size == 0:
            return -1.0
        q = _unit(np.asarray(embedding, dtype=np.float32).ravel())
        if q is None:
            return -1.0
        return float(np.max(self.prototypes @ q))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_observations(
        cls,
        observations: Sequence[FrameObservation],
        config: Optional[EvidenceConfig] = None,
    ) -> Optional["TrackEvidence"]:
        """
        Build one TrackEvidence from all observations of a single track.

        Pipeline: score quality -> drop low quality (never below a floor) ->
        remove embedding outliers -> build medoid prototypes + centroid ->
        compute statistics + confidence.

        Returns None only when given no observations at all. A track with
        observations but no usable embeddings still yields evidence (with empty
        prototypes) so the caller can see it exists.
        """
        cfg = config or EvidenceConfig()
        obs = list(observations)
        if not obs:
            return None

        camera_id = _first(o.camera_id for o in obs)
        track_id = _first(o.track_id for o in obs)
        run_id = _first(o.run_id for o in obs)

        # Physical presence uses EVERY observation, quality aside.
        all_frames = [int(o.frame_index) for o in obs]
        frames = frozenset(all_frames)
        all_point_ids = tuple(o.point_id for o in obs if o.point_id is not None)

        # --- 1. quality scoring + soft filtering -----------------------------
        scored = [(frame_quality_score(o.crop_quality, o.detector_confidence), o)
                  for o in obs]
        kept = _quality_filter(scored, cfg)

        # --- 2. keep only kept-obs that actually carry an embedding ----------
        with_emb = [(q, o, v) for (q, o) in kept
                    for v in (_unit_or_none(o.embedding),) if v is not None]

        if not with_emb:
            # No appearance evidence -- still return a shell so the track is visible.
            return cls(
                camera_id=camera_id, track_id=track_id, run_id=run_id,
                start_frame=min(all_frames), end_frame=max(all_frames), frames=frames,
                observation_count=0, raw_observation_count=len(obs),
                prototypes=np.empty((0, 0), dtype=np.float32), prototype_sizes=(),
                global_centroid=None, embedding_variance=0.0,
                mean_quality=float(np.mean([q for q, _ in scored])) if scored else 0.0,
                track_confidence=0.0, viewpoint_diversity=0,
                normalized_height=_median_normalized_height(obs),
                color_descriptor=None,
                kept_point_ids=(), all_point_ids=all_point_ids, usable=False,
            )

        quals = np.array([q for q, _, _ in with_emb], dtype=np.float32)
        vecs = np.stack([v for _, _, v in with_emb]).astype(np.float32)
        kobs = [o for _, o, _ in with_emb]

        # --- 3. embedding outlier removal ------------------------------------
        keep_idx = remove_embedding_outliers(vecs, cfg)
        vecs = vecs[keep_idx]
        quals = quals[keep_idx]
        kobs = [kobs[i] for i in keep_idx]

        # --- 4. prototypes + centroid ----------------------------------------
        protos, sizes, consistency = build_medoid_prototypes(vecs, cfg)
        centroid = _unit(vecs.mean(axis=0))

        # --- 5. statistics ----------------------------------------------------
        if centroid is not None:
            embedding_variance = float(np.mean(1.0 - (vecs @ centroid)))
        else:
            embedding_variance = 0.0
        mean_quality = float(np.mean(quals)) if quals.size else 0.0
        normalized_height = _median_normalized_height(kobs)
        color_descriptor = _aggregate_color(kobs, quals)

        confidence = _track_confidence(
            n_kept=len(kobs), mean_quality=mean_quality,
            consistency=consistency, sizes=sizes, cfg=cfg)

        kept_point_ids = tuple(o.point_id for o in kobs if o.point_id is not None)

        return cls(
            camera_id=camera_id, track_id=track_id, run_id=run_id,
            start_frame=min(all_frames), end_frame=max(all_frames), frames=frames,
            observation_count=len(kobs), raw_observation_count=len(obs),
            prototypes=protos, prototype_sizes=tuple(sizes),
            global_centroid=centroid, embedding_variance=embedding_variance,
            mean_quality=mean_quality, track_confidence=confidence,
            viewpoint_diversity=len(sizes),
            normalized_height=normalized_height, color_descriptor=color_descriptor,
            kept_point_ids=kept_point_ids, all_point_ids=all_point_ids,
            usable=len(kobs) >= cfg.min_observations,
        )


# =============================================================================
# Quality scoring
# =============================================================================
def frame_quality_score(crop_quality: Optional[dict],
                        detector_confidence: Optional[float] = None) -> float:
    """
    Fold TrackEmbedder's crop_quality dict (+ optional detector confidence) into
    one 0-1 goodness score. This is the RICHER cousin of verifier.bbox_quality_scalar:
    it uses size and occlusion too, and combines factors with a weighted GEOMETRIC
    mean so a single very bad factor (say a heavily occluded crop) drags the whole
    score down -- the arithmetic mean would let a big, bright, sharp-but-occluded
    crop still look "good".

    A hard reject upstream (`accepted is False`) is 0. Missing quality info is a
    neutral 0.5, so an observation we know nothing about neither helps nor hurts.
    """
    if crop_quality is None:
        base = 0.5
    elif crop_quality.get("accepted") is False:
        return 0.0
    else:
        base = None  # computed from factors below

    factors: List[Tuple[float, float]] = []  # (value in [0,1], weight)

    if crop_quality is not None:
        blur = crop_quality.get("blur")
        if blur is not None:
            # Laplacian variance; saturates comfortably above the min_blur (~20) gate.
            factors.append((_clip01(blur / 60.0), 0.25))

        brightness = crop_quality.get("brightness")
        if brightness is not None:
            mid = 127.5
            factors.append((_clip01(1.0 - abs(brightness - mid) / mid), 0.10))

        ratio = crop_quality.get("box_area_ratio")
        if ratio is not None:
            # Fraction of the frame the person occupies; saturates at ~2%.
            factors.append((_clip01(ratio / 0.02), 0.25))

        occlusion = crop_quality.get("occlusion")
        if occlusion is not None:
            factors.append((_clip01(1.0 - occlusion), 0.25))

    if detector_confidence is not None:
        factors.append((_clip01(detector_confidence), 0.15))

    if not factors:
        return base if base is not None else 0.5
    return _weighted_geomean(factors)


# =============================================================================
# Embedding outlier removal
# =============================================================================
def remove_embedding_outliers(vecs: np.ndarray,
                              config: Optional[EvidenceConfig] = None) -> np.ndarray:
    """
    Return the indices of `vecs` (assumed L2-normalized rows) to KEEP after
    dropping appearance outliers -- a back-view frame, a half-occluded crop, a
    frame where the box briefly grabbed a second person.

    Method: robust distance-to-centre. Take the cosine of each vector to the
    (renormalized) mean direction, then keep everything within `outlier_mad_k`
    median-absolute-deviations of the MEDIAN similarity. MAD (not std) so the
    outliers we are hunting don't inflate the very spread used to detect them.
    Below `min_obs_for_outlier` there isn't a stable centre to trust, so nothing
    is dropped. We also never drop below `min_keep` vectors.
    """
    cfg = config or EvidenceConfig()
    n = len(vecs)
    if n < max(2, cfg.min_obs_for_outlier):
        return np.arange(n)

    centre = _unit(vecs.mean(axis=0))
    if centre is None:
        return np.arange(n)
    sims = vecs @ centre
    med = float(np.median(sims))
    mad = float(np.median(np.abs(sims - med)))
    if mad < 1e-6:
        return np.arange(n)   # a tight cluster; nothing stands out

    threshold = med - cfg.outlier_mad_k * 1.4826 * mad   # 1.4826: MAD -> sigma for normal
    keep_mask = sims >= threshold
    if int(keep_mask.sum()) >= max(cfg.min_keep, 1):
        return np.nonzero(keep_mask)[0]
    # Would drop too many -> keep the best min_keep instead.
    order = np.argsort(-sims)
    return np.sort(order[:max(cfg.min_keep, 1)])


# =============================================================================
# Multi-prototype medoids
# =============================================================================
def build_medoid_prototypes(vecs: np.ndarray,
                            config: Optional[EvidenceConfig] = None
                            ) -> Tuple[np.ndarray, List[int], List[float]]:
    """
    Reduce `vecs` (L2-normalized rows) to a small set of representative MEDOIDS --
    actual observed vectors, not averages -- one per distinct viewpoint cluster.

    Greedy, deterministic, cheap (tracks hold tens of vectors, not millions):
      1. Order vectors by centrality (row-sum of similarity): the most typical
         view seeds the first cluster.
      2. Walk them; a vector joins the most-similar existing cluster if that
         similarity clears `prototype_merge_threshold`, else it opens a new
         cluster -- until `max_prototypes`, after which everything joins its
         nearest cluster (no runaway shattering).
      3. Each cluster's medoid is the member most similar, on average, to the
         rest of its cluster.

    Returns (prototypes (P, D) ordered most-populated first, sizes, per-cluster
    consistency = mean similarity of members to their medoid). Empty input ->
    empty (0, 0) array.
    """
    cfg = config or EvidenceConfig()
    n = len(vecs)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32), [], []
    if n == 1:
        return vecs.copy(), [1], [1.0]

    sims = vecs @ vecs.T
    centrality = sims.sum(axis=1)
    order = np.argsort(-centrality)

    seeds: List[int] = []          # medoid-seed index per cluster
    assign = np.full(n, -1, dtype=int)
    for i in order:
        if not seeds:
            seeds.append(i)
            assign[i] = 0
            continue
        seed_sims = [sims[i, s] for s in seeds]
        best = int(np.argmax(seed_sims))
        if seed_sims[best] >= cfg.prototype_merge_threshold or len(seeds) >= cfg.max_prototypes:
            assign[i] = best
        else:
            assign[i] = len(seeds)
            seeds.append(i)

    protos: List[np.ndarray] = []
    sizes: List[int] = []
    consistency: List[float] = []
    for c in range(len(seeds)):
        idx = np.nonzero(assign == c)[0]
        sub = sims[np.ix_(idx, idx)]
        medoid_local = idx[int(np.argmax(sub.sum(axis=1)))]
        protos.append(vecs[medoid_local])
        sizes.append(int(len(idx)))
        consistency.append(float(np.mean(sims[medoid_local, idx])))

    # Most-populated prototype first (deterministic; ties broken by consistency).
    order_out = sorted(range(len(sizes)),
                       key=lambda c: (sizes[c], consistency[c]), reverse=True)
    protos = [protos[c] for c in order_out]
    sizes = [sizes[c] for c in order_out]
    consistency = [consistency[c] for c in order_out]
    return np.stack(protos).astype(np.float32), sizes, consistency


# =============================================================================
# Confidence / uncertainty
# =============================================================================
def _track_confidence(*, n_kept: int, mean_quality: float,
                     consistency: List[float], sizes: List[int],
                     cfg: EvidenceConfig) -> float:
    """
    Bounded 0-1 summary of how much to trust this track's evidence, combining
    three independent factors (multiplied, so any one being weak lowers trust):

      * coverage  -- more observations, up to target_observations, = more evidence.
      * quality   -- the mean crop quality of what we kept.
      * tightness -- the size-weighted mean intra-cluster consistency: a track
                     whose views cohere is more trustworthy than a scattered one.

    This is a heuristic, on purpose -- the same shape a learned model would take
    (see verifier.py). It is EVIDENCE, not a decision.
    """
    if n_kept <= 0:
        return 0.0
    coverage = min(1.0, n_kept / max(1, cfg.target_observations))
    quality = _clip01(mean_quality)
    if sizes and sum(sizes) > 0:
        tightness = float(np.average(consistency, weights=sizes))
    else:
        tightness = 0.0
    tightness = _clip01(tightness)
    return _clip01(coverage * quality * tightness)


# =============================================================================
# Small helpers
# =============================================================================
def _unit(vec: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalize a vector; None if it is zero/degenerate."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n <= 0 or not np.isfinite(n):
        return None
    return v / n


def _unit_or_none(vec) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).ravel()
    if arr.size == 0:
        return None
    return _unit(arr)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _weighted_geomean(factors: List[Tuple[float, float]]) -> float:
    """
    Weighted geometric mean of (value, weight) pairs, values in [0,1]. Values are
    floored at a tiny epsilon so a single 0 doesn't annihilate the whole score to
    0 (which would make every zero-factor observation indistinguishable); it still
    drives the result sharply toward 0, which is the intent.
    """
    eps = 1e-3
    total_w = sum(w for _, w in factors)
    if total_w <= 0:
        return 0.0
    acc = sum(w * np.log(max(eps, v)) for v, w in factors)
    return _clip01(float(np.exp(acc / total_w)))


def color_similarity(a, b) -> float:
    """
    Histogram intersection of two L1-normalized colour descriptors, in [0, 1]:
    1.0 = identical colour distribution, 0.0 = no shared colour. Returns -1.0 if
    either is missing or their lengths disagree, so "no colour evidence" is
    distinguishable from "colours differ" and never silently blocks a decision.
    """
    if a is None or b is None:
        return -1.0
    va = np.asarray(a, dtype=np.float32).ravel()
    vb = np.asarray(b, dtype=np.float32).ravel()
    if va.shape != vb.shape or va.size == 0:
        return -1.0
    return float(np.minimum(va, vb).sum())


def _first(values):
    """First non-None value from an iterable, else None."""
    for v in values:
        if v is not None:
            return v
    return None


def _quality_filter(scored: List[Tuple[float, "FrameObservation"]],
                   cfg: EvidenceConfig) -> List[Tuple[float, "FrameObservation"]]:
    """
    Keep observations scoring at least `min_quality`, but NEVER drop a track
    below a floor of its best observations -- `min_keep` of them, or
    `min_keep_fraction` of the total, whichever is larger. A track made entirely
    of mediocre crops is still a track; we keep its best evidence and let the
    caller judge it thin (via observation_count / usable) rather than erase it.
    Returned best-quality first.
    """
    ordered = sorted(scored, key=lambda qo: qo[0], reverse=True)
    n = len(ordered)
    floor = max(cfg.min_keep, int(np.ceil(cfg.min_keep_fraction * n)))
    floor = min(floor, n)
    passing = [qo for qo in ordered if qo[0] >= cfg.min_quality]
    if len(passing) >= floor:
        return passing
    return ordered[:floor]


def _median_normalized_height(obs: Sequence[FrameObservation]) -> Optional[float]:
    """
    Median normalized height over observations. Prefers a precomputed
    `normalized_height` (persisted by TrackEmbedder); otherwise derives it from
    bbox + frame_shape when both are present.
    """
    heights = []
    for o in obs:
        if o.normalized_height is not None:
            heights.append(float(o.normalized_height))
        elif o.bbox is not None and o.frame_shape is not None:
            _, y1, _, y2 = o.bbox
            frame_h = o.frame_shape[0]
            if frame_h and frame_h > 0:
                heights.append((y2 - y1) / float(frame_h))
    if not heights:
        return None
    return float(np.median(heights))


def _aggregate_color(obs: Sequence[FrameObservation],
                    quals: np.ndarray) -> Optional[np.ndarray]:
    """
    Quality-weighted mean of per-observation colour descriptors, L1-normalized
    back into a histogram. Colour is an appearance signal INDEPENDENT of the
    OSNet embedding (it survives when the CNN is fooled by pose), which is the
    whole point of carrying it. None if no observation supplied a descriptor.
    """
    vecs = []
    weights = []
    for o, q in zip(obs, quals):
        if o.color_descriptor is None:
            continue
        c = np.asarray(o.color_descriptor, dtype=np.float32).ravel()
        if c.size == 0:
            continue
        vecs.append(c)
        weights.append(max(1e-3, float(q)))
    if not vecs:
        return None
    lengths = {v.shape[0] for v in vecs}
    if len(lengths) != 1:
        return None   # inconsistent descriptor sizes -- refuse to guess
    mat = np.stack(vecs)
    w = np.asarray(weights, dtype=np.float32)
    agg = (mat * w[:, None]).sum(axis=0)
    total = float(agg.sum())
    if total <= 0:
        return None
    return (agg / total).astype(np.float32)
