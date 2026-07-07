# Identity Service — Design (Stage 7)


> This document specifies *what* the Identity Service is responsible for and
> *why* the responsibility lines are drawn where they are. It is written to be
> read by an engineer who will implement it later, and to stop identity logic
> from leaking into the ReID model or the vector database.

---

## 0. Where this sits

```
YOLO11 → ByteTrack → ReIDExtractor → TrackEmbedder → [ Identity Service ] → Qdrant
                     (crop→512-d)     (attach to      (assigns global_id)   (ANN index)
                                       tracks)
```

Upstream (already built) gives us, per frame, a set of **tracked detections**,
each with:
- a **per-camera** `track_id` (stable while the person is visible on that camera),
- a 512-d **L2-normalized appearance embedding**,
- a bounding box, a camera id, and a timestamp.

The Identity Service turns those into a **`global_id`**: one stable number per
real human, consistent **across cameras and across time** (including after a
person leaves one camera and reappears on another minutes later).

---

## 1. Why the ReID model must NEVER assign identities

The extractor's contract is `image → embedding`. It is **stateless and
memoryless** by design, and that is exactly why it must not decide identity:

1. **It sees one crop at a time.** Identity is a decision about a crop *relative
   to everyone seen so far and where/when they were seen*. The model has none of
   that context — no gallery, no camera map, no clock. Any identity it "assigned"
   would be a guess from a single image.

2. **Appearance alone is ambiguous — we measured it.** In Phase 4 the
   same-person and different-person cosine distributions **overlap** (same-min
   0.51 < diff-max 0.85). No fixed threshold on the embedding separates people
   cleanly. Identity must be resolved with *more* signal than appearance; the
   model only produces the appearance signal.

3. **Separation of concerns / testability.** The model is a pure function: same
   input → same output, trivially unit-testable and swappable (Market-1501 dev
   weights today, RandPerson weights tomorrow) with **zero** change to identity
   logic. Fuse the two and every model swap risks the identity behavior, and
   neither can be tested in isolation.

4. **It has no write authority.** Assigning an id is a *stateful mutation* of a
   shared global namespace. A stateless feature extractor running in parallel
   across hundreds of camera workers cannot own a consistent global counter.

**Rule:** the model outputs a *measurement*. It never outputs a *decision*.

---

## 2. Why Qdrant (the vector DB) must NEVER assign identities

Qdrant is an **approximate nearest-neighbor index**. It answers exactly one
question: *"which stored vectors are closest to this query vector?"* It must not
decide identity, for reasons symmetric to the model:

1. **It only knows vectors, not the world.** Qdrant returns "vector X is at
   cosine 0.83." It does not know that camera A and camera B don't overlap, that
   the two crops are 4 ms apart (physically impossible for one person to be in
   both), or that the candidate left the building an hour ago. Turning a distance
   into an identity requires those constraints — Qdrant has none of them.

2. **Nearest neighbor ≠ same person.** Because of the distribution overlap, the
   top hit can be a look-alike (similar clothing) while the true match sits at
   rank 3. A "the closest vector *is* the identity" rule guarantees identity
   swaps on crowded scenes. ANN search is a *candidate generator*, not a decider.

3. **Thresholds are context-dependent.** The right cosine cutoff differs by
   camera pair (lighting, viewpoint), by crowd density, and by time gap. Baking a
   decision threshold into the query makes it un-tunable and invisible. The
   decision (and its policy) must live in one auditable place: the service.

4. **Writes must be governed.** Which embeddings get stored, under which
   `global_id`, with what metadata and TTL, is a policy decision (a gallery /
   feature-bank strategy). That governance belongs to the service that owns the
   identity namespace, not to the storage layer.

**Rule:** Qdrant *retrieves candidates*. The Identity Service *decides*.

---

## 3. What the Identity Service does

It is the **only** component with write authority over `global_id`. For each
tracked detection (in practice, per *track* — one decision per track, not per
frame, reusing the track's embedding) it runs a **candidate → score → decide →
commit** loop:

### 3.1 Candidate generation (Qdrant)
Query Qdrant with the track's embedding for the top-K nearest stored vectors,
each carrying metadata: `global_id`, `camera_id`, `last_seen_timestamp`,
`last_known_box`/zone. K is small (e.g. 10). This narrows millions of stored
identities to a handful of *appearance-plausible* candidates. **Recall step
only** — no decision yet.

### 3.2 Constraint filtering & scoring (the actual intelligence)
Each candidate gets a fused score. Appearance is necessary but not sufficient;
the other constraints are what break the ties appearance can't:

- **Appearance similarity** — cosine from Qdrant. The base signal.

- **Temporal constraints** — a person cannot be in two places at once. If a
  candidate `global_id` was seen on *another* camera at a time incompatible with
  reappearing here (Δt too small for any path between the cameras), it is
  **rejected outright**, regardless of cosine. Conversely a very long Δt lowers
  confidence (appearance drifts: lighting, the person put on a coat).

- **Camera topology constraints** — a hand-authored graph of which cameras are
  physically adjacent and the plausible transit-time window between each pair
  (camera A exit → camera B entry takes 5–20 s). A candidate last seen on a
  camera with **no path** to this one within the elapsed time is rejected. This
  is the single biggest false-match killer in multi-camera ReID: it forbids
  teleportation.

- **Motion constraints** — within a camera (and across adjacent cameras with
  overlap), the track's trajectory/velocity must be consistent with the
  candidate's last known position and heading. A match that requires an
  impossible jump across the frame, or motion against the established direction,
  is down-weighted or rejected. (Kalman/track-continuity level check.)

- **Confidence fusion** — combine the surviving signals into one score, e.g. a
  weighted/log-linear combination `w_a·appearance + w_t·temporal_plausibility +
  w_m·motion_plausibility`, with topology/temporal impossibilities acting as
  hard gates (multiply by 0) *before* fusion. Weights are tunable per
  deployment; the fusion policy lives here and nowhere else.

### 3.3 Decision
- **Best fused score ≥ match threshold** → assign that candidate's `global_id`.
- **Below threshold (or all candidates gated out)** → **mint a new `global_id`**
  (this is a person we haven't seen, or can't safely match). Being conservative
  here — preferring a new id over a wrong merge — is usually the right default;
  a spurious merge corrupts two identities' histories, while a spurious split can
  be reconciled later.
- **Ambiguous** (top two candidates within a small margin) → defer / hold the
  track as "unresolved" for a few more frames rather than commit a coin-flip.

### 3.4 Commit
Assign `global_id` to the track, then **update state**: upsert the (possibly
averaged / feature-banked) embedding into Qdrant under that `global_id`, and
update `last_seen` camera/time/position. The gallery/feature-bank policy (keep N
recent vectors per id? running mean? per-camera exemplars?) lives here — this is
the "averaging" concern deliberately kept **out** of TrackEmbedder.

---

## 4. Responsibility boundaries (the whole point)

| Component | Owns | Must NOT do |
|---|---|---|
| ReIDExtractor | crop → embedding | know tracks, ids, cameras, time; assign identity |
| TrackEmbedder | attach embedding to a track (throttled) | assign identity; search DB; average galleries |
| Qdrant | store vectors; return nearest candidates | decide identity; own thresholds/policy |
| **Identity Service** | **fuse signals; own `global_id`; govern writes** | (it is the one place all this is allowed) |

Everything above the Identity Service produces **measurements**. The Identity
Service is the single place that turns measurements into a **decision**, using
context (topology, time, motion) that none of the other components have. Keep it
that way and each piece stays simple, testable, and swappable.

---

## 5. Open questions to resolve at implementation time (not now)
- Per-track vs per-frame decision cadence, and how to revise an early decision.
- Exact fusion function and how weights/thresholds are calibrated per site.
- Gallery/feature-bank strategy and Qdrant retention/TTL.
- Handling re-entry after long absence (appearance drift) vs new person.
- Source of the camera topology graph (manual config vs learned).
