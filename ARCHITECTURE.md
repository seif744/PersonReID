# Architecture

Multi-camera person re-identification (ReID) over recorded CCTV. Given one video
per camera, the system detects and tracks people, embeds their appearance, and
assigns a **reid id** that is stable for the same real person **across cameras**
and **across re-appearances** within a camera. The pipeline is fully
self-contained: it builds and owns its own Qdrant gallery — there is no separate
registration step or external service. Output: annotated videos (boxes +
reid-id labels, with `global_id` kept for compatibility) and a console
run-summary. Per-person crop images are optional (`crops.save`, off by default).

---

## 1. Design principle: separation of concerns

Three layers, kept strictly independent (see `src/identity/DESIGN.md`):

| Layer | Module | Knows about | Does NOT know about |
|---|---|---|---|
| **Appearance** | `reid/extractor.py` | one crop → 512-d vector | cameras, time, identity |
| **Storage** | `database/store.py` | vectors + payloads, nearest-neighbour search | how vectors are produced, camera topology |
| **Identity** | `identity/service.py`, `identity/reconcile.py` | appearance + **where** + **when** | how the model computes a vector |

Why: the model sees one crop with no context; the store only knows vector
distance. Only the identity layer combines appearance with *where* and *when*
a person was seen, so only it is allowed to decide who someone is. That
boundary keeps the code maintainable.

ADR-002 adds two more stages **inside** the identity layer's decision, between
"find candidates" and "decide":

| Stage | Module | Purpose |
|---|---|---|
| **Re-ranking** | `identity/reranking.py` | camera-aware k-reciprocal + Jaccard re-ranking of candidates, on top of raw cosine |
| **Verification** | `identity/verifier.py` | scores each candidate's P(same identity) from multiple signals instead of a single rigid threshold |

---

## 2. Data flow

```
                        config.yaml
                            │
                            ▼
        ┌──────────────  main.py  ──────────────┐
        │   one worker THREAD per camera video   │
        └────────────────────────────────────────┘
                            │
   per frame, per camera:   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. VideoSource          video file      → frame              │
   │ 2. (optional) resize    frame            → frame (downscaled) │
   │ 3. PersonDetector       frame            → [Detection + track_id]  (YOLO11n + ByteTrack)
   │ 4. CropSaver (optional)  frame+dets      → crops/<cam>/id_<t>/  (only if crops.save)
   │ 5. TrackEmbedder        crop             → det.embedding (512-d)  [shared model lock]
   │      └─ quality gate + occlusion gate + throttle/cache          │
   │ 6. IdentityService      embedding+where+when → det.reid_id     [identity lock]
   │      └─ candidate (Qdrant search) → re-rank (Upgrade 1) →       │
   │         verify (Upgrade 2) → decide → COMMIT to the gallery     │
   │ 7. drawing → live display window; box geometry captured for the │
   │      final render (the video is NOT written in this pass)       │
   └─────────────────────────────────────────────────────────────┘
                            │  (all camera threads join)
                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 8. reconcile_tracklets  OFFLINE, whole-gallery view:           │
   │      rebuild identities from tracklets, merge across cameras   │
   │ 9. render_final_videos  re-draw with FINAL reid ids →           │
   │      output_<cam>.mp4  (same person = same REID in every video) │
   │ 10. print_run_summary   → console only (no files written)      │
   └─────────────────────────────────────────────────────────────┘
```

---

## 3. Components

### VideoSource — `src/video_source.py`
Yields decoded frames from a video file, one at a time.

### PersonDetector — `src/detector.py`
YOLO11n (`yolo11n.pt`, confidence ≥ 0.4, COCO class 0 = person) with ByteTrack.
Returns `Detection(x1,y1,x2,y2, confidence, class_id, track_id, ...)`. `track_id`
is **per-camera** and stable frame-to-frame; it may be `None` until ByteTrack
confirms a box. `crop_person()` is the shared "box → safe crop" primitive.

### CropSaver — `src/crop_saver.py` (optional, off by default)
Only active when `crops.save: true`. Writes one crop per tracked person every
`crops.interval` (10) frames, from the **clean** frame (before boxes are drawn),
to `crops/<camera>/id_<track>/`. Not required by the pipeline — the embedder
crops in-memory — it's for debugging / collecting ReID training data.

### ReIDExtractor — `src/reid/extractor.py`
OSNet-AIN-x1_0, loaded once and shared. Pipeline per crop: BGR→RGB → resize
**256×128** → `/255` → ImageNet mean/std → forward in `eval()` mode (512-d
feature) → **L2-normalize**. Unit vectors ⇒ cosine similarity == dot product.
Preprocessing is pinned to match training or embeddings silently degrade.

### TrackEmbedder — `src/reid/service.py`
One per camera (cache keyed by per-camera `track_id`). `process()`:
- **Throttle + cache**: re-embed a track at most every `reid.interval` (10)
  frames; reuse the cached vector otherwise. Evict tracks after `ttl` (300)
  unseen frames.
- **Quality gate** (`_crop_quality`): reject on size / box-area-ratio / aspect /
  blur (Laplacian variance) / brightness.
- **Occlusion gate** (`_occlusion_ratio`): reject if another person's box covers
  more than `max_occlusion_ratio` (0.5) of this box — keeps multi-body crops out
  of the gallery.
- Accepted crops go through **one batched forward pass**; rejected crops keep the
  track's last good vector.

### Registry lookup helper — `main.py`
After embedding a track, `registry_match_for_embedding()` searches the existing
`registry_gallery` collection and returns a label payload when the top hit clears
`registry.threshold`.
- Read-only by design: it does `search()` only.
- Uses the same 512-d cosine embeddings as the Registry service.
- The label formatter prefers `name`, then `employee_id`, and falls back to
  `person_id` if that is all the payload provides.

### PersonVectorStore — `src/database/store.py`
Thin Qdrant wrapper. This is the ONLY gallery the pipeline uses — the identity
layer both writes to it and searches it.
- `add`/`add_many(embedding, payload)` → one point per raw observation.
- `search(embedding, k)` → k nearest by cosine.
- `set_global_id(points, gid)` / `clear_global_id(points)` → payload rewrites
  used by offline reconciliation.
- Backend precedence `client > url > path`; config uses `store.url` for a
  shared Docker/Cloud server, `store.path` for a local single-process folder.

### IdentityService — `src/identity/service.py`
The one component allowed to decide WHO someone is.
- Sticky per track: decide once on a track's first observation, cache
  `(camera, track_id) → reid_id`, reuse thereafter. `global_id` is still
  written in parallel for compatibility with older payload readers.
- `assign()` → `_match_or_mint()`:
  1. **Candidate retrieval**: `store.search()` for the top-`top_k` raw hits,
     filtered to already-identified, non-same-camera-overlapping candidates.
  2. **Re-ranking** (`identity.rerank.enabled`, Upgrade 1): if enabled,
     `CameraAwareReranker` re-scores candidates using real k-reciprocal
     neighbour sets + Jaccard overlap over identity prototypes, with a wider
     neighbourhood window for cross-camera pairs (see §5).
  3. **Verification** (`identity.verification.enabled`, Upgrade 2): if enabled,
     `Verifier` scores each candidate's `P(same identity)` from cosine, the
     re-ranked score, observation count, recency, and crop quality — replacing
     the plain `score >= threshold and gap >= min_score_gap` check. If
     disabled, falls back to that original plain check.
  4. Accept the best candidate only if its score/probability clears the
     configured threshold **and** beats the runner-up by the configured
     margin; otherwise mint a new identity. The same-camera-overlap exclusion
     is a **hard** pre-filter in both paths — never something a soft signal
     can override (two people present in the same camera at the same time can
     never be merged).
- `_commit()` stores the observation under the chosen id, updates its
  prototype bank, span cache, observation count, and last-seen record.
- Warm start rebuilds all of the above from the store on startup, so a
  persistent gallery survives process restarts.

### CameraAwareReranker — `src/identity/reranking.py` (ADR-002 Upgrade 1)
Real camera-aware k-reciprocal + Jaccard re-ranking (CA-Jaccard, CVPR'24) —
**not** `cosine × camera_weight`. Operates over per-identity prototypes (the
mean-pooled, re-normalized bank vectors `IdentityService` already keeps in
memory), since Qdrant only holds raw per-observation points.
- Builds a cached pairwise-cosine neighbour graph over all known identity
  prototypes (cheap on CPU — tens of identities, not millions); rebuilt only
  when the gallery signature changes.
- For a query embedding: finds its reciprocal neighbour set among known
  identities, widening the effective neighbourhood window
  (`cross_camera_k1_boost`) for cross-camera comparisons, since cross-camera
  cosine similarity runs systematically lower.
- Blends `lambda * cosine + (1 - lambda) * (1 - jaccard_distance)`.
- **Small-gallery guard**: with fewer known identities than `k1 + 1`,
  reciprocal sets degenerate to trivial/empty sets and would fabricate a
  "perfect" Jaccard overlap regardless of match quality — below that floor,
  it falls back to plain cosine instead.

### Verifier — `src/identity/verifier.py` (ADR-002 Upgrade 2)
A HEURISTIC placeholder for the "verification layer" — a fixed-weight logistic
combination of signals, not (yet) a trained model, since there is no labeled
same/different-person dataset for this deployment. The interface —
`Verifier.score(features) -> float` — is the same shape a trained
classifier would expose (`model.predict_proba(...)`), and every decision's
full feature vector + resulting probability is appended to
`identity.verification.log_path` (`logs/verification_decisions.jsonl`), so a
real MLP/GBT can be trained on accumulated deployment data later with **no
change to the calling code**.
- Features: cosine score, re-ranked score, `log1p(observation_count)`, frames
  since last same-camera sighting, and a crop-quality scalar.
- Weights are calibrated so cosine ≈ 0.63 sits near the decision boundary --
  measured on this project's footage with the `osnet_x1_0_msmt17` checkpoint.
  After swapping to `osnet_ain_x1_0` (see §6), the decision log
  (`logs/verification_decisions.jsonl`) showed accepted matches averaging
  cosine ~0.79 (min 0.53), rejected candidates averaging ~0.45 (p90 0.59) --
  a wider, cleaner gap. Raising `accept_threshold`/`min_prob_gap` (0.55→0.62,
  0.05→0.08) to exploit that margin was tried but measurably hurt overall
  accuracy (97% -> 89%, more false splits) and was reverted; left at 0.55/0.05.
  **Re-measure and re-anchor these weights whenever the ReID checkpoint or
  camera domain changes** — they are specific to this model's score distribution, not a
  universal constant.

### Legacy reconcile_tracklets — `src/identity/reconcile.py`
Runs once after all cameras finish, with the full gallery visible. Rebuilds
identities from scratch (does **not** trust live `reid_id`s); the **tracklet**
(one camera's view of one track) is the unit of evidence.
1. **Gather** points by `(camera, track_id)` → vectors, point ids, frame span,
   gids.
2. **Suppress** tracklets with `< min_tracklet_observations` (3) via
   `clear_global_id` — 1–2-frame detector blips don't become people.
3. **Prototype** per tracklet (mean of normalized vectors).
4. **Union-find** with a hard conflict guard: same-camera, time-**overlapping**
   tracklets can never merge (provably two people).
5. **Phase 1 — same-camera defrag**: merge same-camera, time-disjoint pairs with
   cosine ≥ `same_camera_threshold` (0.90).
6. **Phase 2 — cross-camera, iterated to convergence**: on cluster prototypes,
   merge different-camera pairs with cosine ≥ `threshold`
   (`identity.reconcile.threshold`, 0.63) **and** (if `require_reciprocal_best`)
   mutual nearest neighbour — stops one tracklet absorbing multiple
   look-alikes. This is not a single pass: after any merges, cluster
   prototypes change, so scores and reciprocal-best partners are recomputed
   from scratch and another round runs, until a round merges nothing. This
   matters for **chains** of mutually-similar fragments — e.g. A's own best
   match is B, but B's own best match is C, not A — which a single pass would
   leave stuck (A↔B never merges because B "prefers" C). Iterating lets the
   chain fully consolidate (A merges with B in round 1; the merged A+B
   cluster's new prototype then reciprocally matches C in round 2) while every
   round still enforces the same mutual-best-match safety rule — merging never
   gets easier than that rule allows, it just gets more chances to apply as
   clusters grow. Each merge strictly reduces the number of clusters by one, so
   this always terminates. This runs **independently of the verifier** above
   (its own separate threshold, not the verifier's `accept_threshold`) —
   recalibrate both together if you change the ReID checkpoint.
7. **Survivor**: each final cluster keeps the smallest existing reid ID
   (deterministic); all its points rewritten via `set_global_id` so both
   `reid_id` and `global_id` stay aligned.

### render_final_videos — `main.py`
Second render pass, after reconciliation. The live pass only captured per-frame
box geometry; this maps each `(camera, track_id)` to its FINAL reid ID
(`build_gid_map`, read back from the store) and re-draws the source frames so the
same person carries the **same REID and colour in every camera's `output_<cam>.mp4`**.

### print_run_summary — `main.py`
Scrolls the store for this `run_id`; counts observations per camera and distinct
reid IDs; flags IDs seen in >1 camera as cross-camera. **Console output only —
writes no files.**

---

## 4. Concurrency

- One **worker thread per camera video**; each owns its detector (independent
  ByteTrack state), crop saver, and TrackEmbedder.
- **Shared, lock-guarded**: the ReID model (`model` lock — one forward pass at a
  time) and the identity service (`identity` lock — serializes `assign()`,
  since it both searches and writes the shared gallery).
- OpenCV windows aren't thread-safe: workers publish frames to a shared buffer;
  the main thread displays (or just joins, headless).

---

## 5. ADR-002: why re-ranking + verification, not just a better threshold

Testing surfaced the failure mode that motivated this: with the original
OSNet/Market1501 checkpoint on this footage, **different people scored closer
than the same person** (a genuine same-person cross-camera match at cosine
~0.72-0.76 vs. different-but-similar-looking people reaching ~0.83). The two
distributions overlapped, so **no single cosine threshold was simultaneously
correct** — raising it fixed false merges but created false splits, and vice
versa. Switching to an OSNet/MSMT17 checkpoint substantially widened that gap
on this footage, and a later swap to OSNet-AIN (`osnet_ain_x1_0`, same family,
added Adaptive Instance Normalization for domain generalization) widened it
further still (see §6) — but the underlying risk (some future camera/domain
reintroducing overlap) is exactly what re-ranking and verification exist to
guard against, not something a single "right" checkpoint permanently solves.

Design principles this drives:
- **False merges are worse than false splits.** A merge fuses two people's
  histories and corrupts both; a split just gives one person a spare ID,
  which reconciliation can still fix later. When uncertain, mint a new
  identity rather than guess.
- **Appearance generates candidates; it doesn't decide identity alone.**
  Re-ranking (Upgrade 1) and verification (Upgrade 2) exist to combine
  appearance with additional structure (neighbourhood overlap, observation
  history, crop quality) before a decision is made — not to replace the
  embedding model. Swapping the ReID *checkpoint* (same OSNet architecture,
  same 512 dimensions) stays in-bounds of that constraint; introducing a
  transformer backbone would not.
- The hard same-camera-overlap physical exclusion is never overridden by a
  soft signal — a physical impossibility is a different kind of fact than a
  similarity score, and stays a hard veto in both the legacy and new decision
  paths.

**Deferred** (documented, not implemented): prototype confidence/variance with
adaptive per-identity thresholds; a camera transition graph rejecting
physically-impossible transitions; a MetaBIN training pass to reduce the
underlying domain gap. These need either real camera topology or accumulated
deployment data this project doesn't have yet — see
`identity/verifier.py`'s decision log as the mechanism that starts collecting
the latter.

---

## 6. Known limitations (model, not plumbing)

The pipeline is correct; the **embedding model is the ceiling** on this domain.

- **History**: the original OSNet/Market1501 checkpoint was out-of-domain for
  this CCTV footage. Measured directly on this project's own videos: genuine
  same-person matches across two different video sessions only reached
  **cosine 0.65–0.76** (even after denoising via prototype averaging), while
  *different* people reached **0.70–0.83** — the distributions **overlapped**,
  so no fixed threshold could perfectly separate them.
- **Previous default (OSNet/MSMT17)**: re-measured on the same footage, this
  checkpoint separated the two cases better than Market1501 — different
  people's prototype cosine topped out **~0.48–0.57**, genuine same-person
  cross-video matches reached **~0.70–0.80** — but on this project's actual
  CCTV footage the two clusters still ran close enough (~0.72 vs. ~0.55) that
  a domain-generalization backbone was worth trying.
- **Current default (OSNet-AIN, `osnet_ain_x1_0`)**: same OSNet family and
  512-d embedding, with Adaptive Instance Normalization added specifically to
  generalize to unseen camera domains — a near-drop-in swap (same torchreid
  API, same 256×128 preprocessing). Measured from the decision log on this
  project's 3-video run: accepted (same-person) matches average cosine
  **~0.79** (min 0.53), rejected candidates average **~0.45** (p90 0.59) — a
  wider, cleaner gap than MSMT17's. Despite that margin, raising
  `identity.threshold`/`identity.reconcile.threshold` (to 0.68) and the
  verifier's `accept_threshold`/`min_prob_gap` (to 0.62/0.08) to exploit it
  was tried and reverted -- it measurably hurt end-to-end accuracy (97% ->
  89%) via more false splits, so all four stay at their MSMT17-era values
  (0.63 / 0.55 / 0.05) for now (§3, §7). The occlusion crop-quality gate
  (`max_occlusion_ratio`) was also tried tighter (0.35) to fight a rare
  occlusion-triggered false merge and reverted for the same reason -- see §7.
- **This is footage-and-checkpoint-specific, not solved in general.** A
  different deployment (different cameras, lighting, clothing diversity)
  could easily reintroduce distribution overlap even with OSNet-AIN. Re-run
  `identity/demo_identity.py`'s sweep (or inspect
  `logs/verification_decisions.jsonl` directly) whenever you point this at
  new footage, and re-anchor the verifier's weights and thresholds if the gap
  changes. If overlap persists, escalate to a foundation backbone (CLIP-ReID /
  SOLIDER) — that's a bigger change (different embedding dimension, Qdrant
  schema rebuild, new preprocessing) done as a separate step.
  That's why re-ranking and verification exist as a safety net on top of raw
  cosine, and why the verifier logs decisions instead of claiming to be a
  finished, trained classifier.
- **If overlap reappears, the fix is still model-side**, roughly cheapest
  first:
  1. Try other available pretrained checkpoints (already done once here).
  2. Fine-tune OSNet on crops collected from your own cameras (`crops.save:
     true` already collects these) with a metric-learning loss (triplet /
     ArcFace) — directly targets a specific deployment's domain gap.
  3. Synthetic pretraining (RandPerson) and/or MetaBIN (see §5) — the
     "textbook" fix, but there is currently no training pipeline in this repo
     to build on (an earlier `tools/`/`notebooks/` scaffold for this was
     removed) — this would mean building one from scratch, with real GPU
     compute and experimentation time.

---

## 7. Key configuration (`config.yaml`)

| Key | Value | Meaning |
|---|---|---|
| `detector.confidence_threshold` | 0.4 | drop weak detections |
| `tracker.config` | bytetrack.yaml | tracker |
| `crops.interval` | 10 | save a crop every N frames |
| `reid.weights` | osnet_ain_x1_0.pth | ReID checkpoint — recalibrate everything below if this changes (§6) |
| `reid.interval` / `ttl` | 10 / 300 | re-embed cadence / cache eviction |
| `reid.quality.max_occlusion_ratio` | 0.5 | reject multi-body crops |
| `store.enabled` | true | the pipeline's own gallery — always on for the default flow |
| `store.url` / `store.path` | http://localhost:6333 / qdrant_data | shared server vs. local single-process folder |
| `identity.enabled` | true | assigns and merges reid ids |
| `identity.threshold` / `min_score_gap` | 0.63 / 0.03 | plain-path match acceptance (used when verification is disabled) |
| `identity.top_k` / `bank_size` | 50 / 20 | candidates considered / prototype bank size |
| `identity.rerank.enabled` | true | ADR-002 Upgrade 1 |
| `identity.rerank.k1` / `cross_camera_k1_boost` | 8 / 4 | reciprocal-neighbourhood size / cross-camera widening |
| `identity.rerank.lambda_` | 0.7 | cosine vs. Jaccard blend weight |
| `identity.verification.enabled` | true | ADR-002 Upgrade 2 |
| `identity.verification.accept_threshold` / `min_prob_gap` | 0.55 / 0.05 | P(same) acceptance / runner-up margin |
| `identity.verification.log_path` | logs/verification_decisions.jsonl | decision log for future model training |
| `identity.reconcile.threshold` | 0.63 | cross-camera merge bar — separate from the verifier, recalibrate together |
| `identity.reconcile.same_camera_threshold` | 0.90 | same-camera defrag merge |
| `identity.reconcile.min_tracklet_observations` | 3 | below this = detector noise |
| `identity.reconcile.require_reciprocal_best` | true | mutual-NN guard on cross-camera merges |
