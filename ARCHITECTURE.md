# Architecture

Multi-camera person re-identification (ReID) over recorded CCTV. Given one video
per camera, the system detects and tracks people, embeds their appearance, and
assigns a **global ID** that is stable for the same real person **across cameras**
and **across re-appearances** within a camera. Outputs: annotated videos, saved
crops, and a cross-camera report.

---

## 1. Design principle: separation of concerns

Three layers, kept strictly independent (see `src/identity/DESIGN.md`):

| Layer | Module | Knows about | Does NOT know about |
|---|---|---|---|
| **Appearance** | `reid/extractor.py` | one crop → 512-d vector | cameras, time, identity |
| **Storage/retrieval** | `database/store.py` | vectors + metadata, nearest-neighbour | who anyone is |
| **Identity** | `identity/service.py`, `identity/reconcile.py` | appearance + **where** + **when** | how vectors are produced |

Why: the model sees one crop with no context; the store only knows vector
distance. Only the identity layer has the camera map and the clock, so only it
can decide identity. This boundary is load-bearing — it keeps failure modes
isolated and the code maintainable.

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
   │ 4. CropSaver            frame+dets       → crops/<cam>/id_<t>/f<frame>.jpg
   │ 5. TrackEmbedder        crop             → det.embedding (512-d)  [shared model lock]
   │      └─ quality gate + occlusion gate + throttle/cache          │
   │ 6. IdentityService      embedding+where+when → det.global_id   [identity lock]
   │      └─ also COMMITS the observation to the vector store        │
   │ 7. drawing + VideoWriter → output_<cam>.mp4                     │
   └─────────────────────────────────────────────────────────────┘
                            │  (all camera threads join)
                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 8. reconcile_tracklets  OFFLINE, whole-gallery view:           │
   │      rebuild identities from tracklets, merge across cameras   │
   │ 9. print_run_summary    → reports/cross_camera_matches/        │
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

### CropSaver — `src/crop_saver.py`
Writes one crop per tracked person every `crops.interval` (10) frames, from the
**clean** frame (before boxes are drawn), to `crops/<camera>/id_<track>/`.

### ReIDExtractor — `src/reid/extractor.py`
OSNet-x1_0, loaded once and shared. Pipeline per crop: BGR→RGB → resize
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

### PersonVectorStore — `src/database/store.py`
Qdrant wrapper. Collection `persons`, dim 512, distance COSINE. Each point is
**one observation**: UUID id, payload `{camera, track_id, frame, global_id,
run_id, crop_quality}`.
- `search(embedding, k)` → k nearest by cosine.
- `set_global_id(points, gid)` / `clear_global_id(points)` → payload rewrites
  used by reconciliation.
- Backend precedence `client > url > path`; config uses `url:
  http://localhost:6333` (Docker), with `path: qdrant_data` as local fallback.

### IdentityService — LIVE — `src/identity/service.py`
The `candidate → decide → commit` loop, serialized under the identity lock.
- **Sticky per track**: decide once on a track's first observation, cache
  `(camera, track_id) → global_id`, reuse thereafter (prevents ID flicker) while
  still committing each new observation.
- **`_match_or_mint`**:
  1. `search(top_k=50)` for nearest observations.
  2. Filter a candidate out if it's same-camera+same-frame, has no `global_id`,
     or fails `_same_camera_overlap` (that ID already spans this frame in this
     camera → two co-present bodies).
  3. Score each candidate by its **bank prototype** (renormalized mean of its
     last `bank_size`=20 embeddings); fall back to raw hit score.
  4. **Accept** iff `best ≥ threshold (0.85)` **and** `best − second ≥
     min_score_gap (0.03)`; else **mint** a new global ID.
- **Commit**: store the observation under the chosen ID, update the bank and the
  per-camera time `_spans` (used by the overlap guard).
- **Warm start**: rebuilds banks/spans and the ID counter from the store on
  construction (survives restarts / persistent store).

### reconcile_tracklets — OFFLINE — `src/identity/reconcile.py`
Runs once after all cameras finish, with the full gallery visible. Rebuilds
identities from scratch (does **not** trust live `global_id`s); the **tracklet**
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
6. **Phase 2 — cross-camera**: on cluster prototypes, merge different-camera pairs
   with cosine ≥ `threshold` (0.85) **and** (if `require_reciprocal_best`) mutual
   nearest neighbour — stops one tracklet absorbing multiple look-alikes.
7. **Survivor**: each final cluster keeps the smallest existing global ID
   (deterministic); all its points rewritten via `set_global_id`.

### print_run_summary — `main.py`
Scrolls the store for this `run_id`; counts observations per camera and distinct
global IDs; flags IDs seen in >1 camera as cross-camera; writes
`reports/cross_camera_matches/summary.txt` + per-ID contact sheets.

---

## 4. Concurrency

- One **worker thread per camera video**; each owns its detector (independent
  ByteTrack state), crop saver, and TrackEmbedder.
- **Shared, lock-guarded**: the ReID model (`model` lock — one forward pass at a
  time), the store, and the IdentityService (`identity` lock — `assign()` is
  serialized, so it is the sole store writer and its state mutations are safe).
- OpenCV windows aren't thread-safe: workers publish frames to a shared buffer;
  the main thread displays (or just joins, headless).

---

## 5. Why two stages (live + offline)

The live decision is **sticky** to prevent per-frame ID flicker, but that creates
a blind spot: a person in **two cameras at the same instant** is decided in each
camera before the other has committed anything, so both mint separate IDs and can
never reconcile live. `reconcile_tracklets` exists precisely to repair this with
a whole-gallery view. Hence: live assignment **and** an offline pass.

Design bias throughout: **prefer a split over a merge.** A wrong merge fuses two
people's histories and is unrecoverable; a split (one person given a spare ID) is
recoverable. Thresholds are set high on purpose.

---

## 6. Known limitations (model, not plumbing)

The pipeline is correct; the **embedding model is the ceiling** on this domain.

- OSNet/Market-1501 is out-of-domain for this CCTV. Measured on the sample
  footage: same-person cross-camera cosine falls to **0.68–0.82** while
  *different* people reach **0.84–0.86**. The distributions **overlap**, so **no
  single threshold** is simultaneously correct.
- Effect on the sample clip: ~8 real people are correctly separated; the count
  reports a few extra short fragments (safe splits); cross-camera linking catches
  the easy front-facing pair and misses hard back-view/occluded pairs.
- **The fix is a domain-appropriate embedding model** — synthetic pretrain
  (RandPerson) and/or fine-tuning on target-domain crops — to widen the
  same-vs-different gap so a threshold works. The identity layer is correct
  scaffolding around that model.

---

## 7. Key configuration (`config.yaml`)

| Key | Value | Meaning |
|---|---|---|
| `detector.confidence_threshold` | 0.4 | drop weak detections |
| `tracker.config` | bytetrack.yaml | tracker |
| `crops.interval` | 10 | save a crop every N frames |
| `reid.interval` / `ttl` | 10 / 300 | re-embed cadence / cache eviction |
| `reid.quality.max_occlusion_ratio` | 0.5 | reject multi-body crops |
| `identity.threshold` | 0.85 | live match acceptance (cosine) |
| `identity.min_score_gap` | 0.03 | best must beat runner-up by this |
| `identity.top_k` / `bank_size` | 50 / 20 | candidates pulled / prototype bank |
| `identity.reconcile.same_camera_threshold` | 0.90 | same-camera defrag merge |
| `identity.reconcile.min_tracklet_observations` | 3 | below this = detector noise |
| `identity.reconcile.require_reciprocal_best` | true | mutual-NN guard on merges |
