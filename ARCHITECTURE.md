# Architecture

Multi-camera person re-identification (ReID) over recorded CCTV. Given one video
per camera, the system detects and tracks people, embeds their appearance, and
assigns a **global ID** that is stable for the same real person **across cameras**
and **across re-appearances** within a camera. Output: annotated videos (boxes +
global-ID labels) and a console run-summary. Per-person crop images are optional
(`crops.save`, off by default).

---

## 1. Design principle: separation of concerns

Three layers, kept strictly independent (see `src/identity/DESIGN.md`):

| Layer | Module | Knows about | Does NOT know about |
|---|---|---|---|
| **Appearance** | `reid/extractor.py` | one crop → 512-d vector | cameras, time, identity |
| **Registry lookup** | `main.py`, `database/store.py` | registry gallery + nearest-neighbour lookup | how vectors are produced, camera topology |
| **Legacy identity** | `identity/service.py`, `identity/reconcile.py` | appearance + **where** + **when** | registry labels, current default inference path |

Why: the model sees one crop with no context; the registry only knows vector
distance plus human-curated payloads. Legacy identity logic still exists for
older runs, but the default inference path is read-only lookup against
`registry_gallery`, then annotation. That boundary keeps the code maintainable
and prevents inference from mutating shared gallery state.

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
   │ 6. IdentityService      embedding+where+when → det.global_id   [identity lock]
   │      └─ also COMMITS the observation to the vector store        │
   │ 7. drawing → live display window; box geometry captured for the │
   │      final render (the video is NOT written in this pass)       │
   └─────────────────────────────────────────────────────────────┘
                            │  (all camera threads join)
                            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 8. reconcile_tracklets  OFFLINE, whole-gallery view:           │
   │      rebuild identities from tracklets, merge across cameras   │
   │ 9. render_final_videos  re-draw with FINAL global ids →         │
   │      output_<cam>.mp4  (same person = same GID in every video)  │
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

### Registry lookup helper — `main.py`
After embedding a track, `registry_match_for_embedding()` searches the existing
`registry_gallery` collection and returns a label payload when the top hit clears
`registry.threshold`.
- Read-only by design: it does `search()` only.
- Uses the same 512-d cosine embeddings as the Registry service.
- The label formatter prefers `name`, then `employee_id`, and falls back to
  `person_id` if that is all the payload provides.

### PersonVectorStore — `src/database/store.py`
Qdrant wrapper. In this repo it is used in read-only mode for inference against
`registry_gallery`; the legacy identity path can still use writable collections
when enabled.
- `search(embedding, k)` → k nearest by cosine.
- `set_global_id(points, gid)` / `clear_global_id(points)` → payload rewrites
  used by the legacy reconciliation path.
- Backend precedence `client > url > path`; config uses `registry.url`
  for the shared server, with `store.url` / `store.path` kept for compatibility.

### Legacy IdentityService — `src/identity/service.py`
Compatibility layer for the older write-to-store pipeline. The current default
config leaves it disabled.
- Sticky per track: decide once on a track's first observation, cache
  `(camera, track_id) → global_id`, reuse thereafter.
- `_match_or_mint()` searches the gallery, filters same-camera overlaps, scores
  candidates with a prototype bank, and either reuses an id or mints a new one.
- `_commit()` stores the observation under the chosen id and updates the bank.
- Warm start rebuilds banks, spans, and the id counter from the store.

### Legacy reconcile_tracklets — `src/identity/reconcile.py`
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

### render_final_videos — `main.py`
Second render pass, after reconciliation. The live pass only captured per-frame
box geometry; this maps each `(camera, track_id)` to its FINAL global ID
(`build_gid_map`, read back from the store) and re-draws the source frames so the
same person carries the **same GID and colour in every camera's `output_<cam>.mp4`**.

### print_run_summary — `main.py`
Scrolls the store for this `run_id`; counts observations per camera and distinct
global IDs; flags IDs seen in >1 camera as cross-camera. **Console output only —
writes no files.**

---

## 4. Concurrency

- One **worker thread per camera video**; each owns its detector (independent
  ByteTrack state), crop saver, and TrackEmbedder.
- **Shared, lock-guarded**: the ReID model (`model` lock — one forward pass at a
  time) and the registry client (`registry` lock — shared read-only access to
  Qdrant). If the legacy identity pipeline is enabled, its `identity` lock
  serializes `assign()` so the old write path stays safe.
- OpenCV windows aren't thread-safe: workers publish frames to a shared buffer;
  the main thread displays (or just joins, headless).

---

## 5. Legacy identity flow (disabled by default)

The write-to-store path is kept for compatibility, but the default inference flow
does not use it. When enabled, the live decision is **sticky** to prevent
per-frame ID flicker, but that creates a blind spot: a person in **two cameras
at the same instant** is decided in each camera before the other has committed
anything, so both mint separate IDs and can never reconcile live.

Design bias throughout: **prefer a split over a merge.** A wrong merge fuses two
people's histories and is unrecoverable; a split (one person given a spare ID)
is recoverable. Thresholds are set high on purpose.

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
| `registry.enabled` | true | current inference uses Registry lookup |
| `registry.collection` | registry_gallery | shared named-person gallery |
| `registry.url` | http://localhost:6333 | shared Qdrant server |
| `registry.threshold` / `top_k` | 0.85 / 5 | accept the top registry match |
| `store.enabled` | false | legacy identity writes are off by default |
| `identity.enabled` | false | legacy write-to-store path is disabled by default |
| `identity.threshold` / `min_score_gap` | 0.85 / 0.03 | legacy live match acceptance |
| `identity.top_k` / `bank_size` | 50 / 20 | legacy candidates / prototype bank |
| `identity.reconcile.same_camera_threshold` | 0.90 | legacy same-camera defrag merge |
| `identity.reconcile.min_tracklet_observations` | 3 | below this = detector noise |
| `identity.reconcile.require_reciprocal_best` | true | legacy mutual-NN guard on merges |
