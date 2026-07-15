# Registry Integration — read this before touching identity matching

This file is a handoff note for whoever wires Inference up to the **Register
service** (a sibling repo at `../Registry`, already built and working). It
explains what Register does, what Inference needs to add, and — just as
important — what Inference must **not** do.

## TL;DR

Register is a separate, already-complete service that lets a human operator
enroll known people (name + employee id) into a Qdrant collection called
`registry_gallery`, on the **same local Qdrant server** Inference already
uses (`http://localhost:6333`). Inference's job is to **read** that
collection to recognize known people — it must never write to it. Everything
else about Inference's current pipeline (detect → track → embed → your
existing `persons`-collection identity logic → draw boxes → render video)
stays exactly as it is today. This is purely an additional **lookup + label**
step layered on top of what already runs.

## What Register does (context, not your code to touch)

Register (`../Registry`) is a separate CLI + Streamlit app an operator runs
manually, offline from Inference:

1. `python register_main.py --source clip.mp4` — captures a video/stream,
   runs the same YOLO11n + ByteTrack detection/tracking Inference uses,
   samples a handful of good-quality crops per track, and stages them on
   disk (`pending_review/`). It does **not** touch Qdrant at this stage.
2. `streamlit run registry_app.py` — a human reviews those crops in a
   browser, assigns each track to a new or existing person (name + optional
   employee id), and approves specific crops.
3. Only for approved crops: embeds them with the **same OSNet checkpoint**
   Inference uses (`osnet_x1_0_market1501.pth`, 512-d, L2-normalized,
   cosine distance) and upserts them into Qdrant collection
   `registry_gallery`.

So `registry_gallery` is a **curated, human-verified** gallery of named
people — smaller and cleaner than Inference's own `persons` collection
(which is unsupervised observations + minted global ids). The two
collections live side by side on the same Qdrant server and never overlap.

### `registry_gallery` schema (what you'll query)

- Collection name: `registry_gallery`
- Vector: 512-d, COSINE distance (identical convention to Inference's own
  `persons` collection — same embedding model, same normalization, so
  cosine scores from both are directly comparable)
- Payload per point (one point = one approved crop):
  ```json
  {
    "person_id": 17,
    "name": "John Doe",
    "employee_id": "EMP001",
    "image_path": "registry/person_0017/crop_003.jpg",
    "created_at": "2026-07-14T10:32:00",
    "quality": 0.91,
    "camera": "cam_219",
    "track_id": 42,
    "frame": 1234
  }
  ```
  A real person typically has several points (one per approved crop), all
  sharing the same `person_id`/`name`/`employee_id`.

- Server: same Qdrant instance Inference already points at
  (`QDRANT_URL` / `store.url` in your own config — no new server, no new
  connection setup needed).

## What Inference needs to add

A **read-only lookup**, additive to the pipeline you already have:

1. After a detection's embedding is computed (same point in `main.py` where
   `embedder.process(...)` runs and `fresh` detections get their
   `det.embedding`), also search `registry_gallery` with that embedding.
2. Use `PersonVectorStore` — it's already generic (collection name is a
   constructor argument), so you don't need a new class, just a second
   instance pointed at `registry_gallery` instead of `persons`:

   ```python
   from database.store import PersonVectorStore

   registry_store = PersonVectorStore(
       url=os.environ.get("QDRANT_URL") or store_cfg.get("url"),
       collection="registry_gallery",
   )
   ```
3. Call `registry_store.search(det.embedding, limit=5)` (the exact same
   method your `IdentityService` already calls on the `persons` store —
   `add`, `add_many`, `set_global_id`, `clear_global_id`, and `reset` on
   this store must simply never be called). Take the top hit; if
   `hit.score >= registry.threshold` (config-driven, calibrate like your
   existing `identity.threshold` — start around 0.85), treat it as a known
   identity: `hit.payload["person_id"]`, `["name"]`, `["employee_id"]`.
4. Feed that into your existing output path — `drawing.py`'s
   `draw_detections()` already renders a label per box (`"GID {n}"` today);
   extend the label to prefer the registered name when a confident match is
   found, e.g. `"John Doe (EMP001)"` instead of / alongside `"GID {n}"`.
   No new rendering pipeline — same boxes, same annotated-video output, same
   `print_run_summary` — just a richer label when a match exists.
5. Because Register upserts to the **same live Qdrant server**, a person
   registered while Inference is running becomes matchable on Inference's
   very next lookup — **no restart required**.

### Suggested config additions (additive only — don't restructure existing keys)

```yaml
registry:
  enabled: true
  collection: registry_gallery
  threshold: 0.85     # cosine to accept a registry match; calibrate like identity.threshold
  top_k: 5
```

## What Inference must NOT do

- **Never write to `registry_gallery`**: no `add`, `add_many`,
  `set_global_id`, `clear_global_id`, or `reset` calls against the
  `PersonVectorStore` instance pointed at it. Read (`search`, `count`,
  `scroll` if you need it for debugging) only.
- **Don't merge this with `IdentityService`'s own gallery logic.** Keep it a
  separate, parallel lookup — `IdentityService` still owns `persons` and its
  own global-id minting/reconciliation exactly as today. The registry lookup
  is an independent enrichment step: "is this a known, named person?" is a
  different question from "have I seen this exact appearance before?"
- **Don't run Register's own scripts from Inference**, and don't create,
  modify, or delete anything under `../Registry/registry/` or
  `../Registry/pending_review/` — those are Register's own outputs.
- **Don't change Register's collection schema.** If a payload field you need
  isn't listed above, that's a signal to update Register (a separate repo),
  not to write a different shape into `registry_gallery` from here.

## Quick sanity check once wired up

1. From `../Registry`: capture + review a short clip, register at least one
   person.
2. Confirm (read-only) that `registry_gallery` has points:
   `PersonVectorStore(url=..., collection="registry_gallery").count()`.
3. Run Inference on footage containing that same person; confirm the
   rendered output label shows their name instead of a bare `GID`, and that
   nothing in `persons` or `registry_gallery`'s point count changed as a
   *side effect* of Inference running (Inference should only ever grow
   `persons`, never `registry_gallery`).
