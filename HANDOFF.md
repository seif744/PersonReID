# Handoff — Multi-Camera Person ReID: v5 Real-Time RTSP Pipeline

**Repo:** `github.com/seif744/PersonReID` (local dir name: `Inference`)
**Branch:** `research` (== `origin/research`; merged to `main` via PR #13, so both carry the work)
**Targets:** A100 GPU server (real-time); CPU dev box (functional only).

---

## 0. TL;DR — current state

- Building a **v5 real-time streaming pipeline**: N RTSP cams → per-camera detect/track + ReID → online identity → live-annotated `output_<cam>.mp4`. **Raw stream is never saved; no offline reconcile** (hard handoff invariants).
- **Done:** ReID model swap (OSNet-AIN), v5 **Stage 0** (scaffold/capability/routing) and **Stage 1** (full threaded skeleton, real-time, produces valid annotated MP4). Both committed + pushed.
- **Next:** **Stage 3** (the identity engine) — reprioritized ahead of Stage 2 because the current pain is *reid quality*, not throughput.
- **Known issue (expected, not a bug):** on `--mode live` today, reids are unstable — same person gets different ids, no cross-camera persistence. That is exactly what **Stage 3** builds. Stage 1 identity is intentionally minimal (plumbing only).
- **CPU is unusable** for this (choppy, bad ids) — never judge quality on CPU. The GPU server is the target.

---

## 1. Goal & hard invariants (from the v5 handoff)

- N RTSP streams (test with 3; a 4th added later — must be **N-generic**, never hardcode a count).
- Real-time → annotated `output_<cam>.mp4` per camera. Device-portable (GPU when present, else CPU).
- **Invariants (non-negotiable):** raw stream **never saved**; **no live reconcile**; false-merge-conservative (mint when unsure); real-time-first (frame-drop = overload protection); the existing **file-batch path must not break** (behavioural-equivalence regression gate).

## 2. Execution rules (FIXED — highest authority)

1. **Priority order** (never trade a higher for a lower): (1) preserve file-batch path → (2) correctness → (3) identity quality → (4) race-free shutdown/valid MP4s → (5) bounded latency → (6) throughput → (7) GPU optimization → (8) code cleanliness.
2. **No architectural drift:** stage boundaries + queue/drop policies are fixed. If a problem appears, STOP + document + ask — don't collapse stages or merge queues.
3. **ByteTrack stop-rule:** batch the *detector + ReID*, keep **one tracker per camera**. If decoupling needs ultralytics internals or > ~1 stage of effort, fall back to per-cam `.track()` + batched embed. Never rewrite ByteTrack.
4. **Per-stage validation gate:** one stage at a time; after each, run its checks, fix failures, get explicit approval before the next.
5. **Topology is fail-open:** cross-camera veto only rejects pairs it has data for; unknown cameras → appearance-only (never blocked). Adjacency + transit-time data **not yet provided**.

---

## 3. The two run modes (IMPORTANT — do not confuse)

| Mode | What it does | Use |
|---|---|---|
| **`--mode live`** | **v5 real-time** pipeline (`src/live/`). Streams → live-annotated `output_<cam>.mp4`. **No recording, no reconcile.** Ctrl-C finalizes + exits. Identity = online active-set (good only after Stage 3). | **The target.** RTSP. |
| `auto` (default) / `--mode batch` | Original **file-batch** flow (+ a record-then-batch stopgap for URLs). Uses **offline reconcile** (good reids) but **saves raw stream** → **forbidden by v5**. Kept ONLY as the untouched file-batch regression gate. | File inputs / regression only. **Do NOT use for RTSP going forward.** |

The good reids seen historically came from the **file-batch offline reconcile** — a non-causal pass the live path is not allowed to use. Live must reach quality via the **online engine (Stage 3)**, which *approximates* (never equals) offline.

---

## 4. What's been done (all committed on `research`)

- **ReID model swap (validated):** `osnet_x1_0_msmt17` → **`osnet_ain_x1_0`** (multi-source domain-generalization checkpoint), 512-d, same preprocessing. `src/reid/extractor.py` (+ `torch.load(weights_only=False)` and `state_dict`/`module.` unwrap for the checkpoint format). Weight committed at `src/reid/weights/osnet_ain_x1_0.pth`.
- **Threshold tuning — tried & REVERTED:** raising identity/verifier thresholds + tightening occlusion hurt accuracy (~97%→~89%); reverted to `identity.threshold 0.63`, `min_score_gap 0.03`, verifier `0.55/0.05`, `max_occlusion_ratio 0.5`. Documented in `config.yaml` comments.
- **v5 Stage 0 (done, approved):** `src/live/` scaffold — `capabilities.py` (device/NVDEC detection, no hardcoded `.cuda()`), `frame.py` (Frame carrier; `ts` stamped once, never regenerated), `decode_backend.py` (`DecodeBackend` ABC + `CPUDecodeBackend` + `NVDECBackend` stub). `live:` config block. `run.mode` routing + `--mode` CLI flag in `main.py`. File-batch regression **passed/approved**.
- **v5 Stage 1 (done, backbone):** full threaded skeleton — `queues.py` (NewestSlot, DropOldestQueue), `capture.py` (per-cam decode + reconnect), `scheduler.py` (freshness batch), `inference.py` (per-cam detector + embedder, bs=1), `identity_stage.py` (evidence-gate + minimal match-or-mint), `render.py` (draw), `writer.py` (x264/mp4v, wall-clock pacing, offline overlay), `pipeline.py` (orchestrator, warm-up, race-free shutdown). Validated locally (file-as-stream → valid annotated MP4, clean finalize). **Real-time, no recording — matches the handoff.**
- **Ops:** `deploy.sh` (rsync alt to git), README §2b server steps (may have been reverted — re-add if missing), requirements torch note.

### src/live/ file map
`capabilities.py` device/NVDEC probe · `frame.py` Frame · `decode_backend.py` decode ABC (CPU impl; NVDEC stub) · `queues.py` NewestSlot + DropOldestQueue · `capture.py` CaptureThread · `scheduler.py` BatchScheduler · `inference.py` InferenceStage · `identity_stage.py` IdentityStage · `render.py` RenderStage · `writer.py` WriterStage · `pipeline.py` LivePipeline.

---

## 5. What's NEXT (future plans, in order)

- **Stage 3 — full identity engine (NEXT; fixes the reids).** In-memory active-set (prototypes/medoids) + **cold reactivation** (fixes *same person → different ids*) + **cross-camera reciprocal-best matching** (fixes *no cross-camera persistence*) + anti-starvation priority queue + pluggable **topology veto** (fail-open stub until adjacency/transit data arrives). Ports the proven policies: same-camera time-overlap HARD veto, thresholds, runner-up margin, mint-when-uncertain. In-memory → **no Qdrant required**. Validated locally on synthetic embeddings (re-acquire / cross-camera link / no false-merge), then GPU-validated on the server.
- **Stage 2 — GPU throughput.** Fair scheduler (fairness + starvation bound) + **batched detector + ReID across cameras, one ByteTrack per camera** (obey the stop-rule). First GPU throughput milestone.
- **Stage 4 — GPU decode (server task).** NVDEC backend. `cv2` here has **no `cudacodec`** and no decord/PyNvVideoCodec, so GPU-resident decode needs a server decode lib (OpenCV-CUDA build / PyNvVideoCodec / torchaudio-CUDA / decord). Until then: CPU decode → 1 upload → GPU inference → 1 download.
- **Stage 5 — ops hardening.** Metrics (per-cam fps, latency, queue depths, drops, **batch size/utilisation**, peaks), warm-up ordering, CUDA OOM/context handling, disk/codec degradation.
- **Stage 6 — acceptance + self-contained server scripts.** `scripts/run_acceptance.py`, `run_gpu_benchmark.py` (4-cam P95 latency, drop %), `run_24h_memory.py`; explicit **multi-N** test.

**Validation cadence:** iterate + logic-test **locally** each stage (deterministic, synthetic — no GPU/video); ship to GPU (git pull) for **milestone** checks: throughput after Stage 2, **id-quality after Stage 3**.

---

## 6. Environment, deploy, run

- **Local dev:** CPU-only WSL2 (no NVIDIA GPU; `torch.cuda.is_available()==False`). `torch 2.12.1+cu130` is a CUDA build that falls back to CPU. GUI needs `libsm6 libice6 libxext6 libxrender1 libgl1` (installed) for OpenCV windows.
- **Server:** A100, reached via **SSH**. Deploy = **`git pull`** (not FileZilla). One-time: `git clone … && git checkout research && python3 -m venv .venv && pip install -r requirements.txt`. GPU auto-used (`device: auto`). `deploy.sh` (rsync) is an alt (set `DEPLOY_TARGET` in gitignored `.deploy.env`).
- **Run live (the target):**
  ```bash
  python main.py --mode live --videos "rtsp://USER:PASS@HOST:554/path" [more URLs...]
  ```
  Headless; writes `output_<cam>.mp4`; Ctrl-C to stop. **No Qdrant needed** for live (identity in-memory). N cameras = more `--videos` URLs.
- **RTSP creds:** inline in the URL, URL-encode specials (`@`→`%40`). Never commit; rotate if leaked.

## 7. Gotchas (don't re-learn the hard way)

- **`--mode live` ≠ record-then-batch.** Live = real-time, no recording, no reconcile. Record-then-batch is forbidden by the handoff (saves raw stream) — don't reach for it.
- **Never judge quality on CPU** — it drops most frames (drop-stale) and starves identity. GPU only.
- **Stage-1 live reids are supposed to be rough** — the identity engine is Stage 3.
- **Two identity systems coexist by design:** batch `reconcile_tracklets` (file path, untouched) vs the fresh live active-set (`src/live/identity_stage.py`).
- **Do not modify** `process_video` / `reconcile_tracklets` / `render_final_videos` / `IdentityService` (file-batch regression gate).
- **Topology data still missing** — needed to activate the cross-camera physical-impossibility veto; until then it's fail-open.

## 8. Pointers
- Pipeline: `src/live/*.py`. Routing + modes: `main.py` (`run.mode`, `--mode`, `run_capture_phase`, `render_final_videos`). Config: `config.yaml` (`live:` block + `identity`/`reid`). Reused: `src/detector.py`, `src/reid/extractor.py`, `src/reid/service.py`, `src/database/store.py`, `src/drawing.py`, `src/video_source.py`.
- Full staged plan (local, not in repo): `~/.claude/plans/functional-kindling-church.md`.
