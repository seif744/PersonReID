"""
src/live/ -- the v5 real-time RTSP streaming pipeline (staged build).

This package is the LIVE path: N RTSP streams -> per-camera batched detection/
ReID -> active-set identity -> live-annotated output_<cam>.mp4, targeting a GPU
server but device-portable to CPU. It is entirely separate from, and does not
touch, the file-batch flow in main.py (process_video / reconcile_tracklets /
render_final_videos), which stays intact (regression gate).

See the plan/architecture doc for the staged design and the FIXED execution
rules (priority order, no architectural drift, ByteTrack stop-rule).

Built so far: Stage 0 (capabilities, Frame carrier, decode-backend abstraction --
CPU done, NVDEC a documented server task), Stage 1 (the full threaded skeleton:
capture/scheduler/inference/identity/render/writer/pipeline), and Stage 3 (the
full in-memory identity engine -- active set with prototype/medoid storage,
same-camera cold reactivation, cross-camera reciprocal-best matching, an
anti-starvation per-camera fair queue, and a pluggable fail-open topology veto;
identity_engine.py + priority.py + topology.py, driven by identity_stage.py, with
NO Qdrant). Still to come: Stage 2 GPU batched detector+ReID throughput, Stage 4
NVDEC decode, Stage 5 ops/metrics hardening, Stage 6 acceptance scripts.

The Stage-3 engine is thread-free and deterministic; its logic is validated on
synthetic embeddings in tests/live/ (run tests/live/run_stage3_acceptance.py).
"""
