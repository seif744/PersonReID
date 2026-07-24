"""
src/live/ -- the v5 real-time RTSP streaming pipeline (staged build).

This package is the LIVE path: N RTSP streams -> per-camera batched detection/
ReID -> active-set identity -> live-annotated output_<cam>.mp4, targeting a GPU
server but device-portable to CPU. It is entirely separate from, and does not
touch, the file-batch flow in main.py (process_video / reconcile_tracklets /
render_final_videos), which stays intact (regression gate).

See the plan/architecture doc for the staged design and the FIXED execution
rules (priority order, no architectural drift, ByteTrack stop-rule).

Stage 0 provides the foundation only: capability detection, the Frame carrier,
and the decode-backend abstraction (CPU implemented; NVDEC is a documented
server task). Later stages add capture/scheduler/inference/identity/render/
writer/pipeline.
"""
