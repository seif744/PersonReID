"""
frame.py  --  the Frame carrier that travels through every live stage.

One object per captured frame, owned by exactly one stage at a time (capture ->
slot -> scheduler -> inference -> identity -> render -> writer). It carries the
pixels plus the metadata every downstream stage needs.

CRITICAL RULE (v5 plan §9): `ts` is the wall-clock time.time() stamped ONCE at
frame read, and is carried UNCHANGED to the payload/logs. Never regenerate a
timestamp later (after inference, at render, ...) or cross-camera temporal logic
(freshness, topology) silently skews. All cameras share this one machine clock,
which is exactly what makes cross-camera time comparable.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Frame:
    cam: str                 # camera / source name (also the output_<cam>.mp4 label)
    ts: float                # wall-clock time.time() at READ -- never regenerated
    frame_index: int         # per-camera monotonic counter (continues across reconnect)
    image: Any = None        # np.ndarray (CPU, BGR) or torch.Tensor (GPU); device below
    device: str = "cpu"      # "cpu" | "cuda:N" -- where `image` currently lives
    # Filled in by later stages; kept here so the frame is the single carrier.
    detections: Any = None   # per-stage results attached downstream
    meta: Dict[str, Any] = field(default_factory=dict)

    def age_ms(self, now: float) -> float:
        """Milliseconds since capture, measured against a wall-clock `now`
        (time.time()). Used by the scheduler's freshness check."""
        return (now - self.ts) * 1000.0
