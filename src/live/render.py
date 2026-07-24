"""
render.py  --  STAGE: draw the annotated frame (per camera).

Takes identity-stamped frames and draws boxes + reid labels, then hands the
annotated image to that camera's writer queue. The Render stage NEVER performs
inference (strict stage separation, v5 §2) -- it only draws, so it can run on
plain CPU threads independent of the (GPU) inference stage.

Device rule: if the frame's image is GPU-resident it is downloaded to CPU ONCE
here (the single GPU->CPU copy per frame) before drawing. In Stage 1 / CPU mode
frames are already CPU numpy arrays, so this is a no-op. Reuses the existing
`draw_detections`/`draw_hud` so overlays match the file pipeline exactly (incl.
the negative-provisional-id "REID ..." handling).
"""

import threading

from drawing import draw_detections, draw_hud


def _to_cpu_bgr(image, device):
    """Single GPU->CPU download point (no-op for CPU frames). Kept tiny and
    guarded so nothing imports torch unless a GPU frame actually arrives."""
    if device == "cpu" or image is None:
        return image
    try:
        import torch
        if isinstance(image, torch.Tensor):
            arr = image.detach().to("cpu").numpy()
            return arr
    except Exception:
        pass
    return image


class RenderStage(threading.Thread):
    def __init__(self, cam, render_queue, writer_queue, stop_event):
        super().__init__(name=f"render-{cam}", daemon=True)
        self.cam = cam
        self.in_q = render_queue
        self.writer_queue = writer_queue
        self.stop_event = stop_event
        self.rendered = 0

    def run(self):
        while not self.stop_event.is_set():
            frame = self.in_q.get(timeout=0.1)
            if frame is None:
                continue
            img = _to_cpu_bgr(frame.image, frame.device)
            if img is None:
                continue
            dets = frame.detections or []
            img = draw_detections(img, dets)
            img = draw_hud(img, person_count=len(dets))
            # Hand the annotated pixels + capture ts to the writer (pacing uses ts).
            self.writer_queue.put((img, frame.ts))
            self.rendered += 1
