"""
demo_detect_reid.py  --  PHASE 5:  YOLO detection  ->  crop  ->  embedding.

============================ WHAT THIS SHOWS ==============================
The wiring between two modules that must stay independent:

    frame --(PersonDetector.detect)--> boxes
    box   --(clamp + slice)----------> BGR crop
    crops --(ReIDExtractor.extract_batch)--> (N, 512) embeddings

There is NO tracker here (Phase 6) and NO identity logic (Phase 7). We only
prove that a real detector's boxes flow cleanly into the extractor and produce
one unit-norm embedding per detected person.

Run from src/:   python -m reid.demo_detect_reid

------------------------------- CROPPING ----------------------------------
YOLO boxes can extend past the frame edge (a person standing at the border).
Slicing with out-of-range / negative indices yields an empty or wrong crop, so
every box is CLAMPED to [0, w] x [0, h] and degenerate boxes are skipped. That
"box -> safe crop" logic is the shared detector.crop_person primitive (Phase 6
consolidated it out of here and crop_saver into one place).

padding=0 (tight box) on purpose: OSNet/Market-1501 was trained on tight person
crops; adding margin injects background and drifts off the training distribution.
============================================================================
"""

import cv2
import numpy as np

from detector import PersonDetector, crop_person
from reid.extractor import ReIDExtractor

VIDEO = "../output_cam_219.mp4"      # real footage; path is relative to src/
FRAMES_TO_SAMPLE = 5                  # a few spaced-out frames, enough to prove flow
FRAME_STRIDE = 20                     # skip frames so we don't sample near-duplicates


def main():
    detector = PersonDetector(model_path="../yolo11n.pt")
    extractor = ReIDExtractor(weights="reid/weights/osnet_x1_0_market1501.pth")

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    frames_done = 0
    frame_index = -1
    while frames_done < FRAMES_TO_SAMPLE:
        ok, frame = cap.read()
        if not ok:
            break                         # video ended
        frame_index += 1
        if frame_index % FRAME_STRIDE != 0:
            continue

        # 1. DETECT (no tracking -> track_id is None on every Detection).
        detections = detector.detect(frame)

        # 2. BOX -> CROP. Keep detections and crops aligned; drop degenerate boxes.
        crops, kept = [], []
        for det in detections:
            crop = crop_person(frame, det)
            if crop is not None:
                crops.append(crop)
                kept.append(det)

        # 3. CROPS -> EMBEDDINGS, one batched forward pass for the whole frame.
        embeddings = extractor.extract_batch(crops)   # (N, 512)

        # Report what happened this frame.
        print(f"frame {frame_index:4d}: {len(detections)} detections, "
              f"{len(crops)} valid crops -> embeddings {embeddings.shape}")
        for det, emb in zip(kept, embeddings):
            box_wh = (det.x2 - det.x1, det.y2 - det.y1)
            print(f"    conf={det.confidence:.2f}  box(wh)={box_wh}  "
                  f"||emb||={np.linalg.norm(emb):.4f}")

        frames_done += 1

    cap.release()

    # A tiny cross-check: within one frame, two DIFFERENT people should NOT be
    # near-identical. We recompute on the last frame that had >=2 people and show
    # the max off-diagonal cosine -- a quick "the flow produces sane vectors" check.
    if len(crops) >= 2:
        S = embeddings @ embeddings.T
        np.fill_diagonal(S, -1.0)
        print(f"\nlast multi-person frame: max cross-person cosine = {S.max():.3f} "
              f"(expected < ~0.85; identical would signal a wiring bug)")


if __name__ == "__main__":
    main()
