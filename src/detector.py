"""
detector.py  --  STAGE 3 + STAGE 4:  detection  AND  tracking.

============================ THE BIG PICTURE ================================
STAGE 3 (detection) answers: "WHERE are the people in THIS frame?"
STAGE 4 (tracking)  answers: "which person is which, ACROSS frames?"

Detection alone is forgetful: it looks at each frame from scratch, so the same
person becomes a brand-new box every frame. A TRACKER fixes that -- it links
boxes across consecutive frames and gives each person a stable TRACK ID
(person #1, person #2, ...) that stays with them while they're on screen.

We use YOLO for detection and Ultralytics' built-in tracker (ByteTrack) for
tracking. ByteTrack works by, each frame:
  1. take the new detections,
  2. predict where each existing track *should* be now,
  3. match new boxes to existing tracks by overlap (+ motion),
  4. keep IDs stable for matches, start new IDs for unmatched boxes, and
     retire IDs that have vanished for a while.
The nice part: Ultralytics does all of this for us inside model.track().

For each person we now return FOUR things: bbox, class id, confidence, and a
track_id (the persistent number). Everything is wrapped in a clean Detection
so the rest of the program never touches YOLO's internals.
============================================================================
"""

from dataclasses import dataclass
from typing import Optional      # Optional[int] means "an int OR None".

import numpy as np
from ultralytics import YOLO     # the library that loads & runs YOLO models.


@dataclass
class Detection:
    """
    One detected person in one frame.

    Coordinates are in PIXELS, measured from the top-left of the image:
      x grows to the right, y grows DOWNWARD (standard for images).
    """
    x1: int            # left edge of the box
    y1: int            # top edge of the box
    x2: int            # right edge of the box
    y2: int            # bottom edge of the box
    confidence: float  # how sure YOLO is (0.0 - 1.0)
    class_id: int      # which COCO class (always 0 = person here)

    # The persistent tracker ID. It's Optional because:
    #   - plain detect() doesn't track, so it stays None there;
    #   - even while tracking, a brand-new box may not have an ID yet for a
    #     frame or two until the tracker "confirms" it.
    track_id: Optional[int] = None

    # The GLOBAL person id from Stage 6+ (ReID). Unlike track_id (which is
    # per-camera and resets when the person leaves), this is meant to be the
    # SAME number for the same person across cameras and time. Filled in by the
    # ReIDService; stays None when ReID is disabled or not yet assigned.
    global_id: Optional[int] = None

    # The ReID APPEARANCE embedding (512-d, L2-normalized) for this detection,
    # attached by TrackEmbedder (Stage 5/6). This is the raw "what they look
    # like" vector; it carries NO identity by itself -- turning embeddings into
    # a global_id is the Identity Service's job (Stage 7). Stays None until a
    # track is embedded (and, under throttling, reuses the track's last vector).
    embedding: Optional[np.ndarray] = None


def crop_person(frame, det, padding=0):
    """
    box (Detection) -> BGR crop out of `frame`, or None if the box is degenerate
    after clamping to the frame bounds.

    This is the ONE shared "box -> safe crop" primitive. It lives here, next to
    Detection, because it's a pure detection-geometry operation with no ReID or
    disk-IO concern -- so both crop_saver.py and the ReID path depend on it
    rather than each keeping their own copy (which would drift over time).

    Clamping matters: YOLO boxes can extend past the frame edge (a person at the
    border), and slicing with out-of-range/negative indices yields an empty or
    wrong crop. max(...,0) stops negatives; min(...,w/h) stops overrun.

    padding grows the box by N pixels per side (0 = tight box). ReID wants tight
    boxes to match training crops; crop_saver may want a margin for debugging.
    """
    h, w = frame.shape[:2]
    x1 = max(det.x1 - padding, 0)
    y1 = max(det.y1 - padding, 0)
    x2 = min(det.x2 + padding, w)
    y2 = min(det.y2 + padding, h)
    if x2 <= x1 or y2 <= y1:
        return None
    # NumPy slicing frame[rows, cols] = frame[y1:y2, x1:x2]: rows are the y-axis.
    return frame[y1:y2, x1:x2]


class PersonDetector:
    """
    Loads the YOLO model once, then either DETECTS (per-frame) or TRACKS
    (across frames) people in each frame you give it.

    We load the model ONCE (in __init__) because loading is slow, but running
    it on a frame is comparatively fast. You never want to reload per frame.
    """

    def __init__(self, model_path="yolo11n.pt",
                 confidence_threshold=0.4, person_class_id=0,
                 tracker_config="bytetrack.yaml"):
        # Loading the weights. The first time this runs, Ultralytics will
        # DOWNLOAD the yolo11n.pt file (~5 MB) automatically, then reuse it.
        print(f"[detector] Loading model '{model_path}' (first run may download it)...")
        self.model = YOLO(model_path)

        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id

        # Which tracking algorithm to use. Ultralytics ships two ready-made
        # configs: "bytetrack.yaml" (fast, great default) and "botsort.yaml"
        # (slower, uses appearance too). We don't need to create these files;
        # they come with the library.
        self.tracker_config = tracker_config

        print("[detector] Model ready.")

    def detect(self, frame):
        """
        STAGE 3 only: run detection on ONE frame, NO tracking.
        Returns a list[Detection] whose track_id is always None.
        Kept for testing / when you don't need IDs.
        """
        results = self.model(
            frame,
            classes=[self.person_class_id],  # only people, ignore other 79 classes
            conf=self.confidence_threshold,  # drop low-confidence boxes
            verbose=False,                   # don't print a report every frame
        )
        return self._parse_results(results)

    def track(self, frame):
        """
        STAGE 3 + 4: detect people AND assign a stable track_id to each,
        linking them to the same people seen in previous frames.

        The important argument is persist=True: it tells Ultralytics "this frame
        is a continuation of the same video, so KEEP the tracker's memory from
        the last call." Without it, the tracker would reset every frame and IDs
        would never persist.
        """
        results = self.model.track(
            frame,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            tracker=self.tracker_config,     # which tracking algorithm
            persist=True,                    # remember tracks across calls
            verbose=False,
        )
        return self._parse_results(results)

    def _parse_results(self, results):
        """
        Shared helper: turn YOLO's raw output into our clean list[Detection].
        Used by BOTH detect() and track() so the extraction logic lives in one
        place. Reads a track_id if one is present (only during tracking).
        """
        detections = []

        # `results` is a list with one entry per input image. We pass one image,
        # so there's one entry, but we loop to be safe.
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # box.xyxy is a tensor shaped (1, 4): [[x1, y1, x2, y2]].
                # [0] takes that inner row, .tolist() turns it into plain numbers.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # box.id holds the track number DURING tracking. It's None for
                # plain detection, and can briefly be None even while tracking
                # (a box not yet confirmed as a track). Guard for both.
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                detections.append(
                    Detection(
                        # Round to whole pixels -- you can't index a pixel at 12.7.
                        x1=int(round(x1)),
                        y1=int(round(y1)),
                        x2=int(round(x2)),
                        y2=int(round(y2)),
                        confidence=confidence,
                        class_id=class_id,
                        track_id=track_id,
                    )
                )

        return detections
