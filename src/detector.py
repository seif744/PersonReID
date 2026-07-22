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
    # Identity Service; stays None when ReID is disabled or not yet assigned.
    global_id: Optional[int] = None

    # User-facing alias for the cross-camera identity. This is kept in sync with
    # global_id so downstream code can talk about a "reid id" without changing
    # the underlying identity pipeline.
    reid_id: Optional[int] = None

    # The ReID APPEARANCE embedding (512-d, L2-normalized) for this detection,
    # attached by TrackEmbedder (Stage 5/6). This is the raw "what they look
    # like" vector; it carries NO identity by itself -- turning embeddings into
    # a global_id is the Identity Service's job (Stage 7). Stays None until a
    # track is embedded (and, under throttling, reuses the track's last vector).
    embedding: Optional[np.ndarray] = None

    # Optional metadata from the ReID crop-quality gate. It is not used by the
    # detector itself, but keeping it on the Detection makes debugging bad crops
    # and gallery inserts straightforward.
    crop_quality: Optional[dict] = None

    # Independent-of-the-CNN appearance evidence, attached by TrackEmbedder
    # alongside the embedding (Stage 5/6). These are MEASUREMENTS, not identity:
    #   color_descriptor -- a torso colour histogram (see reid/color.py); an
    #     appearance signal that survives when pose fools the embedding.
    #   normalized_height -- box height / frame height, a coarse size cue.
    # Both feed the Track Evidence Layer (identity/evidence.py) so identity
    # reasoning has evidence beyond a single cosine score. None until embedded.
    color_descriptor: Optional[np.ndarray] = None
    normalized_height: Optional[float] = None


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
                 tracker_config="bytetrack.yaml", pose_ensemble=None):
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

        # ---- Pose ensemble: split boxes that merged two+ people ----
        # The box detector/tracker sometimes wraps ONE box around two overlapping
        # people (the crops/id_0008 case: two stacked bodies). Embedding that crop
        # blends two identities. To catch it, we run a second, independent model
        # -- a POSE model that finds one skeleton PER PERSON -- and, when a tracker
        # box clearly contains >= 2 distinct pose bodies, we split it into the
        # individual pose boxes. This is an ENSEMBLE: two models agreeing on how
        # many people are really there. Off unless configured.
        self.pose_cfg = pose_ensemble or {}
        self.pose_model = None
        if self.pose_cfg.get("enabled"):
            pose_path = self.pose_cfg.get("pose_model", "yolo11n-pose.pt")
            # Fraction of a pose body-box that must lie inside a tracker box to
            # count as "contained in it". Body containment (not a loose face
            # buffer) is what keeps a neighbour from triggering a false split.
            self.min_containment = float(self.pose_cfg.get("min_containment", 0.6))
            self.pose_conf = float(self.pose_cfg.get("conf", 0.25))
            print(f"[detector] Loading pose ensemble model '{pose_path}'...")
            self.pose_model = YOLO(pose_path)

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
        detections = self._parse_results(results)
        # Ensemble post-step: split any tracker box that merged two+ people.
        if self.pose_model is not None:
            detections = self._split_merged_boxes(frame, detections)
        return detections

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

    # ======================= POSE ENSEMBLE (Stage 3.5) =======================
    # These run only when pose_ensemble is enabled. They take the tracker's
    # boxes and, using an independent pose model's per-person boxes, split any
    # box that clearly contains two+ distinct people into one box per person.

    def _split_merged_boxes(self, frame, detections):
        """
        Replace any tracker box that contains >= 2 distinct pose bodies with the
        individual pose boxes. Boxes with 0-1 contained bodies pass through
        unchanged, so this only ever *fixes* merges -- it never drops a person.
        """
        pose_boxes = self._pose_bodies(frame)
        if not pose_boxes:
            return detections

        out = []
        for det in detections:
            tbox = (det.x1, det.y1, det.x2, det.y2)
            # Pose bodies that sit mostly inside this tracker box.
            contained = [p for p in pose_boxes
                         if self._containment(p, tbox) >= self.min_containment]
            contained = self._dedup_boxes(contained)   # drop double-detections

            if len(contained) >= 2:
                print(f"[detector] ensemble: track {det.track_id} contains "
                      f"{len(contained)} people -> splitting")
                for i, pbox in enumerate(contained):
                    out.append(Detection(
                        x1=int(round(pbox[0])), y1=int(round(pbox[1])),
                        x2=int(round(pbox[2])), y2=int(round(pbox[3])),
                        confidence=det.confidence, class_id=det.class_id,
                        track_id=self._split_track_id(det.track_id, i),
                    ))
            else:
                out.append(det)
        return out

    def _pose_bodies(self, frame):
        """Run the pose model; return one (x1,y1,x2,y2) body box per person."""
        results = self.pose_model(frame, conf=self.pose_conf, iou=0.65,
                                  verbose=False)
        boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                boxes.append(tuple(box.xyxy[0].tolist()))
        return boxes

    @staticmethod
    def _containment(inner, outer):
        """Fraction of `inner` box's area that lies inside `outer`."""
        ix1 = max(inner[0], outer[0])
        iy1 = max(inner[1], outer[1])
        ix2 = min(inner[2], outer[2])
        iy2 = min(inner[3], outer[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return inter / inner_area

    def _dedup_boxes(self, boxes, iou_thresh=0.7):
        """Collapse near-duplicate boxes (the pose model detecting one person twice)."""
        kept = []
        for b in boxes:
            if all(self._iou(b, k) < iou_thresh for k in kept):
                kept.append(b)
        return kept

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _split_track_id(track_id, i):
        """
        Primary person in a split keeps the real track_id; extras get a derived
        id. These derived ids are per-frame (the pose split is recomputed each
        frame), so downstream they read as short tracks -- the ReID noise filter
        and reconciler are what stitch/clean them.
        """
        if track_id is None:
            return None
        return track_id if i == 0 else track_id + 100000 * (i + 1)
