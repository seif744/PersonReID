"""
drawing.py  --  STAGE 5 of the pipeline:  draw bbox + label onto the frame.

============================ THE BIG PICTURE ================================
Detection gives us NUMBERS (box coordinates + confidence). Numbers are hard
to eyeball. So we draw those boxes back ONTO the image, plus a little text
label, and show it in a window. This is purely for US, the humans, to confirm
"yes, it's finding the people correctly". The neural network doesn't need
this; our eyes do. This is called VISUALIZATION or an "annotated frame".

Everything here is done with OpenCV's simple drawing functions, which paint
directly onto the pixel array (they modify the image in place).

Reminder: OpenCV colours are (Blue, Green, Red), each 0-255. So:
  (0, 255, 0)   = pure green
  (0, 0, 255)   = pure red
  (255, 0, 0)   = pure blue
============================================================================
"""

import zlib

import cv2

# The colour we draw when there is NO track id yet (plain detection). Green.
BOX_COLOR = (0, 255, 0)          # BGR = green
TEXT_COLOR = (0, 0, 0)           # BGR = black (drawn on the coloured label patch)
BOX_THICKNESS = 3                # line thickness in pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX  # a built-in OpenCV font
LABEL_FONT_SCALE = 1.0           # id/GID label size (was 0.5)
LABEL_THICKNESS = 2              # id/GID label stroke weight (was 1)

# A fixed palette of distinct, bright BGR colours. Each track id is mapped to
# one of these so the SAME person keeps the SAME colour frame to frame, which
# makes it easy to follow someone with your eyes.
_PALETTE = [
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 0, 255),     # red
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 165, 255),   # orange
    (128, 0, 255),   # pink/red
]


def color_for_id(track_id):
    """
    Return a stable colour for a given track id. We use the id modulo the
    palette length, so id 0 and id 8 share a colour but consecutive ids differ.
    If there's no id (plain detection), fall back to green.
    """
    if track_id is None:
        return BOX_COLOR
    if not isinstance(track_id, int):
        track_id = zlib.crc32(str(track_id).encode("utf-8"))
    return _PALETTE[int(track_id) % len(_PALETTE)]


def draw_detections(frame, detections):
    """
    Draw every detection onto `frame` and return the annotated frame.

    NOTE: OpenCV draws IN PLACE (it edits the array you pass in). We return it
    too, just so the calling code reads naturally.
    """
    for det in detections:
        # A NEGATIVE reid_id is a PROVISIONAL live label: an online-mode track
        # that is still gathering evidence and hasn't resolved to a real global
        # id yet (see IdentityService._assign_online). Treat it as "not yet
        # identified" for both colour and label, so the overlay shows a neutral
        # "pending" marker instead of a scary "REID -1" that then flips.
        provisional = det.reid_id is not None and det.reid_id < 0

        # Colour by REID id when we have one, so the same person keeps the same
        # colour ACROSS cameras (that's the whole point of ReID). Fall back to
        # the per-camera track id, then green. Provisional tracks colour by
        # track id (stable while pending) rather than the throwaway negative id.
        ident = (det.reid_id if (det.reid_id is not None and not provisional)
                 else det.global_id if det.global_id is not None
                 else det.track_id)
        color = color_for_id(ident)

        # 1) The rectangle. cv2.rectangle needs the two opposite corners:
        #    top-left (x1, y1) and bottom-right (x2, y2).
        cv2.rectangle(
            frame,
            (det.x1, det.y1),   # top-left corner
            (det.x2, det.y2),   # bottom-right corner
            color,
            BOX_THICKNESS,
        )

        # 2) The text label. Prefer the cross-camera REID id, else the legacy
        #    global id, else just the track id ("ID 3  0.87"), else "person".
        if provisional:
            # Still deciding who this is -- show a pending marker, not the id.
            label = (f"REID ...  ID{det.track_id}"
                     if det.track_id is not None else "REID ...")
        elif det.reid_id is not None:
            label = (f"REID {det.reid_id}  ID{det.track_id}"
                     if det.track_id is not None else f"REID {det.reid_id}")
        elif det.global_id is not None:
            label = (f"GID {det.global_id}  ID{det.track_id}"
                     if det.track_id is not None else f"GID {det.global_id}")
        elif det.track_id is not None:
            label = f"ID {det.track_id}  {det.confidence:.2f}"
        else:
            label = f"person {det.confidence:.2f}"

        # Measure how big that text will be, so we can draw a filled
        # background patch behind it -- otherwise the text can be unreadable.
        (text_w, text_h), baseline = cv2.getTextSize(
            label, FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)

        # Draw the filled label background just ABOVE the box's top-left corner.
        cv2.rectangle(
            frame,
            (det.x1, det.y1 - text_h - baseline - 6),  # top-left of patch
            (det.x1 + text_w + 4, det.y1),             # bottom-right of patch
            color,
            thickness=-1,   # -1 means "fill the rectangle solid"
        )

        # Draw the text on top of that patch.
        cv2.putText(
            frame,
            label,
            (det.x1 + 2, det.y1 - baseline - 3),  # bottom-left anchor of the text
            FONT,
            LABEL_FONT_SCALE,
            TEXT_COLOR,
            LABEL_THICKNESS,
            cv2.LINE_AA,     # anti-aliased = smoother-looking text
        )

    return frame


def draw_hud(frame, person_count, fps=None):
    """
    Draw a small "heads-up display" in the top-left: how many people are in
    view, and (optionally) how fast we're processing, in frames per second.
    Handy for confirming the pipeline is keeping up in real time.
    """
    text = f"People: {person_count}"
    if fps is not None:
        text += f"   FPS: {fps:.1f}"

    cv2.putText(frame, text, (10, 30), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
