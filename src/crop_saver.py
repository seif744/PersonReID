"""
crop_saver.py  --  saves each tracked person's bounding-box image to disk.

============================ THE BIG PICTURE ================================
Once every person has a stable track_id (from Stage 4), we can CROP the pixels
inside their bounding box and save that little image ("a crop") to a folder.

Why do this? Two reasons:
  1. Debugging: you can open the folder and literally see who track #3 was.
  2. Stage 6 prep: the embedding model turns these crops into "fingerprints"
     (vectors) used to recognise the same person across cameras/angles. Having
     many crops per person, across frames and later across camera angles, is
     exactly what makes that matching robust.

Layout on disk (one sub-folder per person):
    crops/
      id_0001/
        f000012.jpg      <- frame 12
        f000027.jpg      <- frame 27
        ...
      id_0002/
        ...

To avoid saving thousands of near-identical images, we only save a crop for a
given person every N frames (the "interval").
============================================================================
"""

import os
import cv2

from detector import crop_person   # the one shared "box -> safe crop" primitive


class CropSaver:
    """
    Saves bbox crops of tracked people into per-ID folders, throttled so we
    don't write a crop every single frame.
    """

    def __init__(self, output_dir="crops", interval=10, padding=0):
        # Folder to write everything under, e.g. "crops".
        #self.output_dir = output_dir

        # Save a crop for a given track only once every `interval` frames.
        # interval=1 saves every frame; interval=10 saves ~one every 10 frames.
        self.interval = max(1, interval)

        # Optionally grow each box by `padding` pixels on every side, so the
        # crop includes a little margin around the person. 0 = tight box.
        self.padding = padding

        # Remembers the last frame index at which we saved each track_id, so we
        # can enforce the interval per-person. { track_id: last_frame_saved }
        self._last_saved = {}

        # Make the top-level folder now (exist_ok = don't error if it's there).
        # os.makedirs(self.output_dir, exist_ok=True)

    def save(self, frame, detections, frame_index):
        """
        For each tracked detection, crop it out of `frame` and save it if enough
        frames have passed since we last saved that person.

        IMPORTANT: pass the CLEAN frame here (before boxes are drawn on it),
        otherwise the saved crops would have green rectangles painted through
        them. main.py calls this BEFORE the drawing step for exactly that reason.

        `frame_index` is the running frame counter (0, 1, 2, ...).
        Returns how many crops were written this call (handy for logging).
        """
        #saved_count = 0
        extracted_crops=[]

        for det in detections:
            # No ID yet -> we can't file it under a person folder. Skip.
            if det.track_id is None:
                continue

            # Throttle: only save this person again after `interval` frames.
            last = self._last_saved.get(det.track_id)
            if last is not None and (frame_index - last) < self.interval:
                continue

            # Clamp + slice via the shared primitive; None = degenerate box.
            crop = crop_person(frame, det, padding=self.padding)
            if crop is None:
                continue

            # Build the per-person folder path, e.g. "crops/id_0003".
            # :04d zero-pads the id to 4 digits so folders sort nicely.
            #person_dir = os.path.join(self.output_dir, f"id_{det.track_id:04d}")
            #os.makedirs(person_dir, exist_ok=True)

            # File name includes the frame number so crops are ordered in time.
            #filename = f"f{frame_index:06d}.jpg"
            #path = os.path.join(person_dir, filename)

            # Write the image to disk.
            #cv2.imwrite(path, crop)

            # Record that we saved this person on this frame.
            #self._last_saved[det.track_id] = frame_index
            #saved_count += 1
            extracted_crops.append({
                "track_id": det.track_id,
                "crop": crop
            })
            self._last_saved[det.track_id] = frame_index

        return extracted_crops
