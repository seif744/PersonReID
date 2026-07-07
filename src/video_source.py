"""
video_source.py  --  STAGE 1 -> 2 of the pipeline:  VIDEO FILE  ->  frames.

============================ THE BIG PICTURE ================================
A video FILE (like sample.mp4) is not a bunch of images -- it's a single,
compressed blob. To feed it to a neural network we must DECODE it into
individual still images called FRAMES. A video is just many frames shown
quickly (e.g. 25 frames per second = 25 FPS).

This file's ONE job: open an mp4 (or .mov/.avi) file and hand out its frames
one at a time, then stop cleanly when the file ends.

We use OpenCV (`cv2`) for this. Each frame it gives us is a grid of pixels:
a NumPy array of shape (height, width, 3) -- the 3 being the Blue, Green, Red
colour channels. (OpenCV uses B-G-R order, not R-G-B. A historical quirk.)

(No RTSP / camera / reconnect logic here on purpose -- we're reading files.)
============================================================================
"""

import os
import cv2  # OpenCV: the library that decodes video and gives us frames.


class VideoSource:
    """
    Wraps an OpenCV video capture for a LOCAL VIDEO FILE, so the rest of the
    program never touches OpenCV directly -- it just asks for frames.

    Usage:
        with VideoSource("sample.mp4") as cam:
            for frame in cam.frames():
                ...do something with `frame`...
    """

    def __init__(self, path):
        # `path` is the video file to read, e.g. "sample.mp4".
        # It comes from config.yaml -> source.url.
        self.path = path

        # `capture` will hold OpenCV's open-file object once we open it.
        # None means "not opened yet".
        self.capture = None

    def open(self):
        """
        Open the video file and get it ready to read frames from.
        """
        # Fail early with a clear message if the file isn't actually there --
        # otherwise OpenCV fails silently and you get a confusing empty video.
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Video file not found: {self.path!r}. "
                f"Put the file next to main.py, or set the correct path in "
                f"config.yaml under source.url."
            )

        # This is the line that actually opens & prepares the file for decoding.
        self.capture = cv2.VideoCapture(self.path)

        # VideoCapture doesn't raise on a bad/corrupt file; it just isn't
        # "opened". So we check explicitly and complain if it failed.
        if not self.capture.isOpened():
            raise IOError(
                f"Could not open video file: {self.path!r}. "
                f"The file may be corrupt or in an unsupported format."
            )

        return self

    def frames(self):
        """
        A GENERATOR that yields frames one at a time.

        A "generator" produces values lazily: each time the caller's for-loop
        asks for the next item, execution resumes here, grabs ONE frame, and
        hands it back. This means we never load the whole video into memory --
        we process it frame-by-frame.

        Each yielded `frame` is a NumPy array of shape (height, width, 3).
        """
        if self.capture is None:
            self.open()

        while True:
            # .read() returns two things:
            #   ok    -> a boolean: did we get a frame?
            #   frame -> the actual image (or None if ok is False).
            ok, frame = self.capture.read()

            # For a FILE, ok == False means we've reached the END of the video.
            if not ok:
                print("[video_source] End of video file.")
                break

            # Success: hand this single frame to the caller, then pause here
            # until the caller asks for the next one.
            yield frame

    def release(self):
        """Close the file and free the handle. Safe to call more than once."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # ---- These two make `with VideoSource(...) as cam:` work. ----
    # A "context manager" guarantees release() runs when we leave the `with`
    # block -- even if the code inside it crashes -- so the file is always
    # closed properly.
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
