"""
video_source.py  --  STAGE 1 -> 2 of the pipeline:  VIDEO FILE / STREAM -> frames.

============================ THE BIG PICTURE ================================
A video FILE (like sample.mp4) is not a bunch of images -- it's a single,
compressed blob. To feed it to a neural network we must DECODE it into
individual still images called FRAMES. A video is just many frames shown
quickly (e.g. 25 frames per second = 25 FPS).

This file's ONE job: open a source (an mp4/.mov/.avi FILE, or an rtsp://
/http:// live STREAM) and hand out its frames one at a time, then stop cleanly.

We use OpenCV (`cv2`) for this. Each frame it gives us is a grid of pixels:
a NumPy array of shape (height, width, 3) -- the 3 being the Blue, Green, Red
colour channels. (OpenCV uses B-G-R order, not R-G-B. A historical quirk.)

------------------------------ FILES vs STREAMS ----------------------------
A FILE is finite and re-readable: a failed read means end-of-file, and we can
re-open it from the start later (the final re-render pass does exactly that).

A live STREAM (rtsp://, http://, ...) is neither. It has no "start" to seek to
and no natural "end"; a failed read is usually a transient network hiccup, not
the end. It also produces frames on its own clock -- if we process slower than
it emits, frames pile up in OpenCV's internal buffer and we fall further and
further BEHIND real time. So streams get three extra behaviours, all OFF for
files (which keep their original byte-for-byte behaviour):

  1. URL sources skip the on-disk existence check (there is no file to stat).
  2. A failed read triggers a bounded RECONNECT (re-open the capture) with
     backoff, instead of ending the stream on the first blip.
  3. DROP-STALE: we keep only the most RECENT frame each step (grab-and-drop
     the backlog) so latency stays bounded when N cameras outrun the CPU. The
     tradeoff is dropped frames under load, never growing lag.
============================================================================
"""

import os
import time

import cv2  # OpenCV: the library that decodes video and gives us frames.


def is_stream_path(path):
    """True for live-stream URLs (rtsp://, http://, ...) vs local file paths."""
    return isinstance(path, str) and "://" in path


class VideoSource:
    """
    Wraps an OpenCV video capture for a local FILE or a live STREAM, so the
    rest of the program never touches OpenCV directly -- it just asks for
    frames.

    Usage:
        with VideoSource("sample.mp4") as cam:
            for frame in cam.frames():
                ...do something with `frame`...

        with VideoSource("rtsp://cam/stream", stop_event=ev) as cam:
            for frame in cam.frames():
                ...

    stop_event : optional threading.Event. When set, a stream stops waiting on
                 reconnects and `frames()` returns cleanly. Ignored for files.
    reconnect_attempts / reconnect_backoff / drop_stale : stream-only tuning
                 (see module docstring). Defaults are sane; overridden from
                 config.source.stream by main.py.
    """

    def __init__(self, path, stop_event=None, reconnect_attempts=5,
                 reconnect_backoff=1.0, drop_stale=True):
        # `path` is the source to read: a file path e.g. "sample.mp4", or a
        # stream URL e.g. "rtsp://...". It comes from config.yaml / --videos.
        self.path = path
        self.is_stream = is_stream_path(path)

        # Stream-only knobs (harmless/unused for files).
        self.stop_event = stop_event
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.reconnect_backoff = max(0.0, float(reconnect_backoff))
        self.drop_stale = bool(drop_stale)

        # `capture` will hold OpenCV's open-source object once we open it.
        # None means "not opened yet".
        self.capture = None

    def _stopped(self):
        return self.stop_event is not None and self.stop_event.is_set()

    def open(self):
        """
        Open the source and get it ready to read frames from.
        """
        # Fail early with a clear message if a FILE isn't actually there --
        # otherwise OpenCV fails silently and you get a confusing empty video.
        # A STREAM has no on-disk path to stat, so we skip this check for URLs
        # and let the connection attempt below be the thing that succeeds/fails.
        if not self.is_stream and not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Video file not found: {self.path!r}. "
                f"Put the file next to main.py, or set the correct path in "
                f"config.yaml under source.videos."
            )

        # This is the line that actually opens & prepares the source for decoding.
        self.capture = cv2.VideoCapture(self.path)

        # For a live stream, keep OpenCV's internal buffer tiny so a slow
        # consumer reads the freshest frame instead of a growing backlog. This
        # is a best-effort hint (not all backends honour it); the drop-stale
        # grab loop in frames() is the real latency guard.
        if self.is_stream:
            try:
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        # VideoCapture doesn't raise on a bad/corrupt file or an unreachable
        # stream; it just isn't "opened". So we check explicitly and complain.
        if not self.capture.isOpened():
            what = "stream" if self.is_stream else "video file"
            raise IOError(
                f"Could not open {what}: {self.path!r}. "
                + ("The URL may be unreachable, wrong, or need credentials."
                   if self.is_stream else
                   "The file may be corrupt or in an unsupported format.")
            )

        return self

    def _reopen(self):
        """Release and re-open a stream capture (used for reconnect)."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.capture = cv2.VideoCapture(self.path)
        if self.is_stream:
            try:
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return self.capture.isOpened()

    def frames(self):
        """
        A GENERATOR that yields frames one at a time.

        A "generator" produces values lazily: each time the caller's for-loop
        asks for the next item, execution resumes here, grabs ONE frame, and
        hands it back. This means we never load the whole video into memory --
        we process it frame-by-frame.

        Each yielded `frame` is a NumPy array of shape (height, width, 3).

        FILES: identical to the original behaviour -- read frames in order,
        stop on the first failed read (end of file).

        STREAMS: keep only the most recent frame (drop-stale) so we never lag,
        and treat a failed read as a transient hiccup -> bounded reconnect,
        rather than the end of the stream. A set stop_event ends it cleanly.
        """
        if self.capture is None:
            self.open()

        if not self.is_stream:
            # ---- FILE PATH: unchanged from the original implementation. ----
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    print("[video_source] End of video file.")
                    break
                yield frame
            return

        # ---- STREAM PATH ----
        consecutive_failures = 0
        while True:
            if self._stopped():
                print(f"[video_source] Stop requested; closing stream {self.path!r}.")
                break

            frame = self._read_latest()
            if frame is None:
                # Transient failure: attempt a bounded reconnect. reconnect_attempts
                # counts CONSECUTIVE failures before we give up on the stream.
                consecutive_failures += 1
                if consecutive_failures > self.reconnect_attempts:
                    print(f"[video_source] Stream {self.path!r} unavailable after "
                          f"{self.reconnect_attempts} reconnect attempts; stopping.")
                    break
                backoff = self.reconnect_backoff * consecutive_failures
                print(f"[video_source] Stream read failed "
                      f"({consecutive_failures}/{self.reconnect_attempts}); "
                      f"reconnecting in {backoff:.1f}s...")
                # Sleep in short slices so a stop_event is honoured promptly.
                slept = 0.0
                while slept < backoff and not self._stopped():
                    time.sleep(min(0.2, backoff - slept))
                    slept += 0.2
                if self._stopped():
                    break
                self._reopen()
                continue

            consecutive_failures = 0
            yield frame

    def _read_latest(self):
        """
        Read the FRESHEST available frame from a stream, dropping any backlog.

        OpenCV buffers decoded frames; if we call read() once per processed
        frame while processing slower than the stream's fps, we consume stale
        frames and lag grows without bound. So we grab() as many queued frames
        as are immediately available and retrieve() only the last one. Returns
        the frame, or None on a read failure (caller handles reconnect).
        """
        if not self.drop_stale:
            ok, frame = self.capture.read()
            return frame if ok else None

        # Grab (decode-and-discard) up to a small cap of queued frames quickly,
        # then retrieve the most recent. grab() returns False when no frame is
        # ready; the cap stops us spinning forever on a fast source.
        grabbed_any = False
        for _ in range(5):
            if not self.capture.grab():
                break
            grabbed_any = True
        if not grabbed_any:
            # Nothing buffered was grabbable -- fall back to a blocking read so
            # we still wait for the next frame rather than busy-spin.
            ok, frame = self.capture.read()
            return frame if ok else None
        ok, frame = self.capture.retrieve()
        return frame if ok else None

    def release(self):
        """Close the source and free the handle. Safe to call more than once."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # ---- These two make `with VideoSource(...) as cam:` work. ----
    # A "context manager" guarantees release() runs when we leave the `with`
    # block -- even if the code inside it crashes -- so the source is always
    # closed properly.
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
