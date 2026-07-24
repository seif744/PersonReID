"""
decode_backend.py  --  the decode ABSTRACTION (v5 plan pt.1).

The whole point of this file: **CPU decode is AN IMPLEMENTATION, not the
architecture.** Every capture path talks to the `DecodeBackend` interface, so
adding GPU (NVDEC) decode later is a new backend, not a rewrite of the pipeline.

  DecodeBackend (ABC)
    ├── CPUDecodeBackend   -- implemented now; FFmpeg via cv2/VideoSource,
    │                         frames are CPU numpy arrays (uploaded to GPU at
    │                         inference if needed). Works on CPU and GPU boxes.
    └── NVDECBackend       -- INTERFACE ONLY. GPU-resident decode with no
                              upload. A high-priority SERVER integration task
                              (Stage 4), not an optional add-on. Requires a CUDA
                              video-decode lib absent from this repo's deps.

Reconnect is a first-class backend operation so the capture thread never has to
know decode internals.
"""

import abc
import time

from video_source import VideoSource   # reuse tested stream-open + reconnect

# Flipped to True by Stage 4 once a real NVDEC backend is implemented AND a
# server decode library is available. Until then, backend selection resolves to
# CPU even when the hardware probe is positive.
NVDEC_IMPLEMENTED = False


class DecodeBackend(abc.ABC):
    """Decode one source into frames. Frames come with a wall-clock read `ts`.

    Contract:
      open()      -> self, ready to read (raises on a truly unopenable source).
      read()      -> (ok: bool, image, ts: float). ok False = transient failure
                     or end; the caller decides reconnect vs stop.
      reconnect() -> best-effort re-open of the same source (streams).
      release()   -> free the handle. Safe to call more than once.
    """

    @abc.abstractmethod
    def open(self): ...

    @abc.abstractmethod
    def read(self): ...

    @abc.abstractmethod
    def reconnect(self): ...

    @abc.abstractmethod
    def release(self): ...

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.release()


class CPUDecodeBackend(DecodeBackend):
    """
    CPU decode via OpenCV/FFmpeg, reusing VideoSource for stream open + reconnect.

    Frames are CPU BGR numpy arrays. On a GPU box the pipeline uploads them for
    inference (one upload) and downloads once for render -- the portable default
    until an NVDEC backend exists. `drop_stale=False`: this backend returns raw
    frames in order; freshness/dropping is the scheduler's job (v5), keeping the
    decode layer dumb.
    """

    def __init__(self, url, stop_event=None, reconnect_attempts=5,
                 reconnect_backoff=1.0):
        self.url = url
        self._src = VideoSource(
            path=url,
            stop_event=stop_event,
            reconnect_attempts=reconnect_attempts,
            reconnect_backoff=reconnect_backoff,
            drop_stale=False,
        )

    @property
    def is_stream(self):
        """True for rtsp/http URLs (reconnect on read failure); False for files
        (a read failure means end-of-file -> the camera is simply done)."""
        return self._src.is_stream

    def open(self):
        self._src.open()
        return self

    def read(self):
        cap = self._src.capture
        if cap is None:
            return False, None, time.time()
        ok, frame = cap.read()
        return ok, frame, time.time()   # ts stamped at read (v5 §9)

    def reconnect(self):
        # Reuse VideoSource's re-open; returns True if the capture is open again.
        return self._src._reopen()

    def release(self):
        self._src.release()


class NVDECBackend(DecodeBackend):
    """
    GPU (NVDEC) decode -> GPU-resident frames, NO host upload.

    NOT IMPLEMENTED -- SERVER INTEGRATION TASK (v5 plan pt.1, Stage 4).
    This repo's OpenCV has no `cv2.cudacodec`, and no decord/PyNvVideoCodec is
    installed, so GPU-resident decode is impossible with the current deps. On
    the A100, choose ONE decode library and implement this backend:
        - OpenCV built WITH CUDA (`cv2.cudacodec.VideoReader`), or
        - NVIDIA PyNvVideoCodec, or
        - torchaudio StreamReader (CUDA), or
        - decord (GPU).
    Then set NVDEC_IMPLEMENTED = True and have make_decode_backend() select it
    when the capability probe is positive. Until then CPUDecodeBackend is used
    everywhere (correct, one upload + one download).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "NVDECBackend is a server integration task (see class docstring). "
            "CPUDecodeBackend is used until a GPU decode library is available.")

    def open(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def reconnect(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError


def make_decode_backend(url, capabilities=None, stop_event=None,
                        reconnect_attempts=5, reconnect_backoff=1.0):
    """
    Pick the decode backend for `url`.

    Stage 0 / current: always CPU. Even when `capabilities["decode_backend"]`
    reports "nvdec" as a hardware candidate, the NVDEC backend is not
    implemented (NVDEC_IMPLEMENTED is False), so we use CPU. Stage 4 flips this
    on the server once a GPU decode lib is chosen.
    """
    want_nvdec = bool(capabilities and capabilities.get("decode_backend") == "nvdec")
    if want_nvdec and NVDEC_IMPLEMENTED:
        return NVDECBackend(url, stop_event=stop_event,
                            reconnect_attempts=reconnect_attempts,
                            reconnect_backoff=reconnect_backoff)
    return CPUDecodeBackend(url, stop_event=stop_event,
                            reconnect_attempts=reconnect_attempts,
                            reconnect_backoff=reconnect_backoff)
