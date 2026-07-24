"""
capabilities.py  --  runtime capability detection for the live pipeline.

============================ WHAT THIS DOES ================================
Decides, at startup, WHAT this machine can actually do -- so the same code runs
on a CPU laptop and an A100 server with NO source edits and NO hard-coded
`.cuda()` anywhere. Two questions:

  1. Which compute device for inference?  -> detect_device()
        auto -> cuda:0 if a CUDA torch build sees a GPU, else cpu.
  2. Can we decode video ON the GPU (NVDEC)?  -> probe_nvdec()
        Only if this OpenCV build has cv2.cudacodec AND a CUDA device exists.

------------------------------- KEY RULES ---------------------------------
  * NVDEC is probed ONCE. A negative result is PERMANENT for the run -- the
    reconnect logic must never re-probe hardware in a loop (v5 plan). Callers
    cache the capability report and reuse it.
  * Nothing here forces GPU. A CPU-only box gets a fully working (if slower)
    configuration; a GPU box gets GPU inference; NVDEC-to-GPU decode is only
    reported as usable if the decode library is genuinely present.
  * The report is logged (and returned) so a headless server run makes the
    chosen paths obvious from the logs alone.

NOTE (this repo, probed): torch is a CUDA build but the installed OpenCV has NO
cudacodec, so `nvdec_usable` is False here and decode falls back to CPU. True
NVDEC is a server-side dependency task (see decode_backend.NVDECBackend).
============================================================================
"""

import logging

import cv2

try:
    import torch
    _HAS_TORCH = True
except Exception:  # torch missing/broken -> CPU-only, no crash
    _HAS_TORCH = False

log = logging.getLogger("live.capabilities")


def detect_device(requested="auto"):
    """
    Resolve a torch-style device string.

    requested : "auto" | "cpu" | "cuda" | "cuda:N"
    returns   : "cpu" or "cuda:N". Falls back to cpu (with a warning) whenever
                CUDA is requested but unavailable, so a GPU config never crashes
                a CPU box.
    """
    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return "cpu"

    cuda_ok = _HAS_TORCH and torch.cuda.is_available()

    if req == "auto":
        return "cuda:0" if cuda_ok else "cpu"

    if req.startswith("cuda"):
        if cuda_ok:
            return requested if ":" in requested else "cuda:0"
        log.warning("device=%r requested but CUDA is unavailable -> using cpu", requested)
        return "cpu"

    log.warning("unknown device %r -> using cpu", requested)
    return "cpu"


def probe_nvdec(device, requested="auto"):
    """
    One-shot: is GPU video decode (NVDEC via cv2.cudacodec) usable here?

    Returns True only when the device is CUDA, this OpenCV build exposes
    cudacodec, and a CUDA device is present. A False result is meant to be
    treated as PERMANENT for the run (do not re-probe on reconnect).

    Note: this is a capability *candidate* check. Even when True, the actual
    NVDEC decode backend must exist and successfully open a stream to be used;
    see decode_backend.make_decode_backend.
    """
    if (requested or "auto").strip().lower() == "off":
        return False
    if not str(device).startswith("cuda"):
        return False
    if not hasattr(cv2, "cudacodec"):
        return False
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def capability_report(requested_device="auto", requested_nvdec="auto"):
    """
    Detect and log what this machine can do. Returned dict (cache + reuse it):
        {torch, cuda_available, device, opencv, nvdec_usable, decode_backend}
    `decode_backend` is the backend name that will actually be used given both
    the probe AND what is implemented (Stage 0: always "cpu").
    """
    device = detect_device(requested_device)
    cuda_ok = _HAS_TORCH and torch.cuda.is_available()
    nvdec = probe_nvdec(device, requested_nvdec)

    # What we will ACTUALLY use. NVDEC decode backend is not implemented yet
    # (server task), so even a positive probe resolves to CPU decode for now.
    from live import decode_backend as _db
    decode = "nvdec" if (nvdec and _db.NVDEC_IMPLEMENTED) else "cpu"

    report = {
        "torch": (torch.__version__ if _HAS_TORCH else "absent"),
        "cuda_available": cuda_ok,
        "device": device,
        "opencv": cv2.__version__,
        "nvdec_usable": nvdec,
        "decode_backend": decode,
    }
    log.info("capability report: %s", report)
    return report
