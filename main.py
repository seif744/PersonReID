"""
main.py  --  the ENTRY POINT. Run this file to start the pipeline(s).

    python main.py

This is the "conductor". It reads config.yaml and, for EACH video listed under
source.videos, runs the same per-frame pipeline:

    STAGE 1-2   VideoSource      video file  ->  frame
    STAGE 3-4   PersonDetector   frame       ->  [Detection...] (+ track IDs)
    STAGE 6prep CropSaver        crops each tracked person to crops/<name>/
    STAGE 5     drawing          detections  ->  annotated frame

MULTIPLE VIDEOS AT ONCE (and how we show them):
Each video gets its OWN detector (so their track IDs never mix) and its OWN
crops folder. The heavy work (decode + detect + track + draw) runs on one
worker THREAD per video, all at the same time. But OpenCV's windows are NOT
thread-safe, so the worker threads never call imshow directly -- instead each
worker drops its latest annotated frame into a shared buffer, and the MAIN
thread reads that buffer and shows every camera in its own window. This lets
you watch all cameras live, with the tracking boxes, safely.
"""

import argparse
from datetime import datetime
import os
import shutil
import sys
import threading
import time
import yaml   # reads our config.yaml (installed alongside ultralytics)
import cv2

# Put the `src` folder on the import path so EVERY module -- here, in reid/, and
# in database/ -- imports the same bare way (`from detector import ...`). `src`
# is the single source root. Run `python main.py` from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from video_source import VideoSource, is_stream_path
from detector import PersonDetector
from drawing import draw_detections, draw_hud
from crop_saver import CropSaver


def load_dotenv(path=".env"):
    """
    Minimal .env loader (no python-dotenv dependency). Reads KEY=VALUE lines from
    an untracked .env file into os.environ WITHOUT overriding vars already set in
    the real environment (so prod env vars win over a stray local .env). Secrets
    like QDRANT_API_KEY live here; the file is gitignored.
    """
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def load_config(path="config.yaml"):
    """Read config.yaml into a plain Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_run_summary(store, jobs, cfg, run_id=None):
    """
    Print WHAT the run produced -- CONSOLE ONLY, no files written. Reads the
    gallery back from the store: total observations, distinct people
    (reid_ids), and which people were seen in more than one camera (the
    product's core result). Then lists the annotated videos on disk.
    """
    from collections import defaultdict

    print("\n===== RUN SUMMARY =====")
    if run_id is not None:
        print(f"  Run id: {run_id}")

    if store is not None:
        try:
            cam_obs = defaultdict(int)
            gid_obs = defaultdict(int)
            # reid_id -> camera -> set of track ids seen for that person there.
            gid_cameras = defaultdict(lambda: defaultdict(set))
            offset = None
            while True:
                pts, offset = store.client.scroll(
                    store.collection, limit=1000, offset=offset,
                    with_payload=True, with_vectors=False)
                for p in pts:
                    pl = p.payload or {}
                    if run_id is not None and pl.get("run_id") != run_id:
                        continue
                    cam_obs[pl.get("camera")] += 1
                    gid = pl.get("reid_id", pl.get("global_id"))
                    if gid is None:
                        continue
                    gid_obs[gid] += 1
                    gid_cameras[gid][pl.get("camera")].add(pl.get("track_id"))
                if offset is None:
                    break

            total = sum(cam_obs.values())
            cross_ids = sorted(g for g, cams in gid_cameras.items() if len(cams) > 1)
            print(f"  Store: {total} observations -> "
                  f"{len(gid_obs)} distinct people (reid_ids)")
            print(f"  Cross-camera people: {len(cross_ids)}")
            for gid in cross_ids:
                parts = []
                for camera in sorted(gid_cameras[gid]):
                    tracks = sorted(t for t in gid_cameras[gid][camera] if t is not None)
                    tracks_txt = " + ".join(f"track {int(t):04d}" for t in tracks)
                    parts.append(f"{camera} ({tracks_txt})")
                print(f"    GID {gid}: " + " + ".join(parts))
        except Exception as e:
            print(f"  Store: (could not summarize: {e})")

    # Annotated videos, if written.
    if cfg.get("display", {}).get("save_annotated"):
        vids = [f"output_{name}.mp4" for name, *_ in jobs
                if os.path.exists(f"output_{name}.mp4")]
        if vids:
            print(f"  Annotated videos: {', '.join(vids)}")
    print("=======================\n")


def get_screen_size():
    """
    Return the monitor's (width, height) in pixels so we can fit windows on
    screen. Uses tkinter (standard library). Falls back to 1080p if unavailable.
    """
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()  # don't actually show tkinter's own window
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return (1920, 1080)


def gui_available():
    """
    True only if OpenCV can actually OPEN a window in this environment.

    Why a subprocess: when the display/Qt backend can't initialise (a headless
    box, or a WSL without a working X server -- the classic "Could not load the
    Qt platform plugin xcb"), OpenCV does not raise a catchable Python error --
    it ABORTS the whole process (SIGABRT / core dump). So we cannot probe it
    in-process with try/except. We probe in a throwaway child process instead:
    if the child aborts, we learn windows are unusable WITHOUT taking down the
    real pipeline, and fall back to headless (the saved videos still render).
    """
    # No DISPLAY on Linux => definitely no windows; skip the (slow) probe.
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "import cv2; cv2.namedWindow('__probe__'); cv2.destroyAllWindows()"],
            capture_output=True, timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


# Video file extensions we recognise when scanning a directory.
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg")


def _is_url(path):
    """True for stream URLs (rtsp://, http://) which have no on-disk file to check."""
    return isinstance(path, str) and "://" in path


def normalize_sources(videos):
    """
    Turn a list of video entries into a clean list of (name, path) with UNIQUE
    names (names become crops/<name>/ and output_<name>.mp4, so collisions would
    overwrite each other).

    Each entry may be either:
      - a plain string path/URL  -> name is derived from the file name, OR
      - a dict {name, path}      -> we use the given friendly name.
    """
    result = []
    used = set()
    for entry in videos:
        if isinstance(entry, dict):
            path = entry["path"]
            name = entry.get("name") or os.path.splitext(os.path.basename(str(path)))[0]
        else:
            path = entry
            name = os.path.splitext(os.path.basename(str(path)))[0]

        name = name or "camera"
        # De-duplicate: cam, cam_2, cam_3, ...
        base, k = name, 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        result.append((name, path))
    return result


def resolve_sources(args, cfg):
    """
    Decide which videos to process, dynamically. Precedence:
        --videos <paths...>   >   --videos-dir <dir>   >   config source.videos

    So the pipeline runs on ANY videos the user points at, without editing code
    or config. Returns (sources, origin_description). Validates that local files
    exist (stream URLs are skipped). Exits with a clear message on bad input.
    """
    if args.videos:
        sources = normalize_sources(list(args.videos))
        origin = "--videos"
    elif args.videos_dir:
        if not os.path.isdir(args.videos_dir):
            raise SystemExit(f"[main] --videos-dir is not a directory: {args.videos_dir}")
        files = [os.path.join(args.videos_dir, f)
                 for f in sorted(os.listdir(args.videos_dir))
                 if f.lower().endswith(VIDEO_EXTS)]
        if not files:
            raise SystemExit(f"[main] No video files ({', '.join(VIDEO_EXTS)}) "
                             f"found in {args.videos_dir}")
        sources = normalize_sources(files)
        origin = f"--videos-dir {args.videos_dir}"
    else:
        sources = normalize_sources(cfg.get("source", {}).get("videos", []))
        origin = "config.yaml (source.videos)"

    missing = [p for _, p in sources if not _is_url(p) and not os.path.exists(p)]
    if missing:
        raise SystemExit("[main] These video files were not found:\n  "
                         + "\n  ".join(missing))
    return sources, origin


def clear_directory(path):
    """Remove all contents of a directory, creating it if needed."""
    if not path:
        return
    if os.path.exists(path):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path) and not os.path.islink(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
    else:
        os.makedirs(path, exist_ok=True)


def build_gid_map(store, run_id):
    """
    Read the store (AFTER reconciliation) and return the FINAL global id for each
    camera-local track: {(camera, track_id): global_id}. One track should carry a
    single global id, but we take the most common as a safety net. This is what
    lets the re-render label the same person with the same GID in every camera.
    """
    from collections import defaultdict, Counter
    if store is None:
        return {}
    votes = defaultdict(Counter)
    offset = None
    while True:
        pts, offset = store.client.scroll(
            store.collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if run_id is not None and pl.get("run_id") != run_id:
                continue
            gid = pl.get("reid_id", pl.get("global_id"))
            if gid is None:
                continue
            votes[(pl.get("camera"), pl.get("track_id"))][gid] += 1
        if offset is None:
            break
    return {key: counter.most_common(1)[0][0] for key, counter in votes.items()}


def render_final_videos(jobs, cfg, shared, store, run_id):
    """
    SECOND PASS -- write output_<camera>.mp4 with the FINAL (post-reconciliation)
    global ids. The live pass only captured box geometry; cross-camera identity is
    settled afterwards by reconcile.py. Re-drawing here means a person who walked
    from cam A to cam B carries the SAME "GID n" (and the same colour) in BOTH
    output videos -- the core proof the pipeline works. We re-decode the source
    frames (cheap; no model) and draw the captured boxes, so boxes/track ids match
    the live pass exactly.
    """
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    gid_map = build_gid_map(store, run_id)
    resize_width = cfg["source"].get("resize_width", 0)
    fps = float(cfg["display"].get("output_fps", 20.0))

    def render_one(name, path):
        annos = shared["annotations"].get(name)
        if not annos:
            return
        out_path = f"output_{name}.mp4"
        writer = None
        try:
            with VideoSource(path=path) as cam:
                for frame_index, frame in enumerate(cam.frames()):
                    if frame_index >= len(annos):
                        break  # only re-render the frames the live pass processed
                    if resize_width and frame.shape[1] > resize_width:
                        scale = resize_width / frame.shape[1]
                        frame = cv2.resize(
                            frame,
                            (resize_width, int(round(frame.shape[0] * scale))),
                            interpolation=cv2.INTER_AREA)

                    dets = []
                    for anno in annos[frame_index]:
                        if isinstance(anno, dict):
                            x1 = anno["x1"]
                            y1 = anno["y1"]
                            x2 = anno["x2"]
                            y2 = anno["y2"]
                            track_id = anno.get("track_id")
                            conf = anno.get("confidence", 0.0)
                        else:
                            x1, y1, x2, y2, track_id, conf = anno[:6]

                        dets.append(
                            SimpleNamespace(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                track_id=track_id, confidence=conf,
                                global_id=gid_map.get((name, track_id)),
                                reid_id=gid_map.get((name, track_id)),
                            )
                        )
                    frame = draw_detections(frame, dets)
                    frame = draw_hud(frame, person_count=len(dets))

                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    writer.write(frame)
        except Exception as e:
            print(f"[render] {name}: ERROR re-rendering: {e}")
        finally:
            if writer is not None:
                writer.release()
                print(f"[render] Final annotated video -> {out_path}")

    max_workers = min(len(jobs), os.cpu_count() or len(jobs) or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(render_one, name, path) for name, path, *_ in jobs]
        for fut in futures:
            fut.result()


# =============================================================================
# PHASE 1 (LIVE mode only): record raw streams to disk, no ML.
# Decoupling capture from processing is what lets N cameras run without any
# real-time constraint -- reading+writing frames is cheap and keeps up, and the
# heavy detect/track/embed/reconcile happens OFFLINE afterwards on the recording
# (Phase 2 = the ordinary batch pipeline). One recorder thread per stream; an
# optional raw monitoring window runs on the main thread.
# =============================================================================
def capture_stream(name, url, rec_path, fps, stream_cfg, shared, stop_event):
    """
    Read `url` and write every decoded frame to `rec_path` at `fps`, until
    stop_event is set (q / Ctrl-C) or the source ends. NO detection/embedding --
    just decode + write, so it keeps up with real time for several cameras.
    Publishes the latest frame to `shared` for the optional monitoring window.

    drop_stale is OFF here: we want a faithful, full-frame-rate recording, not
    the freshest-only frames the live display wanted. Reconnect stays ON so a
    transient RTSP hiccup doesn't end the capture.
    """
    tag = f"[{name}]"
    writer = None
    n = 0
    try:
        with VideoSource(
            path=url,
            stop_event=stop_event,
            reconnect_attempts=stream_cfg.get("reconnect_attempts", 5),
            reconnect_backoff=stream_cfg.get("reconnect_backoff", 1.0),
            drop_stale=False,
        ) as cam:
            print(f"{tag} Recording stream -> {rec_path}")
            for frame in cam.frames():
                if stop_event.is_set():
                    break
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(rec_path, fourcc, fps, (w, h))
                writer.write(frame)
                n += 1
                # Publish for the monitoring window (a copy so the display thread
                # can read it while we move on).
                with shared["lock"]:
                    shared["frames"][name] = frame.copy()
                if n % 60 == 0:
                    print(f"{tag} recording... {n} frames")
    except Exception as e:
        print(f"{tag} ERROR while recording: {e}")
    finally:
        if writer is not None:
            writer.release()
        with shared["lock"]:
            shared["done"].add(name)
        print(f"{tag} Recorded {n} frames -> {rec_path}")


def run_capture_phase(sources, cfg):
    """
    Record every STREAM source to recorded_<name>.mp4 (files pass through). Runs
    one recorder thread per stream plus an optional raw monitoring window on the
    main thread; returns {name: recorded_path} for the streams. Blocks until the
    user stops it ('q' in a window, or Ctrl-C) or every stream ends on its own.
    """
    stream_sources = [(n, p) for n, p in sources if is_stream_path(p)]
    if not stream_sources:
        return {}

    stream_cfg = cfg.get("source", {}).get("stream", {}) or {}
    out_cfg = cfg.get("output", {}) or {}
    disp_cfg = cfg.get("display", {}) or {}
    # Recording fps: streams report unreliable CAP_PROP_FPS, so we use a fixed
    # config fps for the recording. Phase 2 re-renders at display.output_fps
    # regardless (same as any file), so what matters is that every frame is kept.
    fps = float(out_cfg.get("fps_default", disp_cfg.get("output_fps", 20.0)))

    shared = {"lock": threading.Lock(), "frames": {}, "done": set()}
    stop_event = threading.Event()

    recorded = {}
    threads = []
    for name, url in stream_sources:
        rec_path = f"recorded_{name}.mp4"
        recorded[name] = rec_path
        t = threading.Thread(
            target=capture_stream,
            args=(name, url, rec_path, fps, stream_cfg, shared, stop_event),
            name=f"rec-{name}",
        )
        t.start()
        threads.append(t)

    names = [n for n, _ in stream_sources]
    want_window = disp_cfg.get("show_window", True)
    print(f"[capture] Recording {len(names)} stream(s): {', '.join(names)}. "
          f"Press 'q' in a window or Ctrl-C to stop and start processing.")
    try:
        if want_window and gui_available():
            # Reuse the monitoring window loop: it shows the raw frames and quits
            # on 'q'. total_videos = number of streams so it exits when all end.
            run_display(names, cfg, shared, stop_event, len(names))
        else:
            if want_window:
                print("[capture] No usable GUI display -> recording HEADLESS. "
                      "Press Ctrl-C to stop.")
            # Headless: wait for the streams (which only end on stop / failure).
            for t in threads:
                t.join()
    except KeyboardInterrupt:
        print("\n[capture] Ctrl-C -> stopping capture...")

    stop_event.set()
    for t in threads:
        t.join(timeout=10)
    cv2.destroyAllWindows()
    return recorded


# =============================================================================
# The WORKER: runs the whole pipeline for ONE video, on its own thread.
# It does NOT touch any window. It only:
#   - reads frames, detects+tracks, saves crops, draws boxes, and
#   - publishes its latest annotated frame into `shared` for the main thread.
# =============================================================================
def process_video(name, path, detector, crop_saver, embedder, store, identity,
                  locks, cfg, shared, stop_event, run_id):
    trk_cfg = cfg["tracker"]
    # 0 = process the whole video; a positive number stops early (handy for a
    # quick CPU test without waiting for hundreds of frames).
    max_frames = cfg["source"].get("max_frames", 0)
    # 0 = keep native resolution. A positive width downscales every frame (keeping
    # aspect) BEFORE detect/track/crop/embed -- QHD footage on CPU is dominated by
    # pixel count, so e.g. 1280 is dramatically faster for dev with little ReID
    # cost. Everything downstream uses the same resized frame, so coords stay
    # consistent (crops are smaller too).
    resize_width = cfg["source"].get("resize_width", 0)
    show_window = cfg["display"].get("show_window", True)

    tag = f"[{name}]"
    frame_index = 0
    processed_here = 0   # how many fresh embeddings THIS camera has checked
    # Per-frame box geometry captured for the FINAL re-render pass. We record the
    # boxes + track ids (NOT the pixels) as we go, so after cross-camera
    # reconciliation we can re-draw the video with the final global ids without
    # re-running detection. annotations[f] = dicts with box coords, track id,
    # and confidence.
    annotations = []

    try:
        with VideoSource(path=path) as cam:
            print(f"{tag} Pipeline running on '{path}'.")
            for frame in cam.frames():        # STAGE 1-2: one frame at a time

                # Stop early if the user pressed 'q' in the main thread.
                if stop_event.is_set():
                    break

                # ---- Optional downscale (dev speed on CPU) -----------------
                # Do it FIRST so every later stage sees the same smaller frame.
                if resize_width and frame.shape[1] > resize_width:
                    scale = resize_width / frame.shape[1]
                    frame = cv2.resize(
                        frame, (resize_width, int(round(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)

                # ---- STAGE 3 (+4): detect, with or without track IDs --------
                if trk_cfg["enabled"]:
                    detections = detector.track(frame)   # Stage 4: with IDs
                else:
                    detections = detector.detect(frame)  # Stage 3: no IDs

                # ---- STAGE 6 prep: crop from the CLEAN frame (before drawing)-
                if crop_saver is not None:
                    crop_saver.save(frame, detections, frame_index)

                # ---- STAGE 5/6: embed tracked people + check Qdrant ---------
                # Uses the CLEAN frame (before boxes are drawn). We embed each
                # track (throttled + cached by TrackEmbedder) and hand fresh
                # embeddings to the Identity Service, which searches its own
                # gallery and assigns a stable GLOBAL id (or falls back to just
                # persisting raw observations when identity assignment is off).
                if embedder is not None:
                    with locks["model"]:            # shared model: one at a time
                        embedder.process(frame, detections, frame_index)
                    fresh = embedder.last_embedded   # tracks re-embedded this frame

                    if identity is not None and fresh:
                        # Turn fresh embeddings into stable GLOBAL ids. The service
                        # searches the gallery AND commits under the chosen id, so
                        # it is the sole writer -- serialize it with one lock.
                        with locks["identity"]:
                            for d in fresh:
                                gid = identity.assign(
                                    name, d.track_id, d.embedding, frame_index,
                                    run_id=run_id,
                                    crop_quality=d.crop_quality)
                                d.global_id = gid
                                d.reid_id = gid
                        processed_here += len(fresh)
                    elif store is not None and fresh:
                        # Identity off: just persist the raw observations.
                        payloads = [
                            {
                                "camera": name,
                                "track_id": d.track_id,
                                "frame": frame_index,
                                "run_id": run_id,
                                "crop_quality": d.crop_quality,
                            }
                            for d in fresh
                        ]
                        with locks["store"]:
                            store.add_many([d.embedding for d in fresh], payloads)
                        processed_here += len(fresh)

                # ---- Record box geometry for the final re-render pass --------
                # Captured BEFORE drawing (clean coords). The annotated video is
                # NOT written here: it is written in a second pass after
                # reconciliation so its labels use the FINAL global ids (so the
                # same person carries the same GID across both cameras' videos).
                annotations.append([
                    {
                        "x1": d.x1,
                        "y1": d.y1,
                        "x2": d.x2,
                        "y2": d.y2,
                        "track_id": d.track_id,
                        "confidence": d.confidence,
                    }
                    for d in detections
                ])

                if show_window:
                    # ---- STAGE 5: draw (modifies frame in place) ----------
                    frame = draw_detections(frame, detections)
                    frame = draw_hud(frame, person_count=len(detections))

                    # ---- Publish latest frame for the main thread to show --
                    # We store a COPY so the display thread can read it while
                    # we move on to the next frame. Skip this entirely in
                    # headless runs; the final output video is rendered later.
                    with shared["lock"]:
                        shared["frames"][name] = frame.copy()

                frame_index += 1

                # ---- Heartbeat: prove the pipeline is alive on long videos ---
                # Full videos on CPU take minutes; without this the terminal
                # looks frozen. Log progress every 30 frames.
                if frame_index % 30 == 0:
                    print(f"{tag} frame {frame_index}: {len(detections)} people, "
                          f"{processed_here} fresh embeddings checked so far")

                # Stop early if a frame cap was set (for quick tests).
                if max_frames and frame_index >= max_frames:
                    print(f"{tag} Reached max_frames={max_frames}, stopping.")
                    break

    except Exception as e:
        # One video failing shouldn't kill the others -- log and move on.
        print(f"{tag} ERROR: {e}")
    finally:
        # Hand the captured geometry to the main thread for the re-render pass,
        # and mark this camera as finished.
        with shared["lock"]:
            shared["annotations"][name] = annotations
            shared["done"].add(name)
        print(f"{tag} Done. Processed {frame_index} frames.")


# =============================================================================
# The DISPLAY loop: runs on the MAIN thread. Reads each camera's latest frame
# from the shared buffer and shows it in its own window. This is the ONLY place
# that calls OpenCV's window functions.
# =============================================================================
def run_display(names, cfg, shared, stop_event, total_videos):
    disp_cfg = cfg["display"]
    window_scale = disp_cfg.get("window_scale", 1.0)

    screen_w, screen_h = get_screen_size()
    created = set()   # which windows we've already created + sized
    # Some environments (headless servers, a WSL without WSLg/X) can't open a
    # GUI window -- cv2 raises then. Rather than crash the whole run (which would
    # also lose the saved videos), we fall back to a headless wait: the session
    # keeps running and is stopped with Ctrl-C, which still finalizes outputs.
    gui_ok = True

    while True:
        # Take a quick snapshot of the latest frames under the lock.
        with shared["lock"]:
            frames = dict(shared["frames"])
            done = set(shared["done"])

        if gui_ok:
            try:
                for name, frame in frames.items():
                    if name not in created:
                        # Size each window to fit the screen. When there are several
                        # cameras, give each a slice of the screen width so they fit
                        # side by side (0.9 / N), keeping aspect ratio.
                        h, w = frame.shape[:2]
                        frac = 0.9 / max(1, total_videos)
                        fit = min(1.0, (screen_w * frac) / w, (screen_h * 0.9) / h)
                        scale = fit * window_scale
                        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(name, max(1, int(w * scale)), max(1, int(h * scale)))
                        created.add(name)

                    cv2.imshow(name, frame)

                # waitKey both refreshes the windows AND reads the keyboard. 'q' quits.
                if (cv2.waitKey(30) & 0xFF) == ord("q"):
                    print("[display] 'q' pressed -> stopping all videos.")
                    stop_event.set()
                    break
            except cv2.error as e:
                gui_ok = False
                print(f"[display] Live windows unavailable ({e}); continuing "
                      f"HEADLESS. Press Ctrl-C to stop and finalize the saved "
                      f"videos.")

        if not gui_ok:
            # No windows: can't read 'q', so wait for stop (Ctrl-C sets it) or
            # for every source to finish on its own.
            if stop_event.is_set():
                break
            time.sleep(0.03)

        # Exit once every video has finished. (We've already shown whatever
        # frames each produced; a video that failed instantly just won't have
        # a window.)
        if len(done) == total_videos:
            print("[display] All videos finished.")
            break

    # Keep the final frames up until a key is pressed, so nothing vanishes.
    if gui_ok and created and not stop_event.is_set():
        print("[display] Press any key in a window to close.")
        cv2.waitKey(0)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-camera person ReID pipeline.")
    p.add_argument(
        "--reset", action="store_true",
        help="Wipe the vector store collection before running, so a test starts "
             "from a clean gallery. DESTRUCTIVE -- deletes all stored embeddings.",
    )
    p.add_argument(
        "--videos", nargs="+", metavar="VIDEO", default=None,
        help="One or more video files or RTSP/HTTP stream URLs to process, "
             "space-separated. Overrides config source.videos. Each camera's name "
             "is derived from the file name. Example: "
             "--videos cam_a.mp4 cam_b.mp4",
    )
    p.add_argument(
        "--videos-dir", metavar="DIR", default=None,
        help="Process every video file in this directory "
             f"({'/'.join(e.lstrip('.') for e in VIDEO_EXTS)}). "
             "Overrides config source.videos.",
    )
    p.add_argument(
        "--mode", choices=["auto", "live", "batch"], default=None,
        help="Override live.run.mode. auto (default; live if a source is a URL) | "
             "live (v5 real-time src/live/ pipeline) | batch (file pipeline). "
             "Handy for testing the live pipeline without editing config.yaml.",
    )
    return p.parse_args()


def main():
    # ---- Load settings ------------------------------------------------------
    args = parse_args()
    load_dotenv()          # pull secrets (QDRANT_API_KEY, ...) from untracked .env
    cfg = load_config()
    det_cfg = cfg["detector"]
    trk_cfg = cfg["tracker"]
    crop_cfg = cfg["crops"]
    disp_cfg = cfg["display"]

    sources, src_origin = resolve_sources(args, cfg)
    if not sources:
        print("[main] No videos to process. Pass --videos <paths...>, "
              "--videos-dir <dir>, or set source.videos in config.yaml.")
        return
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[main] Preparing {len(sources)} video(s) from {src_origin}: "
          f"{', '.join(n for n, _ in sources)}")
    print(f"[main] Run id: {run_id}")

    # ---- v5 live-pipeline routing (Stage 0) --------------------------------
    # The new real-time streaming pipeline lives in src/live/ and is built in
    # stages. `live.run.mode` selects the path; a capability report is logged so
    # the chosen device/decode is visible even on a headless server. This block
    # does NOT change file-batch behaviour: mode `auto` (default) keeps today's
    # logic (record-then-batch for URLs, batch for files); `batch` forces the
    # file path; `live` is reserved for the v5 pipeline (Stage 1+).
    live_cfg = cfg.get("live", {}) or {}
    run_cfg = live_cfg.get("run", {}) or {}
    run_mode = args.mode or run_cfg.get("mode", "auto")   # --mode overrides config
    try:
        from live.capabilities import capability_report
        cap = capability_report(
            requested_device=run_cfg.get("device", "auto"),
            requested_nvdec=(live_cfg.get("capture", {}) or {}).get("nvdec", "auto"),
        )
        print(f"[main] Capability report: {cap}")
    except Exception as e:
        print(f"[main] (capability report unavailable: {e})")

    if run_mode == "live":
        # v5 real-time pipeline (src/live/). Handles files OR streams; produces
        # output_<cam>.mp4 live. Separate from the file-batch + record-then-batch
        # paths below (which `auto`/`batch` still use).
        from live.pipeline import LivePipeline
        LivePipeline(sources, cfg).run()
        print("[main] All done.")
        return

    # A run is "live" if ANY source is a stream URL (rtsp://, http://, ...).
    # LIVE mode is a two-phase design that AVOIDS any real-time processing
    # constraint (so N cameras never build up latency):
    #   Phase 1 (here): cheaply RECORD each raw stream to recorded_<name>.mp4
    #     with NO ML -- just decode + write, which easily keeps up with several
    #     cameras -- while showing an optional raw monitoring window. Stop with
    #     'q' or Ctrl-C.
    #   Phase 2 (the rest of main): run the EXACT SAME batch pipeline as a video
    #     file run on those recordings -> detect/track/embed/assign/reconcile/
    #     re-render -> output_<name>.mp4 at full fps and full quality.
    # A files-only run skips Phase 1 entirely and is byte-identical to before.
    # `run.mode: batch` forces the file path even if a URL was passed.
    live_run = (run_mode != "batch") and any(is_stream_path(p) for _, p in sources)
    recorded_paths = {}    # name -> recorded_<name>.mp4 (stream sources only)
    if live_run:
        recorded_paths = run_capture_phase(sources, cfg)
        if not any(os.path.exists(p) for p in recorded_paths.values()):
            print("[main] No stream frames were recorded; nothing to process.")
            return
        # Converge with the file path: each stream source becomes its recording;
        # any file sources in the same run stay as-is. From here on `sources`
        # are all local files and Phase 2 is an ordinary batch run.
        sources = [(name, recorded_paths.get(name, path)) for name, path in sources]
        print("[main] Capture finished -> processing the recording(s) offline "
              "through the full pipeline (no real-time constraint).")

    if crop_cfg.get("save"):
        clear_directory(crop_cfg.get("dir", "crops"))
        print(f"[main] Cleared crops directory: {crop_cfg.get('dir', 'crops')}")

    # ---- Build one detector + crop saver PER video --------------------------
    # Each detector has its OWN tracker memory (IDs never bleed between videos).
    # Each crop saver writes to crops/<name>/ so crops stay separate.
    jobs = []  # each: (name, path, detector, crop_saver)
    for name, path in sources:
        detector = PersonDetector(
            model_path=det_cfg["model"],
            confidence_threshold=det_cfg["confidence_threshold"],
            person_class_id=det_cfg["person_class_id"],
            tracker_config=trk_cfg["config"],
            pose_ensemble=det_cfg.get("pose_ensemble"),
        )
        crop_saver = None
        if crop_cfg["save"]:
            per_video_dir = os.path.join(crop_cfg["dir"], name)
            crop_saver = CropSaver(
                output_dir=per_video_dir,
                interval=crop_cfg["interval"],
                padding=crop_cfg["padding"],
            )
            print(f"[main] {name}: crops -> {per_video_dir}/")
        jobs.append((name, path, detector, crop_saver))

    # ---- Build the SHARED ReID model + Qdrant access ------------------------
    # ONE extractor for the whole run: the model is the expensive resource, so we
    # load it once, share it across camera threads, and guard it with a lock
    # (worker threads must not run the model concurrently). ONE store, because all
    # cameras must write to the SAME gallery. But each camera gets its OWN
    # TrackEmbedder: its cache is keyed by track_id, and track_ids are per-camera
    # (cam A's track 1 is not cam B's track 1), so the caches must not mix.
    extractor = None
    store = None
    identity = None
    embedders = {}
    locks = {
        "model": threading.Lock(),      # shared model: one forward pass at a time
        "store": threading.Lock(),      # raw store writes (identity-off fallback)
        "identity": threading.Lock(),   # identity: mints ids + writes gallery
    }

    reid_cfg = cfg.get("reid", {})
    if reid_cfg.get("enabled"):
        from reid.extractor import ReIDExtractor
        from reid.service import TrackEmbedder
        print("[main] Loading ReID model (shared across all cameras)...")
        extractor = ReIDExtractor(
            weights=reid_cfg["weights"],
            device=reid_cfg.get("device"),
        )
        for name, _ in sources:
            embedders[name] = TrackEmbedder(
                extractor,
                interval=reid_cfg.get("interval", 10),
                ttl=reid_cfg.get("ttl", 300),
                quality=reid_cfg.get("quality"),
                max_embeddings_per_track=reid_cfg.get("max_embeddings_per_track", 0),
                warmup_embeddings=reid_cfg.get("warmup_embeddings", 3),
            )
        print("[main] ReID enabled -> embedding tracked people (no ids assigned).")

    store_cfg = cfg.get("store", {})
    if store_cfg.get("enabled"):
        from database.store import PersonVectorStore
        # URL: env var wins over config.yaml. API key: env only (never committed).
        url = os.environ.get("QDRANT_URL") or store_cfg.get("url") or None
        api_key = os.environ.get("QDRANT_API_KEY") or None
        path = store_cfg.get("path", "qdrant_data")
        store = PersonVectorStore(path=path, url=url, api_key=api_key)
        backend = url if url else f"LOCAL '{path}'"
        print(f"[main] Vector store ready at {backend} "
              f"(existing points: {store.count()}).")

        if args.reset:
            store.reset()
            print(f"[main] --reset: cleared the store (now {store.count()} points).")

    # The Identity Service needs the store as its gallery. It is SHARED across
    # cameras (global ids are only "global" if every camera decides from the same
    # gallery), and it is the sole writer once enabled.
    id_cfg = cfg.get("identity", {})
    if id_cfg.get("enabled") and store is not None:
        from identity.service import IdentityService
        # NOTE: the online decision core stays OFF here (online_cfg=None). In the
        # record-then-batch design every run that reaches this point processes
        # local files (recordings or user files), so identity is owned by the
        # SAME sticky live-assign + offline reconcile as a video-file run -- that
        # is what gives the saved MP4 its file-quality ids. (The online core in
        # service.py remains available/gated for a future live-preview need.)
        identity = IdentityService(
            store,
            threshold=id_cfg.get("threshold", 0.85),
            top_k=id_cfg.get("top_k", 10),
            bank_size=id_cfg.get("bank_size", 20),
            min_score_gap=id_cfg.get("min_score_gap", 0.03),
            rerank_cfg=id_cfg.get("rerank"),
            verification_cfg=id_cfg.get("verification"),
            online_cfg=None,
        )
        print("[main] Identity Service enabled -> assigning GLOBAL ids.")

    # ---- Shared state between worker threads and the display loop ------------
    # frames: {name: latest annotated frame}   done: {names that finished}
    shared = {
        "lock": threading.Lock(),
        "frames": {},
        "done": set(),
        "annotations": {},   # {name: [per-frame box geometry]} for the re-render
    }
    stop_event = threading.Event()  # set to True to ask all workers to stop

    # ---- Start one worker thread per video ----------------------------------
    threads = []
    for name, path, detector, crop_saver in jobs:
        t = threading.Thread(
            target=process_video,
            args=(name, path, detector, crop_saver, embedders.get(name), store,
                  identity, locks, cfg, shared, stop_event, run_id),
            name=name,
        )
        t.start()
        threads.append(t)

    # ---- Display on the main thread (or just wait, if windows are off) -------
    # Ctrl-C is caught so a deliberately-stopped live session still finalizes:
    # we set stop_event and fall through to the reconcile + re-render below,
    # producing the saved videos just like pressing 'q' does.
    show_window = disp_cfg.get("show_window", True)
    try:
        if show_window:
            run_display([n for n, *_ in jobs], cfg, shared, stop_event, len(jobs))
        else:
            # Headless: nothing to show, just wait for all videos to finish.
            for t in threads:
                t.join()
    except KeyboardInterrupt:
        print("\n[main] Interrupted (Ctrl-C) -> stopping all sources and "
              "finalizing outputs...")

    # Ask workers to stop (if the user quit) and wait for them to wind down.
    stop_event.set()
    for t in threads:
        t.join(timeout=5)

    cv2.destroyAllWindows()

    # ---- STAGE 7b: offline reconciliation ----------------------------------
    # The live Identity Service decides each track once and never revisits it, so
    # a person seen in two cameras AT THE SAME TIME is minted twice and can't be
    # matched live. Now that every camera has finished and the gallery is fully
    # populated, merge global ids that are the same person across cameras.
    recon_cfg = id_cfg.get("reconcile", {}) if id_cfg else {}
    if identity is not None and recon_cfg.get("enabled"):
        from identity.reconcile import reconcile_tracklets
        threshold = recon_cfg.get("threshold")
        if threshold is None:
            threshold = id_cfg.get("threshold", 0.85)
        print("[main] Reconciling identities across cameras...")
        reconcile_tracklets(
            store,
            threshold=threshold,
            run_id=run_id,
            same_camera_threshold=recon_cfg.get("same_camera_threshold", 0.90),
            require_reciprocal_best=recon_cfg.get("require_reciprocal_best", True),
            min_tracklet_observations=recon_cfg.get("min_tracklet_observations", 1),
        )

    # ---- Final render: annotated videos with the reconciled global ids ------
    # Done AFTER reconciliation so the same person carries the same GID (and
    # colour) across every camera's output video.
    if disp_cfg.get("save_annotated"):
        print("[main] Rendering annotated videos with final global ids...")
        render_final_videos(jobs, cfg, shared, store, run_id)

    print_run_summary(store, jobs, cfg, run_id=run_id)

    # ---- Clean up raw stream recordings (LIVE mode only) --------------------
    # The recordings were just an intermediate to reach the batch pipeline. By
    # default we delete them once output_<name>.mp4 exists; keep them with
    # output.keep_recording: true (e.g. to re-process or audit the raw capture).
    if recorded_paths:
        keep = cfg.get("output", {}).get("keep_recording", False)
        for name, rec_path in recorded_paths.items():
            if not os.path.exists(rec_path):
                continue
            if keep:
                print(f"[main] Keeping raw recording: {rec_path}")
            else:
                try:
                    os.remove(rec_path)
                    print(f"[main] Removed raw recording: {rec_path}")
                except OSError as e:
                    print(f"[main] Could not remove {rec_path}: {e}")

    print("[main] All done.")


if __name__ == "__main__":
    # This guard means main() only runs when you execute `python main.py`,
    # and is REQUIRED for threads to behave correctly.
    main()
