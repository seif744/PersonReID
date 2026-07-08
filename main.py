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
import yaml   # reads our config.yaml (installed alongside ultralytics)
import cv2
import numpy as np

# Put the `src` folder on the import path so EVERY module -- here, in reid/, and
# in database/ -- imports the same bare way (`from detector import ...`). `src`
# is the single source root. Run `python main.py` from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from video_source import VideoSource
from detector import PersonDetector
from drawing import draw_detections, draw_hud
from crop_saver import CropSaver


def _track_dir(crop_dir, camera, track_id):
    """Return the crop folder for one camera-local track."""
    if track_id is None:
        return None
    return os.path.join(crop_dir, camera, f"id_{int(track_id):04d}")


def _sample_crop_paths(track_dir, max_images=6):
    """Pick a small, time-spread sample from one track's saved crops."""
    if not track_dir or not os.path.isdir(track_dir):
        return []
    files = sorted(
        os.path.join(track_dir, f)
        for f in os.listdir(track_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(files) <= max_images:
        return files
    if max_images <= 1:
        return [files[0]]
    indexes = np.linspace(0, len(files) - 1, max_images, dtype=int)
    return [files[i] for i in indexes]


def _put_label(img, text, x, y, scale=0.5, thickness=1):
    """Draw readable black text on a light report image."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (20, 20, 20), thickness, cv2.LINE_AA)


def _make_cross_camera_contact_sheet(gid, gid_info, crop_dir, report_dir,
                                     thumbs_per_track=6):
    """
    Build a side-by-side visual report for one global_id that appeared in 2+
    cameras. Rows are camera-local tracks; columns are sampled saved crops.
    """
    rows = []
    for camera in sorted(gid_info["cameras"]):
        tracks = gid_info["cameras"][camera]["tracks"]
        for track_id in sorted((t for t in tracks if t is not None), key=int):
            track_dir = _track_dir(crop_dir, camera, track_id)
            paths = _sample_crop_paths(track_dir, thumbs_per_track)
            rows.append((camera, track_id, paths, track_dir))

    if not rows:
        return None

    thumb_w, thumb_h = 120, 180
    left_w = 230
    pad = 12
    header_h = 48
    row_h = thumb_h + 34
    width = left_w + pad + thumbs_per_track * (thumb_w + pad) + pad
    height = header_h + len(rows) * row_h + pad
    sheet = np.full((height, width, 3), 245, dtype=np.uint8)

    cams = ", ".join(sorted(gid_info["cameras"]))
    _put_label(sheet, f"GID {gid} - seen in: {cams}", 14, 30, scale=0.75, thickness=2)

    y = header_h
    for camera, track_id, paths, track_dir in rows:
        frames = gid_info["cameras"][camera]["tracks"][track_id]["frames"]
        frame_text = ""
        if frames:
            frame_text = f" frames {min(frames)}-{max(frames)}"
        _put_label(sheet, f"{camera}", 14, y + 28, scale=0.6, thickness=2)
        _put_label(sheet, f"track {int(track_id):04d}{frame_text}", 14, y + 52)
        _put_label(sheet, os.path.relpath(track_dir, os.getcwd()), 14, y + 76, scale=0.42)

        x = left_w
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(thumb_w / max(1, w), thumb_h / max(1, h))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            tile = np.full((thumb_h, thumb_w, 3), 230, dtype=np.uint8)
            ox = (thumb_w - new_w) // 2
            oy = (thumb_h - new_h) // 2
            tile[oy:oy + new_h, ox:ox + new_w] = resized
            sheet[y:y + thumb_h, x:x + thumb_w] = tile
            _put_label(sheet, os.path.splitext(os.path.basename(path))[0],
                       x, y + thumb_h + 18, scale=0.42)
            x += thumb_w + pad

        y += row_h

    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, f"gid_{int(gid):04d}.jpg")
    cv2.imwrite(out_path, sheet)
    return out_path


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
    Print WHAT the run produced, so a headless run isn't a black box. Reads the
    ground truth back from the store (per-camera observations, distinct
    global_ids, cross-camera people) and lists the artifacts on disk.
    """
    from collections import defaultdict

    print("\n===== RUN SUMMARY =====")
    if run_id is not None:
        print(f"  Run id: {run_id}")

    # Per-camera crops on disk.
    crop_dir = cfg.get("crops", {}).get("dir", "crops")
    for name, *_ in jobs:
        cam_dir = os.path.join(crop_dir, name)
        if os.path.isdir(cam_dir):
            tracks = [d for d in os.listdir(cam_dir)
                      if os.path.isdir(os.path.join(cam_dir, d))]
            jpgs = sum(len(os.listdir(os.path.join(cam_dir, t))) for t in tracks)
            print(f"  {name}: {len(tracks)} tracks, {jpgs} crops -> {cam_dir}/")

    # Embeddings + identities, read back from the store.
    if store is not None:
        try:
            cam_obs = defaultdict(int)
            gid_info = defaultdict(lambda: {
                "observations": 0,
                "frames": [],
                "cameras": defaultdict(lambda: {
                    "observations": 0,
                    "tracks": defaultdict(lambda: {
                        "observations": 0,
                        "frames": [],
                    }),
                }),
            })
            offset = None
            while True:
                pts, offset = store.client.scroll(
                    store.collection, limit=1000, offset=offset,
                    with_payload=True, with_vectors=False)
                for p in pts:
                    pl = p.payload or {}
                    if run_id is not None and pl.get("run_id") != run_id:
                        continue
                    camera = pl.get("camera")
                    track_id = pl.get("track_id")
                    frame = pl.get("frame")
                    gid = pl.get("global_id")
                    cam_obs[camera] += 1
                    if gid is None:
                        continue

                    info = gid_info[gid]
                    info["observations"] += 1
                    if frame is not None:
                        info["frames"].append(frame)

                    cam_info = info["cameras"][camera]
                    cam_info["observations"] += 1

                    track_info = cam_info["tracks"][track_id]
                    track_info["observations"] += 1
                    if frame is not None:
                        track_info["frames"].append(frame)
                if offset is None:
                    break
            total = sum(cam_obs.values())
            cross_ids = sorted(
                gid for gid, info in gid_info.items()
                if len(info["cameras"]) > 1
            )
            print(f"  Store: {total} observations -> "
                  f"{len(gid_info)} distinct people (global_ids)")
            print(f"  Cross-camera people: {len(cross_ids)}")

            report_dir = os.path.join("reports", "cross_camera_matches")
            clear_directory(report_dir)
            if cross_ids:
                summary_path = os.path.join(report_dir, "summary.txt")
                report_lines = [
                    "Cross-camera matches",
                    "====================",
                    f"run_id: {run_id}",
                    "",
                ]

                crop_dir = cfg.get("crops", {}).get("dir", "crops")
                print("  Cross-camera match details:")
                for gid in cross_ids:
                    info = gid_info[gid]
                    cameras = sorted(info["cameras"])
                    if info["frames"]:
                        frame_span = f"{min(info['frames'])}-{max(info['frames'])}"
                    else:
                        frame_span = "unknown"
                    sheet_path = _make_cross_camera_contact_sheet(
                        gid, info, crop_dir, report_dir)
                    sheet_display = sheet_path if sheet_path else "(no crops found)"

                    header = (
                        f"    GID {gid}: {', '.join(cameras)}; "
                        f"{info['observations']} observations; frames {frame_span}"
                    )
                    print(header)
                    print(f"      contact sheet: {sheet_display}")
                    report_lines.append(header.strip())
                    report_lines.append(f"contact sheet: {sheet_display}")

                    for camera in cameras:
                        cam_info = info["cameras"][camera]
                        track_bits = []
                        for track_id in sorted(cam_info["tracks"],
                                               key=lambda v: (-1 if v is None else int(v))):
                            track_info = cam_info["tracks"][track_id]
                            frames = track_info["frames"]
                            if frames:
                                span = f"{min(frames)}-{max(frames)}"
                            else:
                                span = "unknown"
                            if track_id is None:
                                track_label = "track unknown"
                                crop_path = "(no crop folder)"
                            else:
                                track_label = f"track {int(track_id):04d}"
                                crop_path = _track_dir(crop_dir, camera, track_id)
                            track_bits.append(
                                f"{track_label} frames {span} "
                                f"({track_info['observations']} obs) -> {crop_path}"
                            )
                        line = f"      {camera}: " + "; ".join(track_bits)
                        print(line)
                        report_lines.append(line.strip())
                    report_lines.append("")

                with open(summary_path, "w") as f:
                    f.write("\n".join(report_lines))
                print(f"  Cross-camera report: {summary_path}")
            else:
                summary_path = os.path.join(report_dir, "summary.txt")
                with open(summary_path, "w") as f:
                    f.write(f"No cross-camera matches for run_id: {run_id}\n")
                print(f"  Cross-camera report: {summary_path}")
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


def normalize_sources(videos):
    """
    Turn the config's source.videos list into a clean list of (name, path).

    Each entry may be either:
      - a plain string path  -> name is derived from the file name, OR
      - a dict {name, path}  -> we use the given friendly name.
    """
    result = []
    for entry in videos:
        if isinstance(entry, dict):
            path = entry["path"]
            name = entry.get("name") or os.path.splitext(os.path.basename(path))[0]
        else:
            path = entry
            name = os.path.splitext(os.path.basename(path))[0]
        result.append((name, path))
    return result


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


# =============================================================================
# The WORKER: runs the whole pipeline for ONE video, on its own thread.
# It does NOT touch any window. It only:
#   - reads frames, detects+tracks, saves crops, draws boxes, and
#   - publishes its latest annotated frame into `shared` for the main thread.
# =============================================================================
def process_video(name, path, detector, crop_saver, embedder, store, identity,
                  locks, cfg, shared, stop_event, run_id):
    disp_cfg = cfg["display"]
    trk_cfg = cfg["tracker"]
    save_annotated = disp_cfg.get("save_annotated", False)
    # 0 = process the whole video; a positive number stops early (handy for a
    # quick CPU test without waiting for hundreds of frames).
    max_frames = cfg["source"].get("max_frames", 0)
    # 0 = keep native resolution. A positive width downscales every frame (keeping
    # aspect) BEFORE detect/track/crop/embed -- QHD footage on CPU is dominated by
    # pixel count, so e.g. 1280 is dramatically faster for dev with little ReID
    # cost. Everything downstream uses the same resized frame, so coords stay
    # consistent (crops are smaller too).
    resize_width = cfg["source"].get("resize_width", 0)

    tag = f"[{name}]"
    writer = None
    frame_index = 0
    stored_here = 0   # how many embeddings THIS camera has written to the store

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

                # ---- STAGE 5/6: embed tracked people + STORE in Qdrant -------
                # Uses the CLEAN frame (before boxes are drawn). We embed each
                # track (throttled + cached by TrackEmbedder) and persist only the
                # FRESH observations to the vector store. We deliberately do NOT
                # assign global_id here -- turning stored vectors into identities
                # is the Identity Service's job (Stage 7). global_id stays None.
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
                                d.global_id = identity.assign(
                                    name, d.track_id, d.embedding, frame_index,
                                    run_id=run_id,
                                    crop_quality=d.crop_quality)
                        stored_here += len(fresh)
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
                        stored_here += len(fresh)

                # ---- STAGE 5: draw (modifies frame in place) ----------------
                frame = draw_detections(frame, detections)
                frame = draw_hud(frame, person_count=len(detections))

                # ---- Save an annotated output video, if requested -----------
                if save_annotated:
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_path = f"output_{name}.mp4"
                        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                        print(f"{tag} Writing annotated video -> {out_path}")
                    writer.write(frame)

                # ---- Publish latest frame for the main thread to display -----
                # We store a COPY so the main thread can read it while we move
                # on to draw the next frame. The lock makes the swap safe.
                with shared["lock"]:
                    shared["frames"][name] = frame.copy()

                frame_index += 1

                # ---- Heartbeat: prove the pipeline is alive on long videos ---
                # Full videos on CPU take minutes; without this the terminal
                # looks frozen. Log progress every 30 frames.
                if frame_index % 30 == 0:
                    print(f"{tag} frame {frame_index}: {len(detections)} people, "
                          f"{stored_here} embeddings stored so far")

                # Stop early if a frame cap was set (for quick tests).
                if max_frames and frame_index >= max_frames:
                    print(f"{tag} Reached max_frames={max_frames}, stopping.")
                    break

    except Exception as e:
        # One video failing shouldn't kill the others -- log and move on.
        print(f"{tag} ERROR: {e}")
    finally:
        if writer is not None:
            writer.release()
        # Mark this camera as finished so the main thread knows.
        with shared["lock"]:
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

    while True:
        # Take a quick snapshot of the latest frames under the lock.
        with shared["lock"]:
            frames = dict(shared["frames"])
            done = set(shared["done"])

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

        # Exit once every video has finished. (We've already shown whatever
        # frames each produced; a video that failed instantly just won't have
        # a window.)
        if len(done) == total_videos:
            print("[display] All videos finished.")
            break

    # Keep the final frames up until a key is pressed, so nothing vanishes.
    if created and not stop_event.is_set():
        print("[display] Press any key in a window to close.")
        cv2.waitKey(0)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-camera person ReID pipeline.")
    p.add_argument(
        "--reset", action="store_true",
        help="Wipe the vector store collection before running, so a test starts "
             "from a clean gallery. DESTRUCTIVE -- deletes all stored embeddings.",
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

    sources = normalize_sources(cfg["source"]["videos"])
    if not sources:
        print("[main] No videos listed under source.videos in config.yaml.")
        return
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[main] Preparing {len(sources)} video(s): "
          f"{', '.join(n for n, _ in sources)}")
    print(f"[main] Run id: {run_id}")

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

    # ---- Build the SHARED ReID model + vector store -------------------------
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
        from identity.constraints import ConstraintGate
        gate = ConstraintGate.from_config(id_cfg.get("constraints"))
        identity = IdentityService(
            store,
            threshold=id_cfg.get("threshold", 0.85),
            top_k=id_cfg.get("top_k", 10),
            bank_size=id_cfg.get("bank_size", 20),
            min_score_gap=id_cfg.get("min_score_gap", 0.03),
            constraints=gate,
        )
        gate_msg = " (topology/timing gate ON)" if gate is not None else ""
        print(f"[main] Identity Service enabled -> assigning GLOBAL ids{gate_msg}.")

    # ---- Shared state between worker threads and the display loop ------------
    # frames: {name: latest annotated frame}   done: {names that finished}
    shared = {
        "lock": threading.Lock(),
        "frames": {},
        "done": set(),
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
    show_window = disp_cfg.get("show_window", True)
    if show_window:
        run_display([n for n, *_ in jobs], cfg, shared, stop_event, len(jobs))
    else:
        # Headless: nothing to show, just wait for all videos to finish.
        for t in threads:
            t.join()

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
        from identity.reconcile import reconcile_global_ids, reconcile_tracklets
        threshold = recon_cfg.get("threshold")
        if threshold is None:
            threshold = id_cfg.get("threshold", 0.85)
        print("[main] Reconciling identities across cameras...")
        if recon_cfg.get("tracklet_first", True):
            reconcile_tracklets(
                store,
                threshold=threshold,
                run_id=run_id,
                same_camera_threshold=recon_cfg.get("same_camera_threshold", 0.90),
                require_reciprocal_best=recon_cfg.get("require_reciprocal_best", True),
                min_tracklet_observations=recon_cfg.get("min_tracklet_observations", 1),
            )
        else:
            reconcile_global_ids(
                store,
                threshold=threshold,
                run_id=run_id,
                block_same_camera_overlap=recon_cfg.get("block_same_camera_overlap", True),
                require_reciprocal_best=recon_cfg.get("require_reciprocal_best", True),
            )

    print_run_summary(store, jobs, cfg, run_id=run_id)
    print("[main] All done.")


if __name__ == "__main__":
    # This guard means main() only runs when you execute `python main.py`,
    # and is REQUIRED for threads to behave correctly.
    main()
