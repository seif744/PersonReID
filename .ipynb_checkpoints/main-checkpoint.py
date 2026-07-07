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

import os
import threading
import yaml   # reads our config.yaml (installed alongside ultralytics)
import cv2

from src.video_source import VideoSource
from src.detector import PersonDetector
from src.drawing import draw_detections, draw_hud
from src.crop_saver import CropSaver


def load_config(path="config.yaml"):
    """Read config.yaml into a plain Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


# =============================================================================
# The WORKER: runs the whole pipeline for ONE video, on its own thread.
# It does NOT touch any window. It only:
#   - reads frames, detects+tracks, saves crops, draws boxes, and
#   - publishes its latest annotated frame into `shared` for the main thread.
# =============================================================================
def process_video(name, path, detector, crop_saver, reid_service, cfg, shared, stop_event):
    disp_cfg = cfg["display"]
    trk_cfg = cfg["tracker"]
    save_annotated = disp_cfg.get("save_annotated", False)
    # 0 = process the whole video; a positive number stops early (handy for a
    # quick CPU test without waiting for hundreds of frames).
    max_frames = cfg["source"].get("max_frames", 0)

    tag = f"[{name}]"
    writer = None
    frame_index = 0

    try:
        with VideoSource(path=path) as cam:
            print(f"{tag} Pipeline running on '{path}'.")
            for frame in cam.frames():        # STAGE 1-2: one frame at a time

                # Stop early if the user pressed 'q' in the main thread.
                if stop_event.is_set():
                    break

                # ---- STAGE 3 (+4): detect, with or without track IDs --------
                if trk_cfg["enabled"]:
                    detections = detector.track(frame)   # Stage 4: with IDs
                else:
                    detections = detector.detect(frame)  # Stage 3: no IDs

                # ---- STAGE 6 prep: crop from the CLEAN frame (before drawing)-
                if crop_saver is not None:
                    crop_saver.save(frame, detections, frame_index)

                # ---- STAGE 6: ReID -> assign a GLOBAL id to each person ------
                # Uses the CLEAN frame too (same reason as crops). The service is
                # SHARED across cameras, so ids are consistent everywhere.
                if reid_service is not None:
                    for det in detections:
                        det.global_id = reid_service.identify(
                            name, det, frame, frame_index
                        )

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


def main():
    # ---- Load settings ------------------------------------------------------
    cfg = load_config()
    det_cfg = cfg["detector"]
    trk_cfg = cfg["tracker"]
    crop_cfg = cfg["crops"]
    disp_cfg = cfg["display"]

    sources = normalize_sources(cfg["source"]["videos"])
    if not sources:
        print("[main] No videos listed under source.videos in config.yaml.")
        return

    print(f"[main] Preparing {len(sources)} video(s): "
          f"{', '.join(n for n, _ in sources)}")

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

    # ---- Build ONE ReID service, SHARED by every camera ---------------------
    # Global ids are only "global" if all cameras assign from the same gallery,
    # so there is exactly one extractor + one gallery for the whole run.
    reid_service = None
    reid_cfg = cfg.get("reid", {})
    if reid_cfg.get("enabled"):
        from src.reid import ReIDExtractor, ReIDService
        print("[main] Loading ReID model (shared across all cameras)...")
        extractor = ReIDExtractor(
            config_file=reid_cfg["config_file"],
            weights_path=reid_cfg.get("weights"),   # None -> ImageNet baseline
            device=reid_cfg.get("device", "cpu"),
        )
        reid_service = ReIDService(
            extractor,
            threshold=reid_cfg.get("threshold", 0.5),
            ema_alpha=reid_cfg.get("ema_alpha", 0.2),
            interval=reid_cfg.get("interval", 10),
        )
        print("[main] ReID enabled -> assigning GLOBAL ids across cameras.")

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
            args=(name, path, detector, crop_saver, reid_service, cfg, shared, stop_event),
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
    print("[main] All done.")


if __name__ == "__main__":
    # This guard means main() only runs when you execute `python main.py`,
    # and is REQUIRED for threads to behave correctly.
    main()
