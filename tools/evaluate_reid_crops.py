"""
Evaluate a ReID checkpoint on labeled crop images.

Two input formats are supported:

1. Directory labels:
   root/person_id/*.jpg
   root/camera/person_id/*.jpg

2. CSV manifest:
   path,person_id,camera

Run examples:
   python tools/evaluate_reid_crops.py --root labeled_crops
   python tools/evaluate_reid_crops.py --manifest labels.csv --cross-camera-only
   python tools/evaluate_reid_crops.py --weights path/to/model.pth --root labeled_crops
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from reid.extractor import ReIDExtractor


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "src", "reid", "weights", "osnet_x1_0_market1501.pth")


def load_manifest(path):
    items = []
    base = os.path.dirname(os.path.abspath(path))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "person_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            img_path = row["path"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(base, img_path)
            items.append({
                "path": img_path,
                "person_id": row["person_id"],
                "camera": row.get("camera") or "",
            })
    return items


def load_from_root(root):
    items = []
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        images = [
            f for f in filenames
            if os.path.splitext(f.lower())[1] in IMAGE_EXTS
        ]
        if not images:
            continue
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        if len(parts) == 1:
            camera, person_id = "", parts[0]
        elif len(parts) >= 2:
            camera, person_id = parts[-2], parts[-1]
        else:
            continue
        for filename in images:
            items.append({
                "path": os.path.join(dirpath, filename),
                "person_id": person_id,
                "camera": camera,
            })
    return items


def load_embeddings(items, extractor, batch_size):
    valid = []
    crops = []
    for item in items:
        img = cv2.imread(item["path"])
        if img is None:
            continue
        valid.append(item)
        crops.append(img)

    chunks = []
    for i in range(0, len(crops), batch_size):
        chunks.append(extractor.extract_batch(crops[i:i + batch_size]))
    if chunks:
        embeddings = np.concatenate(chunks, axis=0)
    else:
        embeddings = np.empty((0, 512), dtype=np.float32)
    return valid, embeddings


def average_precision(y_true, scores):
    order = np.argsort(-scores)
    y = y_true[order].astype(bool)
    if not y.any():
        return None
    hits = np.cumsum(y)
    ranks = np.arange(1, len(y) + 1)
    return float((hits[y] / ranks[y]).mean())


def retrieval_metrics(items, embeddings, cross_camera_only):
    labels = np.array([x["person_id"] for x in items])
    cameras = np.array([x["camera"] for x in items])
    sims = embeddings @ embeddings.T

    rank1_hits = []
    aps = []
    usable = 0
    for i in range(len(items)):
        valid = np.ones(len(items), dtype=bool)
        valid[i] = False
        if cross_camera_only:
            valid &= cameras != cameras[i]
        positives = valid & (labels == labels[i])
        if not positives.any():
            continue

        usable += 1
        candidate_idx = np.where(valid)[0]
        scores = sims[i, candidate_idx]
        y_true = labels[candidate_idx] == labels[i]
        best = candidate_idx[int(np.argmax(scores))]
        rank1_hits.append(labels[best] == labels[i])
        ap = average_precision(y_true, scores)
        if ap is not None:
            aps.append(ap)

    return {
        "queries": usable,
        "rank1": float(np.mean(rank1_hits)) if rank1_hits else 0.0,
        "map": float(np.mean(aps)) if aps else 0.0,
    }


def threshold_sweep(items, embeddings, thresholds, cross_camera_only):
    labels = np.array([x["person_id"] for x in items])
    cameras = np.array([x["camera"] for x in items])
    sims = embeddings @ embeddings.T
    iu = np.triu_indices(len(items), k=1)
    pair_scores = sims[iu]
    same = labels[iu[0]] == labels[iu[1]]
    if cross_camera_only:
        keep = cameras[iu[0]] != cameras[iu[1]]
        pair_scores = pair_scores[keep]
        same = same[keep]

    rows = []
    for threshold in thresholds:
        pred = pair_scores >= threshold
        tp = int((pred & same).sum())
        fp = int((pred & ~same).sum())
        fn = int((~pred & same).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append((threshold, precision, recall, f1, tp, fp, fn))
    return rows


def print_dataset_summary(items):
    people = defaultdict(int)
    cameras = defaultdict(int)
    for item in items:
        people[item["person_id"]] += 1
        cameras[item["camera"]] += 1
    print(f"images: {len(items)}")
    print(f"people: {len(people)}")
    print(f"cameras: {len(cameras)}")
    for cam, count in sorted(cameras.items()):
        name = cam or "(unknown)"
        print(f"  {name}: {count} images")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="Labeled crop root directory.")
    src.add_argument("--manifest", help="CSV with path,person_id,camera columns.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cross-camera-only", action="store_true")
    parser.add_argument("--thresholds", default="0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    args = parser.parse_args()

    items = load_manifest(args.manifest) if args.manifest else load_from_root(args.root)
    if not items:
        raise SystemExit("No labeled images found.")

    print_dataset_summary(items)
    extractor = ReIDExtractor(weights=args.weights, device=args.device)
    items, embeddings = load_embeddings(items, extractor, args.batch_size)
    if len(items) < 2:
        raise SystemExit("Need at least two readable images.")

    print(f"\ndevice: {extractor.device}")
    mode = "cross-camera only" if args.cross_camera_only else "all valid pairs"
    print(f"mode: {mode}\n")

    metrics = retrieval_metrics(items, embeddings, args.cross_camera_only)
    print("retrieval:")
    print(f"  queries: {metrics['queries']}")
    print(f"  rank-1:  {metrics['rank1']:.3f}")
    print(f"  mAP:     {metrics['map']:.3f}")

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    print("\nthreshold sweep:")
    print(f"{'thr':>6} {'prec':>7} {'recall':>7} {'f1':>7} {'tp':>6} {'fp':>6} {'fn':>6}")
    for threshold, precision, recall, f1, tp, fp, fn in threshold_sweep(
            items, embeddings, thresholds, args.cross_camera_only):
        print(f"{threshold:>6.2f} {precision:>7.3f} {recall:>7.3f} "
              f"{f1:>7.3f} {tp:>6} {fp:>6} {fn:>6}")


if __name__ == "__main__":
    main()
