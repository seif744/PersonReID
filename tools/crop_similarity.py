"""
crop_similarity.py -- measure appearance similarity between two crop folders.

Given two directories of person crops (e.g. one track per camera), embed every
crop with the SAME ReID extractor the pipeline uses, then report how similar the
two sets are. Answers: "would the Identity Service have matched these as the same
person, and at what threshold?"

Nothing is hardcoded: folders come from argv, weights/device come from config.yaml
(overridable via flags). Run from the project root:

    python tools/crop_similarity.py crops/cam_219/id_0004 crops/cam_224/id_0001
"""

import argparse
import os
import sys

import numpy as np
import yaml

# Same import convention as main.py: src is the source root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cv2  # noqa: E402
from reid.extractor import ReIDExtractor  # noqa: E402


def load_crops(folder):
    """Read every image in a folder as a BGR crop. Returns (paths, crops)."""
    exts = (".jpg", ".jpeg", ".png")
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )
    paths, crops = [], []
    for p in files:
        img = cv2.imread(p)
        if img is not None and img.size > 0:
            paths.append(p)
            crops.append(img)
    return paths, crops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder_a", help="first crop folder (e.g. crops/cam_219/id_0004)")
    ap.add_argument("folder_b", help="second crop folder (e.g. crops/cam_224/id_0001)")
    ap.add_argument("--config", default="config.yaml", help="config to read reid weights/device from")
    ap.add_argument("--weights", default=None, help="override reid weights path")
    ap.add_argument("--device", default=None, help="override device (cpu/cuda)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    reid_cfg = cfg.get("reid", {})
    weights = args.weights or reid_cfg["weights"]
    device = args.device or reid_cfg.get("device")

    print(f"Loading ReID model: {weights} (device={device})")
    extractor = ReIDExtractor(weights=weights, device=device)

    results = {}
    for label, folder in (("A", args.folder_a), ("B", args.folder_b)):
        paths, crops = load_crops(folder)
        if not crops:
            print(f"ERROR: no readable crops in {folder}")
            return
        embs = extractor.extract_batch(crops)  # (N, 512), already L2-normalized
        results[label] = (folder, paths, embs)
        print(f"  {label}: {folder} -> {len(crops)} crops embedded")

    (fa, _, ea), (fb, _, eb) = results["A"], results["B"]

    # Every A-crop vs every B-crop cosine (dot product of unit vectors).
    cross = ea @ eb.T                       # (Na, Nb)
    # Prototype = mean embedding per set (what the Identity Service's bank scores).
    proto_a = ea.mean(0); proto_a /= (np.linalg.norm(proto_a) + 1e-12)
    proto_b = eb.mean(0); proto_b /= (np.linalg.norm(proto_b) + 1e-12)
    proto_sim = float(proto_a @ proto_b)

    # Within-set spread, for context on how tight each track's own crops are.
    within_a = (ea @ ea.T)[np.triu_indices(len(ea), k=1)]
    within_b = (eb @ eb.T)[np.triu_indices(len(eb), k=1)]

    print("\n===== CROSS-CAMERA SIMILARITY =====")
    print(f"  A: {fa}")
    print(f"  B: {fb}")
    print(f"\n  cross-set cosine (all {cross.size} pairs):")
    print(f"      min   {cross.min():.3f}")
    print(f"      mean  {cross.mean():.3f}")
    print(f"      max   {cross.max():.3f}")
    print(f"  prototype-vs-prototype cosine: {proto_sim:.3f}   <-- bank-score analog")
    if within_a.size:
        print(f"\n  within-A cosine  mean {within_a.mean():.3f} (min {within_a.min():.3f})")
    if within_b.size:
        print(f"  within-B cosine  mean {within_b.mean():.3f} (min {within_b.min():.3f})")

    thr = reid_cfg and cfg.get("identity", {}).get("threshold", 0.80)
    print(f"\n  identity threshold in config: {thr}")
    verdict_score = max(proto_sim, float(cross.mean()))
    if proto_sim >= thr or cross.max() >= thr:
        print("  -> Would likely MATCH at some frame (a pair clears the threshold).")
    else:
        print("  -> Would NOT match: no pair reaches the threshold. This true pair is missed.")
    print("===================================\n")


if __name__ == "__main__":
    main()
