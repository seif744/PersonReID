"""
model_separability.py  --  isolate the ReID MODEL from the pipeline.

PURPOSE
Answer one question with evidence: when identities come out wrong, is it a code
/ threshold bug, or is the appearance model itself unable to separate these
people? This script removes EVERYTHING except the model -- no vector store, no
Identity Service, no reconciliation, no thresholds, no live/threading path. It
reads ground-truth crop folders straight off disk, embeds them with the same
extractor the pipeline uses, and measures the raw embeddings.

It runs three checks:

  CONTROL A -- inference is correct.
      Split one tracklet's own crops into two halves and compare. The same person
      through the same code should score ~0.95+. If it does, preprocessing,
      normalization, and the forward pass are all working -- the pipeline's
      embedding code is NOT the bug.

  CONTROL B -- embeddings are non-degenerate.
      Report the spread of gallery scores. If everything is ~1.0 the model
      collapsed; a healthy spread means it IS discriminating something.

  THE TEST -- threshold-independent retrieval.
      Take a probe tracklet (a person in camera B), rank every tracklet in
      camera A by similarity, and ask: is the KNOWN same person ranked #1?
      Ranking uses no threshold at all. If the true match is not rank-1 -- i.e. a
      DIFFERENT person scores higher -- then no threshold, reconciler, or
      constraint can fix it, because the model put the wrong answer on top. That
      is, by definition, a model failure, not a code failure.

Nothing is hardcoded: probe, gallery, and the ground-truth identity come from
argv. Run from the project root, e.g.:

    python tools/model_separability.py \
        --probe crops/cam_224/id_0001 \
        --gallery crops/cam_219 \
        --truth id_0004
"""

import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cv2  # noqa: E402
from reid.extractor import ReIDExtractor  # noqa: E402


def load_crops(folder):
    files = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    crops = [img for img in (cv2.imread(p) for p in files) if img is not None and img.size]
    return crops


def prototype(embs):
    p = embs.mean(0)
    return p / (np.linalg.norm(p) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="probe tracklet folder (person in camera B)")
    ap.add_argument("--gallery", required=True, help="gallery camera folder containing id_* tracklets")
    ap.add_argument("--truth", required=True, help="the gallery tracklet folder name that is the SAME person")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rc = cfg.get("reid", {})
    ex = ReIDExtractor(weights=rc["weights"], device=rc.get("device"))

    def embed(folder):
        crops = load_crops(folder)
        if not crops:
            return None
        return ex.extract_batch(crops)   # (N, 512), L2-normalized

    probe = embed(args.probe)
    if probe is None:
        print(f"ERROR: no crops in {args.probe}")
        return
    probe_proto = prototype(probe)

    galleries = sorted(
        d for d in glob.glob(os.path.join(args.gallery, "id_*")) if os.path.isdir(d))
    gnames = {os.path.basename(d): d for d in galleries}
    if args.truth not in gnames:
        print(f"ERROR: truth '{args.truth}' not found in {args.gallery}")
        return

    # ---- CONTROL A: split-half self consistency (is inference correct?) -------
    print("\n[CONTROL A] Split-half self-similarity (should be ~0.95+ if the code is correct)")
    for label, embs in (("probe " + os.path.basename(args.probe), probe),
                        ("truth " + args.truth, embed(gnames[args.truth]))):
        if embs is None or len(embs) < 2:
            print(f"  {label}: too few crops")
            continue
        mid = len(embs) // 2
        h1, h2 = prototype(embs[:mid]), prototype(embs[mid:])
        print(f"  {label}: {float(h1 @ h2):.3f}   ({len(embs)} crops)")

    # ---- Retrieval: probe vs every gallery tracklet (NO THRESHOLD) ------------
    scored = []
    for name, folder in gnames.items():
        embs = embed(folder)
        if embs is None:
            continue
        scored.append((float(probe_proto @ prototype(embs)), name))
    scored.sort(reverse=True)

    print("\n[CONTROL B] Gallery score spread (healthy = a real range, not all ~1.0)")
    vals = [s for s, _ in scored]
    print(f"  min {min(vals):.3f}  max {max(vals):.3f}  spread {max(vals)-min(vals):.3f}")

    print(f"\n[THE TEST] Rank cam-A tracklets by similarity to probe '{os.path.basename(args.probe)}':")
    truth_rank = None
    for i, (s, name) in enumerate(scored, 1):
        mark = "  <== TRUE SAME PERSON (ground truth)" if name == args.truth else ""
        if name == args.truth:
            truth_rank = i
        print(f"  #{i}  {name}: {s:.3f}{mark}")

    truth_score = dict((n, s) for s, n in scored)[args.truth]
    top_score, top_name = scored[0]

    print("\n===== VERDICT =====")
    print(f"  True match '{args.truth}' ranked #{truth_rank} of {len(scored)} (score {truth_score:.3f}).")
    if truth_rank == 1:
        runner = scored[1]
        print(f"  Rank-1 is CORRECT. Margin over #2 ({runner[1]}): {truth_score - runner[0]:+.3f}")
        print("  -> Model puts the right answer on top; identity errors are a threshold/logic")
        print("     matter, tunable in code.")
    else:
        print(f"  Rank-1 is WRONG: '{top_name}' (a DIFFERENT person) scores {top_score:.3f} "
              f"> true {truth_score:.3f}.")
        # How many different-identity tracklets outrank or tie the true match?
        confusers = [(s, n) for s, n in scored if n != args.truth and s >= truth_score]
        print(f"  {len(confusers)} different-identity tracklet(s) score >= the true match.")
        print("  -> No threshold separates true from false here: any cutoff low enough to accept")
        print(f"     the true match ({truth_score:.3f}) also accepts higher-scoring wrong people.")
        print("     This is a MODEL failure (embeddings rank the wrong identity first), NOT a")
        print("     code or threshold bug. Reconciler/constraints operate on these same numbers.")
    print("===================\n")


if __name__ == "__main__":
    main()
