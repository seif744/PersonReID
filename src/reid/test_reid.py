"""
test_reid.py  --  STAGE 5 validation:  do the embeddings SEPARATE people?

============================ WHAT THIS PROVES ==============================
Phase 3 proved the plumbing (shape (512,), float32, L2 norm == 1). That is
necessary but NOT sufficient: an ImageNet-pretrained backbone also produces a
unit-norm 512-d vector -- it just doesn't tell people apart. This script proves
the thing that actually matters: crops of the SAME person are close, crops of
DIFFERENT people are far.

------------------------------- THE METRIC --------------------------------
Embeddings are L2-normalized, so cosine similarity == dot product, in [-1, 1]:
    1.0  = identical direction (very similar appearance)
    0.0  = orthogonal (unrelated)
In practice ReID sims are positive (crops share "person-ness"); what matters is
the GAP between the same-person and different-person distributions, not their
absolute values.

--------------------------- GROUND TRUTH & HONESTY ------------------------
The crop folders are named by PER-CAMERA track id (ByteTrack). So:
  * SAME-person pairs  = two crops inside the same id_XXXX folder.  (trusted)
  * DIFF-person pairs  = two crops from different folders OF THE SAME CAMERA.
We do NOT form cross-camera pairs: we have no cross-camera identity labels, so a
cross-camera "different" pair might secretly be the same human. Resolving that
is the Identity Service's job (Phase 7), not this test's. Keeping the test
intra-camera keeps the ground truth clean.

------------------------------ HOW TO READ IT -----------------------------
A healthy osnet_x1_0 (Market-1501) on decent crops should show:
    same-person mean cosine   ~0.6 - 0.9
    diff-person mean cosine   ~0.1 - 0.4
    a clear gap between the two means (roughly >0.2)
The old ImageNet-ResNet50 baseline had a gap of ~0.13 and overlapping tails --
that's the failure we're checking we no longer have. Real-world crops (occlusion,
motion blur, pose) will always leave SOME overlap in the tails; perfect
separation is not expected and not required.
============================================================================
"""

import glob
import os
from itertools import combinations

import cv2
import numpy as np

from reid.extractor import ReIDExtractor

CROPS_ROOT = os.path.join(os.path.dirname(__file__), "crops")
WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "osnet_x1_0_market1501.pth")


def load_embeddings_by_identity(extractor):
    """
    Walk crops/<cam>/<id_XXXX>/*.jpg and return:
        { camera_name: { identity_name: (N, 512) embeddings } }
    Embeddings are computed per-folder in one batched forward pass.
    """
    result = {}
    for cam in sorted(os.listdir(CROPS_ROOT)):
        cam_dir = os.path.join(CROPS_ROOT, cam)
        if not os.path.isdir(cam_dir):
            continue
        result[cam] = {}
        for ident in sorted(os.listdir(cam_dir)):
            id_dir = os.path.join(cam_dir, ident)
            if not os.path.isdir(id_dir):
                continue
            files = sorted(glob.glob(os.path.join(id_dir, "*.jpg")))
            crops = [cv2.imread(f) for f in files]
            crops = [c for c in crops if c is not None]   # skip unreadable files
            if crops:
                result[cam][ident] = extractor.extract_batch(crops)
    return result


def summarize(name, sims):
    """Print count / mean / std / min / max for a similarity distribution."""
    sims = np.asarray(sims)
    print(f"  {name:26s} n={sims.size:6d}  "
          f"mean={sims.mean():+.3f}  std={sims.std():.3f}  "
          f"min={sims.min():+.3f}  max={sims.max():+.3f}")
    return sims


def main():
    extractor = ReIDExtractor(weights=WEIGHTS)
    print(f"device: {extractor.device}\n")

    embs = load_embeddings_by_identity(extractor)

    # ---- Phase 4 explicit sanity print: shape / dtype / norm of one vector ----
    first_cam = next(iter(embs))
    first_id = next(iter(embs[first_cam]))
    sample = embs[first_cam][first_id][0]
    print("SANITY  shape:", sample.shape,
          " dtype:", sample.dtype,
          f" L2 norm: {np.linalg.norm(sample):.6f}\n")

    same_sims, diff_sims = [], []

    # Cosine == dot product because vectors are unit-norm. We do all pairwise
    # comparisons INSIDE each camera to keep the ground truth honest.
    for cam, identities in embs.items():
        # SAME-person: every unordered pair within a folder.
        for ident, X in identities.items():
            S = X @ X.T                        # (N, N) cosine matrix
            iu = np.triu_indices(len(X), k=1)  # upper triangle, no diagonal
            same_sims.extend(S[iu].tolist())

        # DIFF-person: every crop of id A vs every crop of id B, A != B, same cam.
        for id_a, id_b in combinations(identities, 2):
            Sab = embs[cam][id_a] @ embs[cam][id_b].T
            diff_sims.extend(Sab.ravel().tolist())

    print("COSINE SIMILARITY DISTRIBUTIONS (intra-camera):")
    same = summarize("same-person (within id)", same_sims)
    diff = summarize("diff-person (across ids)", diff_sims)

    gap = same.mean() - diff.mean()
    print(f"\n  separation (mean gap): {gap:+.3f}")

    # A single-number "how separable" score: the fraction of correctly ordered
    # random (same, diff) pairs == AUC of same-vs-diff. 1.0 = perfect, 0.5 = chance.
    # Computed on a capped sample so it stays fast regardless of dataset size.
    rng = np.random.default_rng(0)
    a = rng.choice(same, size=min(20000, same.size))
    b = rng.choice(diff, size=min(20000, diff.size))
    auc = float((a[:, None] > b[None, :]).mean())
    print(f"  separability (AUC same>diff): {auc:.3f}")

    # ---- Verdict -----------------------------------------------------------
    # These thresholds are a smoke gate, not a research metric. A working ReID
    # backbone on real crops clears them comfortably; the ImageNet baseline did
    # not (gap ~0.13, AUC ~0.6).
    ok = gap > 0.20 and auc > 0.85
    print("\nRESULT:", "PASS -- embeddings separate identities" if ok
          else "FAIL -- weak separation, investigate (weights? preprocessing?)")


if __name__ == "__main__":
    main()
