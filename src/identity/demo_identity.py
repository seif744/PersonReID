"""
demo_identity.py  --  STAGE 7 test:  does the Identity Service re-identify?

============================ WHAT THIS PROVES ==============================
The identity DECISION, measured HONESTLY against true identities.

We have 7 real people (each crops/<cam>/<id_XXXX> folder). We feed each person's
crops as an ENROLL track, then their held-out crops as NEW re-appearance tracks,
and see how the service clusters them into global_ids.

A PERFECT result = exactly 7 global_ids, each containing exactly one real person.
Two failure modes:
  * MERGE  -> one global_id ends up containing 2+ real people (different people
             fused into one identity). Corrupts histories. The worst error.
  * SPLIT  -> one real person spread across 2+ global_ids (same person given
             several identities). A miss; recoverable.

We SWEEP the decision threshold to show the core lesson: because same-person and
different-person cosine distributions OVERLAP (Phase 4: same-min 0.51 <
diff-max 0.85), NO single threshold gives clean 7-and-only-7. Low threshold ->
merges; high threshold -> splits. This is precisely why appearance alone is not
enough and the camera-topology / temporal / motion constraints (DESIGN.md) exist.

Run from src/:   python -m identity.demo_identity
============================================================================
"""

import glob
import os
from collections import defaultdict

import cv2

from reid.extractor import ReIDExtractor
from database.store import PersonVectorStore
from identity.service import IdentityService

CROPS_ROOT = os.path.join(os.path.dirname(__file__), "..", "reid", "crops")
WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "reid", "weights",
                       "osnet_x1_0_market1501.pth")
HOLD_OUT = 3
THRESHOLDS = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


def load_identities(extractor):
    out = {}
    for cam in sorted(os.listdir(CROPS_ROOT)):
        cam_dir = os.path.join(CROPS_ROOT, cam)
        if not os.path.isdir(cam_dir):
            continue
        for ident in sorted(os.listdir(cam_dir)):
            files = sorted(glob.glob(os.path.join(cam_dir, ident, "*.jpg")))
            if len(files) <= HOLD_OUT:
                continue
            crops = [cv2.imread(f) for f in files]
            out[f"{cam}/{ident}"] = {"cam": cam, "emb": extractor.extract_batch(crops)}
    return out


def evaluate(identities, threshold):
    """Run the full enroll+query flow at one threshold; return cluster metrics."""
    store = PersonVectorStore(path=":memory:")
    service = IdentityService(store, threshold=threshold)

    gid_to_people = defaultdict(set)   # global_id -> set of TRUE person labels
    q_track = 100000

    for track_num, (label, data) in enumerate(identities.items()):
        embs = data["emb"]
        # enroll: all crops but the held-out ones, as ONE track
        for e in embs[:-HOLD_OUT]:
            gid = service.assign(data["cam"], track_num, e, frame_index=0)
            gid_to_people[gid].add(label)
        # query: each held-out crop as its OWN brand-new track (a re-appearance)
        for e in embs[-HOLD_OUT:]:
            q_track += 1
            gid = service.assign(data["cam"], q_track, e, frame_index=0)
            gid_to_people[gid].add(label)

    num_ids = len(gid_to_people)
    merged_ids = sum(1 for people in gid_to_people.values() if len(people) > 1)

    people_to_gids = defaultdict(set)
    for gid, people in gid_to_people.items():
        for p in people:
            people_to_gids[p].add(gid)
    split_people = sum(1 for gids in people_to_gids.values() if len(gids) > 1)

    return num_ids, merged_ids, split_people


def main():
    extractor = ReIDExtractor(weights=WEIGHTS)
    identities = load_identities(extractor)
    n_people = len(identities)
    print(f"{n_people} real people. PERFECT result = {n_people} global_ids, "
          f"0 merges, 0 splits.\n")

    print(f"{'threshold':>10} {'global_ids':>11} {'merged_ids':>11} {'split_people':>13}")
    for t in THRESHOLDS:
        num_ids, merged, split = evaluate(identities, t)
        flag = "  <- perfect" if (num_ids == n_people and merged == 0 and split == 0) else ""
        print(f"{t:>10.2f} {num_ids:>11} {merged:>11} {split:>13}{flag}")

    print("\nRead it: low threshold -> few ids, many MERGES (different people fused).")
    print("High threshold -> many ids, SPLITS (one person = several ids).")
    print("The overlap means appearance alone can't win cleanly -> constraints needed.")


if __name__ == "__main__":
    main()
