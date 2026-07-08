"""
demo_store_search.py  --  STAGE 6 end-to-end:  real crops -> Qdrant -> search.

============================ WHAT THIS PROVES ==============================
The full storage+retrieval path on YOUR real crops:

    crop --(ReIDExtractor)--> embedding --(PersonVectorStore.add)--> Qdrant
    held-out crop --> embedding --> PersonVectorStore.search --> ranked matches

We treat each crops/<cam>/<id_XXXX> folder as one identity (same clean ground
truth as test_reid.py). For each identity we STORE most crops and HOLD OUT a few
as queries. A working system, given a held-out query, returns that same person's
other crops at the TOP.

We report retrieval accuracy:
  * top-1  = the single closest stored crop is the same person
  * top-5  = the same person appears among the 5 closest

Run from src/:   python -m database.demo_store_search

Note: this uses a PERSISTENT store at ../qdrant_data and calls reset() at the
start so reruns are clean. A real system would never reset() -- it would keep
appending observations over days/weeks.
============================================================================
"""

import glob
import os

import cv2

from reid.extractor import ReIDExtractor
from database.store import PersonVectorStore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT_CROPS = os.path.join(PROJECT_ROOT, "crops")
LEGACY_CROPS = os.path.join(os.path.dirname(__file__), "..", "reid", "crops")
if os.path.isdir(ROOT_CROPS):
    CROPS_ROOT = ROOT_CROPS
else:
    CROPS_ROOT = LEGACY_CROPS
WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "reid", "weights",
                       "osnet_x1_0_market1501.pth")
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "qdrant_data")

HOLD_OUT = 3   # crops per identity reserved as queries (the rest are stored)


def main():
    extractor = ReIDExtractor(weights=WEIGHTS)
    store = PersonVectorStore(path=QDRANT_PATH)
    store.reset()                       # clean slate for a repeatable demo
    print(f"store ready at {QDRANT_PATH} (count={store.count()})\n")

    queries = []   # (identity_label, embedding) held out for searching

    # ---- POPULATE: embed every crop, store most, hold out a few per identity --
    for cam in sorted(os.listdir(CROPS_ROOT)):
        cam_dir = os.path.join(CROPS_ROOT, cam)
        if not os.path.isdir(cam_dir):
            continue
        for ident in sorted(os.listdir(cam_dir)):
            files = sorted(glob.glob(os.path.join(cam_dir, ident, "*.jpg")))
            if len(files) <= HOLD_OUT:
                continue                # too few to both store and query
            label = f"{cam}/{ident}"

            crops = [cv2.imread(f) for f in files]
            embs = extractor.extract_batch(crops)   # (N, 512)

            # store all but the last HOLD_OUT; hold those out as queries
            store_embs = embs[:-HOLD_OUT]
            store_files = files[:-HOLD_OUT]
            payloads = [{"camera": cam, "identity": label, "file": os.path.basename(f)}
                        for f in store_files]
            store.add_many(store_embs, payloads)

            for e in embs[-HOLD_OUT:]:
                queries.append((label, e))

    print(f"stored {store.count()} crops across identities; "
          f"holding out {len(queries)} query crops\n")

    # ---- ONE concrete example so you can see a real search result ------------
    ex_label, ex_emb = queries[0]
    print(f"EXAMPLE query is really '{ex_label}'. Top-3 matches Qdrant returns:")
    for h in store.search(ex_emb, limit=3):
        mark = "OK" if h.payload["identity"] == ex_label else "  "
        print(f"   [{mark}] score={h.score:.3f}  {h.payload['identity']}  "
              f"({h.payload['file']})")

    # ---- AGGREGATE: retrieval accuracy over all held-out queries -------------
    top1 = top5 = 0
    for label, emb in queries:
        hits = store.search(emb, limit=5)
        labels = [h.payload["identity"] for h in hits]
        if labels and labels[0] == label:
            top1 += 1
        if label in labels:
            top5 += 1

    n = len(queries)
    print(f"\nRETRIEVAL ACCURACY over {n} queries:")
    print(f"   top-1: {top1}/{n} = {top1/n:.1%}   (closest crop is the right person)")
    print(f"   top-5: {top5}/{n} = {top5/n:.1%}   (right person in the 5 closest)")


if __name__ == "__main__":
    main()
