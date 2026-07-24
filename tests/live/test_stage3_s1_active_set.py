"""
Stage 3 -- Substage 1 validation: active identity set + prototype/medoid storage,
evidence gate, conservative mint, sticky-once-assigned.

Acceptance (this substage):
  * A track shows a PROVISIONAL (negative) id until it has min_evidence_obs views.
  * Continuous track retains ONE positive ReID once resolved (Stage-3 criterion 1).
  * The active set stores a usable prototype/medoid: score(same person) high,
    score(different person) low.
  * Substage-1 matching is mint-only: two DISTINCT tracks get DISTINCT ids.
  * Frames with no fresh embedding keep the sticky id (and don't create new ids).
"""

from _synth import Check, person, observe
from live.identity_engine import IdentityEngine


def main():
    c = Check("Stage3-S1 active-set / prototype-medoid / evidence-gate")
    eng = IdentityEngine(min_evidence_obs=3)

    alice = person(1)

    # --- evidence gate: provisional (negative) until min_obs views ----------
    id1 = eng.assign("cam_a", 10, observe(alice, seed=100), None, ts=1.0, has_fresh_emb=True)
    id2 = eng.assign("cam_a", 10, observe(alice, seed=101), None, ts=1.1, has_fresh_emb=True)
    c.ok(id1 < 0 and id2 < 0, "track is PROVISIONAL (negative id) before the gate")
    id3 = eng.assign("cam_a", 10, observe(alice, seed=102), None, ts=1.2, has_fresh_emb=True)
    c.ok(id3 > 0, "track RESOLVES to a positive id at the evidence gate (obs==3)")

    # --- continuous track retains ONE reid (criterion 1) --------------------
    ids = [id3]
    for k in range(20):
        ids.append(eng.assign("cam_a", 10, observe(alice, seed=200 + k), None,
                               ts=1.3 + 0.1 * k, has_fresh_emb=True))
    c.ok(len(set(ids)) == 1, f"continuous track keeps a single reid (saw {set(ids)})")
    gid_alice = id3

    # --- prototype/medoid storage is usable ---------------------------------
    c.ok(eng.store.has(gid_alice), "identity is stored in the active set")
    c.ok(eng.store.obs_count(gid_alice) >= 3, "observation count accumulated")
    s_same = eng.store.score(gid_alice, observe(alice, seed=999))
    s_diff = eng.store.score(gid_alice, person(2))
    c.ok(s_same is not None and s_same > 0.9, f"score(same person) high ({s_same:.3f})")
    c.ok(s_diff is not None and s_diff < 0.5, f"score(different person) low ({s_diff:.3f})")
    proto = eng.store.prototype(gid_alice)
    c.ok(proto is not None and abs(float((proto @ proto)) - 1.0) < 1e-4,
         "prototype is a unit vector")

    # --- mint-only in substage 1: a distinct track gets a distinct id -------
    bob = person(2)
    idb = None
    for k in range(3):
        idb = eng.assign("cam_a", 11, observe(bob, seed=300 + k), None,
                         ts=5.0 + 0.1 * k, has_fresh_emb=True)
    c.ok(idb > 0 and idb != gid_alice, f"distinct track mints a distinct id ({idb} != {gid_alice})")

    # --- no-fresh-embedding frames keep the sticky id -----------------------
    sticky = eng.assign("cam_a", 10, None, None, ts=9.0, has_fresh_emb=False)
    c.eq(sticky, gid_alice, "sticky id retained on a frame with no fresh embedding")
    c.eq(eng.minted, 2, "exactly two identities minted (alice, bob)")

    c.done()


if __name__ == "__main__":
    main()
