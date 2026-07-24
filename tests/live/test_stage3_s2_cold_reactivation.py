"""
Stage 3 -- Substage 2 validation: cold reactivation (same-camera re-acquire lane)
+ the same-camera time-overlap HARD veto.

Acceptance (this substage):
  * A person who exits and re-enters the SAME camera after a short absence gets
    the SAME reid back (Stage-3 criterion 2: cold reactivation, fixes
    "same person -> different ids").
  * The same-camera time-overlap HARD veto holds: a second body co-present in the
    camera while the identity is still on screen is NOT merged into it -> a new
    id is minted (false-merge-conservative).
  * A genuinely different person in the same camera is not reactivated as the
    first (low score -> mint).
"""

from _synth import Check, person, observe
from live.identity_engine import IdentityEngine


def _resolve_track(eng, cam, tid, base, t0, n=3, noise=0.05, dt=0.2):
    """Drive one track through the evidence gate; return (last_id, next_ts)."""
    last = None
    t = t0
    for k in range(n):
        last = eng.assign(cam, tid, observe(base, noise=noise, seed=hash((tid, k)) % 10_000),
                          None, ts=t, has_fresh_emb=True)
        t += dt
    return last, t


def main():
    c = Check("Stage3-S2 cold reactivation + same-camera overlap veto")
    eng = IdentityEngine(min_evidence_obs=3, same_camera_threshold=0.90)
    alice = person(1)

    # 1) Alice appears on cam_a (track 10) and resolves; keep her on screen a bit.
    g_alice, t = _resolve_track(eng, "cam_a", 10, alice, t0=1.0)
    c.ok(g_alice > 0, f"alice resolves to a positive id ({g_alice})")
    for k in range(5):
        eng.assign("cam_a", 10, observe(alice, seed=500 + k), None,
                   ts=t + 0.2 * k, has_fresh_emb=True)     # span now ~[1.0, 2.4]

    # 2) COLD REACTIVATION: alice leaves, returns much later as a NEW track id (11)
    #    in the SAME camera -> must get the SAME id back (not a fresh mint).
    g_react, _ = _resolve_track(eng, "cam_a", 11, alice, t0=30.0)
    c.eq(g_react, g_alice, "cold reactivation: re-entered same camera -> SAME reid")
    c.ok(eng.reacquired >= 1, "reactivation went through the same-camera lane")
    minted_after_react = eng.minted

    # 3) OVERLAP VETO: a second body that looks like alice appears in cam_a WHILE
    #    alice(track 11) is still on screen -> cannot be the same identity.
    #    Keep track 11 alive across the interloper's window so the span overlaps.
    for k in range(6):
        eng.assign("cam_a", 11, observe(alice, seed=700 + k), None,
                   ts=31.0 + 0.2 * k, has_fresh_emb=True)   # span extends to ~32.0
    g_twin, _ = _resolve_track(eng, "cam_a", 12, alice, t0=31.2)   # overlaps [31.0,32.0]
    c.ok(g_twin != g_alice,
         f"overlap veto: co-present look-alike is NOT merged ({g_twin} != {g_alice})")
    c.ok(eng.minted > minted_after_react, "co-present look-alike minted a fresh id")

    # 4) A genuinely different person on cam_a does not reactivate as alice.
    bob = person(2)
    g_bob, _ = _resolve_track(eng, "cam_a", 13, bob, t0=60.0)
    c.ok(g_bob != g_alice, f"different person is not reacquired as alice ({g_bob})")

    # 5) CO-ACTIVE VETO (the A100 bug): two tracks on screen at the SAME time in
    #    one camera must NOT share an id even when the span-bracket check misses
    #    (the 2nd track resolves just AFTER the 1st track's span end -- the timing
    #    hole that merged two co-present people). Worst case: identical appearance.
    eng2 = IdentityEngine(min_evidence_obs=3, same_camera_threshold=0.90,
                          coactive_window_sec=2.0)
    twin_look = person(7)
    ga = _resolve_track(eng2, "cam_z", 1, twin_look, t0=1.0)[0]   # A: span ends ~1.4
    # B resolves at ts 1.6-2.0, i.e. AFTER A's last span point -> span check misses,
    # but A is still live (seen 0.2s ago) -> co-active veto must fire.
    gb = _resolve_track(eng2, "cam_z", 2, twin_look, t0=1.6)[0]
    c.ok(not eng2.store.same_camera_overlap(ga, "cam_z", 2.0),
         "span-bracket check MISSES the co-present pair (the original bug)")
    c.ok(gb != ga, f"co-active veto: co-present tracks get DISTINCT ids ({gb} != {ga})")
    c.ok(eng2.coactive_vetoes >= 1, "co-active veto fired (prevented the false merge)")

    c.done()


if __name__ == "__main__":
    main()
