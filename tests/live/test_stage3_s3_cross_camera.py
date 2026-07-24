"""
Stage 3 -- Substage 3 validation: cross-camera reciprocal-best matching, plus the
two conservative guards (runner-up margin + reciprocal-best) that keep it from
false-merging.

Acceptance (this substage):
  * A person moving between two cameras (similar appearance) gets the SAME reid
    when confidence is sufficient (Stage-3 criterion 3).
  * A low-confidence look-alike below cross_camera_threshold is NOT merged (mint)
    (Stage-3 criterion 4: prefer minting).
  * Runner-up MARGIN guard: two near-equal cross-camera candidates -> ambiguous
    -> mint (don't guess).
  * RECIPROCAL-BEST guard: our best candidate has an even better twin partner of
    its own -> we are not its reciprocal best -> mint (don't absorb a look-alike
    crowd). Uses exact orthonormal geometry so the decision is deterministic.
"""

from _synth import Check, person, observe, blend, cos, orthobasis
from live.identity_engine import IdentityEngine


def _resolve(eng, cam, tid, vec_fn, t0, n=3, dt=0.2):
    last, t = None, t0
    for k in range(n):
        last = eng.assign(cam, tid, vec_fn(k), None, ts=t, has_fresh_emb=True)
        t += dt
    return last


def main():
    c = Check("Stage3-S3 cross-camera reciprocal-best + margin/reciprocal guards")

    # 1) CLEAN CROSS-CAMERA LINK (criterion 3) -------------------------------
    eng = IdentityEngine(min_evidence_obs=3, cross_camera_threshold=0.63, accept_margin=0.03)
    alice = person(1)
    g_a = _resolve(eng, "cam_a", 10, lambda k: observe(alice, seed=10 + k), t0=1.0)
    # keep alice on cam_a a moment, then she appears on cam_b (degraded view)
    for k in range(4):
        eng.assign("cam_a", 10, observe(alice, seed=40 + k), None, ts=2.0 + 0.2 * k, has_fresh_emb=True)
    g_b = _resolve(eng, "cam_b", 20, lambda k: observe(alice, noise=0.5, seed=70 + k), t0=20.0)
    c.eq(g_b, g_a, "cross-camera: same person on a second camera -> SAME reid")
    c.ok(eng.linked >= 1, "link went through the cross-camera lane")

    # 2) LOW-CONFIDENCE LOOK-ALIKE -> MINT (criterion 4) ---------------------
    eng = IdentityEngine(min_evidence_obs=3, cross_camera_threshold=0.63)
    alice = person(1)
    g_a = _resolve(eng, "cam_a", 10, lambda k: observe(alice, seed=10 + k), t0=1.0)
    third = person(5)
    zbase = blend(alice, third, 0.35)                 # cos(z, alice) ~= 0.47 (< 0.63)
    c.ok(cos(zbase, alice) < 0.63, f"look-alike is genuinely sub-threshold (cos={cos(zbase, alice):.3f})")
    g_z = _resolve(eng, "cam_b", 20, lambda k: observe(zbase, seed=90 + k), t0=20.0)
    c.ok(g_z != g_a, f"low-confidence look-alike is NOT merged -> mint ({g_z} != {g_a})")

    # 3) RUNNER-UP MARGIN GUARD -> MINT --------------------------------------
    # Q sits equidistant between two distinct identities (both >= threshold);
    # neither wins by the margin, so we refuse to guess.
    eng = IdentityEngine(min_evidence_obs=3, cross_camera_threshold=0.63, accept_margin=0.03)
    e = orthobasis(3, seed=7)
    va, vb = e[0], e[1]                                # exactly orthonormal
    g_x = _resolve(eng, "cam_x", 1, lambda k: va, t0=1.0)
    g_y = _resolve(eng, "cam_y", 2, lambda k: vb, t0=1.0)
    c.ok(g_x != g_y, "two orthogonal people are distinct identities")
    q = blend(va, vb, 0.5)                             # cos ~0.707 to BOTH
    g_q = _resolve(eng, "cam_z", 3, lambda k: q, t0=10.0)
    c.ok(g_q != g_x and g_q != g_y,
         f"ambiguous (within-margin) candidates -> mint ({g_q} not in {{{g_x},{g_y}}})")

    # 4) RECIPROCAL-BEST GUARD -> MINT ---------------------------------------
    # G1 is Q's clear best (margin passes), but G1 has a near-twin G2 (cos 0.90)
    # that is a BETTER partner for G1 than Q is -> Q is not G1's reciprocal best.
    eng = IdentityEngine(min_evidence_obs=3, cross_camera_threshold=0.63, accept_margin=0.03)
    a, b, cc = orthobasis(3, seed=11)
    twin = blend(a, b, 0.674)                          # cos(twin, a) ~= 0.90
    qv = blend(a, cc, 0.531)                           # cos(qv, a) ~= 0.75
    c.ok(abs(cos(twin, a) - 0.90) < 0.02, f"twin cos to a is ~0.90 ({cos(twin, a):.3f})")
    c.ok(abs(cos(qv, a) - 0.75) < 0.02, f"query cos to a is ~0.75 ({cos(qv, a):.3f})")
    # G1 (identity 'a') and its twin, kept co-present on cam_a so the twin can't
    # merge into G1 (overlap veto) -> two distinct, highly-similar identities.
    for k, t in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        eng.assign("cam_a", 100, a, None, ts=t, has_fresh_emb=True)       # G1 span [1,5]
    g1 = eng._tracks[("cam_a", 100)]["gid"]
    g2 = _resolve(eng, "cam_a", 101, lambda k: twin, t0=2.0)              # overlaps -> mint
    c.ok(g1 != g2, f"twin minted a distinct id under the overlap veto ({g1} != {g2})")
    links_before = eng.linked
    g_q = _resolve(eng, "cam_b", 102, lambda k: qv, t0=20.0)
    c.ok(g_q != g1 and g_q != g2,
         f"reciprocal-best fails (twin is g1's better partner) -> mint ({g_q})")
    c.eq(eng.linked, links_before, "no cross-camera link was recorded for the ambiguous query")

    c.done()


if __name__ == "__main__":
    main()
