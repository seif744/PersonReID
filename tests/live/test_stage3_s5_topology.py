"""
Stage 3 -- Substage 5 validation: the pluggable topology veto.

Acceptance (this substage):
  * FailOpenTopology (used when no map is configured) allows EVERY transition ->
    cross-camera links are appearance-only, never blocked.
  * The veto is actually WIRED into the engine's cross-camera lane: a topology
    that says "not enough time to get there" turns a link that WOULD succeed on
    appearance into a fresh mint.
  * GraphTopology fails open on any camera it doesn't know (invariant §2.5).
  * GraphTopology enforces shortest-path MINIMUM transit time, undirected and
    multi-hop -- validated on the ACTUAL deployment layout (cam_224/cam_219 in one
    room, cam_213 the chokepoint, cam_206 reachable only through cam_213).
"""

from _synth import Check, person, observe
from live.identity_engine import IdentityEngine
from live.topology import FailOpenTopology, GraphTopology


def _resolve(eng, cam, tid, base, t0, n=3, noise=0.05, dt=0.2):
    last, t = None, t0
    for k in range(n):
        last = eng.assign(cam, tid, observe(base, noise=noise, seed=hash((cam, tid, k)) % 9999),
                          None, ts=t, has_fresh_emb=True)
        t += dt
    return last


def _cross_scenario(topology):
    """alice on cam_a (~ts 1-2.6), then cam_b (~ts 20.4); returns (g_a, g_b, linked)."""
    eng = IdentityEngine(min_evidence_obs=3, cross_camera_threshold=0.63, topology=topology)
    alice = person(1)
    g_a = _resolve(eng, "cam_a", 10, alice, t0=1.0)
    for k in range(4):
        eng.assign("cam_a", 10, observe(alice, seed=40 + k), None, ts=2.0 + 0.2 * k, has_fresh_emb=True)
    g_b = _resolve(eng, "cam_b", 20, alice, t0=20.0, noise=0.5)
    return g_a, g_b, eng.linked


def main():
    c = Check("Stage3-S5 topology veto (fail-open + shortest-path min transit)")

    # --- interface: FailOpenTopology allows everything ----------------------
    fo = FailOpenTopology()
    c.ok(fo.allowed("a", "b", 0.0, 5.0), "FailOpenTopology allows an arbitrary cross pair")
    c.ok(fo.allowed("a", "a", 0.0, 0.0), "FailOpenTopology allows same-camera")

    # --- wired default: link succeeds (topology never blocks) ---------------
    g_a, g_b, linked = _cross_scenario(FailOpenTopology())
    c.eq(g_b, g_a, "fail-open default: cross-camera link succeeds")
    c.ok(linked >= 1, "link recorded")

    # --- veto is WIRED: 'too fast to get there' forces a mint ---------------
    # gap in the scenario is ~17.8s; a 30s min transit means alice couldn't have
    # reached cam_b yet -> veto -> mint.
    reject = GraphTopology(edges=[("cam_a", "cam_b", 30.0)])
    g_a2, g_b2, linked2 = _cross_scenario(reject)
    c.ok(g_b2 != g_a2, f"too-fast transition vetoed -> mint ({g_b2} != {g_a2})")
    c.eq(linked2, 0, "no cross-camera link recorded when the transition is vetoed")

    # --- GraphTopology fails open on unknown cameras ------------------------
    partial = GraphTopology(edges=[("cam_a", "cam_x", 5.0)])     # cam_b is unknown
    g_a3, g_b3, linked3 = _cross_scenario(partial)
    c.eq(g_b3, g_a3, "unknown camera -> fail open -> link still succeeds")

    # --- DEPLOYMENT LAYOUT: room {cam_224,cam_219} -- cam_213 -- cam_206 -----
    # cam_224 & cam_219 are CO-VISIBLE (overlapping views) -> 0s: a simultaneous
    # cross-camera link must NOT be vetoed. The chokepoint edges keep real
    # transit-time vetoes for the non-overlapping hops.
    g = GraphTopology(edges=[
        ("cam_224", "cam_219", 0.0),   # same room, co-visible
        ("cam_224", "cam_213", 3.0),
        ("cam_219", "cam_213", 3.0),
        ("cam_213", "cam_206", 4.0),
    ])
    c.ok(g.allowed("cam_224", "cam_219", 100.0, 100.0),
         "co-visible cameras: simultaneous link ALLOWED (dt=0, transit=0)")
    c.ok(g.allowed("cam_219", "cam_224", 100.0, 100.0),
         "co-visible is undirected: allowed both ways at dt=0")
    c.ok(abs(g.min_transit("cam_206", "cam_224") - 7.0) < 1e-6,
         "cam_206 -> cam_224 min transit = 7s (4+3 via cam_213 chokepoint)")
    c.ok(abs(g.min_transit("cam_206", "cam_219") - 7.0) < 1e-6,
         "cam_206 -> cam_219 min transit = 7s (multi-hop)")
    c.ok(not g.allowed("cam_206", "cam_224", 100.0, 105.0),
         "cam_206->cam_224 in 5s vetoed (< 7s through the chokepoint)")
    c.ok(g.allowed("cam_206", "cam_224", 100.0, 110.0),
         "cam_206->cam_224 in 10s allowed (>= 7s)")
    c.ok(not g.allowed("cam_213", "cam_206", 100.0, 102.0),
         "chokepoint hop still vetoes too-fast (2s < 4s)")
    c.ok(g.allowed("cam_224", "cam_999", 100.0, 100.1),
         "unknown camera in the layout -> fail open")

    c.done()


if __name__ == "__main__":
    main()
