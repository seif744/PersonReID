"""
constraints.py  --  STAGE 7 gating:  WHERE + WHEN checks on identity matches.

============================ WHY THIS EXISTS ==============================
The vector store answers "who does this crop look most like?" -- appearance only.
On real footage that is not enough: two different people in similar clothing score
almost as high as the same person seen twice (we measured different-people pairs
at 0.84-0.86 against a true 0.89). Appearance alone WILL fuse look-alikes.

A constraint gate adds the physics the camera geometry already knows, turning
"closest vector" into "closest PLAUSIBLE person". It answers a different question
than the model: not "do they look alike?" but "COULD this even be the same person,
given where and when each was seen?" A candidate that fails is rejected no matter
how high its appearance score.

Two gates are implemented here (the two that can veto a match):

  * TEMPORAL + TOPOLOGY (combined): a person leaving camera A and appearing in
    camera B must have had time to walk the path A->B. Each ordered camera pair
    gets a plausible transit window [min, max] seconds. A candidate last seen in
    A is rejected for a match in B if the elapsed time is outside that window --
    too fast (can't teleport) or too slow (they'd be long gone). A window with
    min 0 encodes OVERLAPPING fields of view, where the same person legitimately
    appears in both cameras at the same instant (exactly our cam_219/cam_224
    case). Pairs with no rule fall back to `default_policy`.

MOTION down-weighting (the third hook in service.py) needs per-track trajectory
/ bbox velocity, which is not plumbed into the identity path yet; it's left as a
documented extension rather than a stub that does nothing.

------------------------------- THE CLOCK ---------------------------------
Cross-camera timing needs a shared clock. Frame indices are per-video, so this
gate treats them as a common timeline (both recordings started together) and
converts frame gaps to seconds via `fps`. When cameras are NOT synchronized,
give each a start offset in the topology config; until then the synchronized
assumption is explicit here rather than hidden.

Default posture: OFF. With no topology configured the gate allows everything, so
enabling the class never silently changes matching until a real camera map is
supplied. This mirrors service.py's "prefer a split over a wrong merge" bias.
============================================================================
"""


class ConstraintGate:
    """
    Decides whether a candidate identity is physically plausible for a new
    observation, from camera topology + timing. See module docstring.
    """

    def __init__(self, transitions=None, fps=20.0, default_policy="allow"):
        """
        transitions : list of dicts, each {from, to, min_seconds, max_seconds}.
                      Directional: `from`->`to` is the travel A->B. max_seconds
                      may be omitted (None) to mean "no upper bound".
        fps         : frames-per-second, to convert frame gaps to seconds. If
                      falsy, gaps are compared directly in FRAMES and the window
                      bounds are read as frames too.
        default_policy : "allow" or "reject" for camera pairs with no rule.
        """
        self.fps = float(fps) if fps else None
        self.default_allow = str(default_policy).lower() != "reject"
        self._windows = {}   # (from_cam, to_cam) -> (min, max_or_None)
        for t in (transitions or []):
            src, dst = t["from"], t["to"]
            self._windows[(src, dst)] = (
                float(t.get("min_seconds", 0.0)),
                None if t.get("max_seconds") is None else float(t["max_seconds"]),
            )

    def allows(self, camera, frame, cand_camera, cand_frame):
        """
        Return (ok, reason). ok=True means the candidate is plausible here/now.
        `cand_camera`/`cand_frame` are where/when the candidate identity was most
        recently seen. reason is a short string when rejected, else None.
        """
        # No location/time info, or a reappearance in the SAME camera: appearance
        # is the only thing to go on -- don't veto. (Same-camera continuity is the
        # tracker's job; the sticky track->id map already handles it.)
        if cand_camera is None or cand_frame is None or camera == cand_camera:
            return True, None

        window = self._windows.get((cand_camera, camera))
        if window is None:
            if self.default_allow:
                return True, None
            return False, f"no configured path {cand_camera}->{camera}"

        lo, hi = window
        gap = abs(frame - cand_frame)
        dt = gap / self.fps if self.fps else gap
        unit = "s" if self.fps else "f"
        if dt < lo:
            return False, f"{cand_camera}->{camera} too fast ({dt:.1f}{unit} < {lo}{unit})"
        if hi is not None and dt > hi:
            return False, f"{cand_camera}->{camera} too slow ({dt:.1f}{unit} > {hi}{unit})"
        return True, None

    @classmethod
    def from_config(cls, cfg):
        """Build from the `identity.constraints` block, or None if disabled/absent."""
        if not cfg or not cfg.get("enabled"):
            return None
        return cls(
            transitions=cfg.get("transitions"),
            fps=cfg.get("fps", 20.0),
            default_policy=cfg.get("default_policy", "allow"),
        )
