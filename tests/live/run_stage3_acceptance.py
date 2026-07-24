"""
run_stage3_acceptance.py  --  the Stage-3 gate: run every substage logic test and
map them onto the explicit Stage-3 acceptance criteria from the handoff.

    python tests/live/run_stage3_acceptance.py

Deterministic, synthetic, no GPU/video/threads (local validation cadence). Exits
non-zero if any test fails. The final criterion (file-batch regression) is not a
Python assertion here -- it is guaranteed structurally: Stage 3 touches only
src/live/*, so process_video / reconcile_tracklets / render_final_videos /
IdentityService are byte-identical (verify with `git status`).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# test file -> the acceptance criteria it exercises
SUITE = [
    ("test_stage3_s1_active_set.py",
     ["#1 continuous track retains ONE reid (+ prototype/medoid storage, evidence gate)"]),
    ("test_stage3_s2_cold_reactivation.py",
     ["#2 exit + re-enter SAME camera -> same reid (cold reactivation)",
      "#4 co-present look-alike NOT merged (same-camera overlap veto)"]),
    ("test_stage3_s3_cross_camera.py",
     ["#3 cross-camera same person -> same reid when confident",
      "#4 low-confidence / ambiguous look-alikes NOT merged (prefer minting)"]),
    ("test_stage3_s4_priority.py",
     ["(engine) anti-starvation fair queue: fairness + starvation bound + shedding"]),
    ("test_stage3_s5_topology.py",
     ["(engine) topology veto is fail-open by default and correctly wired"]),
]


def main():
    print("=" * 72)
    print("STAGE 3 ACCEPTANCE  --  in-memory identity engine (no Qdrant)")
    print("=" * 72)
    failed = []
    for test_file, criteria in SUITE:
        print(f"\n>>> {test_file}")
        for cr in criteria:
            print(f"      covers: {cr}")
        rc = subprocess.run([sys.executable, os.path.join(HERE, test_file)]).returncode
        if rc != 0:
            failed.append(test_file)

    print("\n" + "=" * 72)
    print("ACCEPTANCE CRITERIA")
    print("=" * 72)
    checks = [
        ("1  Continuous track retains one ReID", "test_stage3_s1_active_set.py"),
        ("2  Exit + re-enter same camera -> same ReID (cold reactivation)", "test_stage3_s2_cold_reactivation.py"),
        ("3  Cross-camera, similar appearance -> same ReID when confident", "test_stage3_s3_cross_camera.py"),
        ("4  Similar people NOT merged when confidence low (prefer minting)", "test_stage3_s3_cross_camera.py"),
        ("5  File-batch regression unchanged", None),
    ]
    for label, dep in checks:
        if dep is None:
            print(f"  [n/a] {label}  (structural: `git status` shows only src/live/* changed)")
        else:
            mark = "FAIL" if dep in failed else "PASS"
            print(f"  [{mark}] {label}")

    if failed:
        print(f"\nRESULT: FAIL ({len(failed)} test file(s): {', '.join(failed)})")
        sys.exit(1)
    print("\nRESULT: PASS  (criteria 1-4 validated on synthetic embeddings; "
          "criterion 5 structural)")


if __name__ == "__main__":
    main()
