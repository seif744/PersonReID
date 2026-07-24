"""
_synth.py  --  shared helpers for the Stage-3 identity-engine logic tests.

Deterministic synthetic embeddings so the engine's decisions can be validated
with NO GPU, video, models or threads (handoff validation cadence: "iterate +
logic-test locally each stage, deterministic, synthetic"). Also puts repo `src/`
on sys.path so tests can `from live.identity_engine import ...`.

A "person" is a fixed random unit vector in R^D. An "observation" is that vector
plus a little noise, renormalized -- so same-person cosine stays high (~0.97+)
and two independent people stay low (~0 in high-D), letting us dial confidence
by choosing how close two base vectors are.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DIM = 512


def person(seed, dim=DIM):
    """A fixed unit 'appearance' vector for a person id (by seed)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def observe(base, noise=0.05, seed=None):
    """One noisy observation of `base`, renormalized (stays close to base).

    `noise` is the approximate L2 magnitude of the perturbation RELATIVE to the
    unit base (so it must be scaled by 1/sqrt(dim) per component, else in high-D
    the noise vector's norm ~ noise*sqrt(dim) swamps the signal). The resulting
    cosine(base, obs) ~= 1/sqrt(1 + noise^2): noise=0.05 -> ~0.999 (continuous
    track), noise~0.7 -> ~0.82, noise~1.0 -> ~0.71 (cross-camera-ish)."""
    rng = np.random.default_rng(seed)
    dim = base.shape[0]
    v = base + (noise / np.sqrt(dim)) * rng.standard_normal(base.shape).astype(np.float32)
    return v / np.linalg.norm(v)


def blend(a, b, alpha):
    """Unit vector alpha*a + (1-alpha)*b -- to manufacture a target cosine
    between two people (e.g. a look-alike at a chosen similarity)."""
    v = alpha * a + (1.0 - alpha) * b
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def orthobasis(n, seed=0, dim=DIM):
    """`n` EXACTLY orthonormal unit vectors (QR of a random matrix). Lets the
    margin / reciprocal-best tests pin cosines analytically -- e.g. cos of
    blend(e_i, e_j, alpha) with e_i is exactly alpha/sqrt(alpha^2+(1-alpha)^2) --
    with none of the ~0.05 slack two independent random vectors carry in R^dim."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, n)).astype(np.float32)
    q, _ = np.linalg.qr(m)
    return [q[:, i] / np.linalg.norm(q[:, i]) for i in range(n)]


class Check:
    """Tiny assert harness: collects pass/fail lines, prints, exits non-zero on
    any failure so the shell/CI sees the result without a test framework."""

    def __init__(self, title):
        self.title = title
        self.failures = []
        self.n = 0

    def ok(self, cond, msg):
        self.n += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            self.failures.append(msg)

    def eq(self, got, want, msg):
        self.ok(got == want, f"{msg} (got={got!r}, want={want!r})")

    def done(self):
        print(f"\n{self.title}: {self.n - len(self.failures)}/{self.n} checks passed")
        if self.failures:
            print("FAILURES:")
            for f in self.failures:
                print(f"  - {f}")
            sys.exit(1)
        print("OK\n")
