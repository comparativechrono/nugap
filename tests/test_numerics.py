"""Fast numerical-correctness checks (CI subset of validation/validate_numerics.py).

These are independent of the package internals: exact closed-form values, the
1/w-invariance of the chordal metric, and a small from-scratch reference
implementation. The full harness (including the robust-stability theorem check)
lives in validation/validate_numerics.py and is run on demand because it is
compute-heavy.
"""
import numpy as np
from nugap import tf, nu_gap


def _chordal(h1, h2):
    return np.abs(h1 - h2) / np.sqrt((1 + np.abs(h1) ** 2) * (1 + np.abs(h2) ** 2))


def test_closed_form_constant_gains():
    for k1, k2 in [(0.0, 1.0), (2.0, -0.5), (3.0, 0.25)]:
        exact = abs(k1 - k2) / np.sqrt((1 + k1 ** 2) * (1 + k2 ** 2))
        assert abs(nu_gap(tf([k1], [1.0]), tf([k2], [1.0])) - exact) < 1e-9


def test_closed_form_winding_saturates_to_one():
    # 1/(s+1) vs 1/(s-1): stable vs unstable -> winding condition fails -> 1
    assert abs(nu_gap(tf([1.0], [1.0, 1.0]), tf([1.0], [1.0, -1.0])) - 1.0) < 1e-9


def test_reciprocal_invariance():
    # chordal kappa is invariant under w -> 1/w, so for stable minimum-phase
    # systems nu_gap(P1,P2) == nu_gap(1/P1,1/P2)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(40):
        d1 = np.real(np.poly([0.6 * (2 * rng.random() - 1), 0.6 * (2 * rng.random() - 1)]))
        n1 = np.real(np.poly([0.5 * (2 * rng.random() - 1), 0.5 * (2 * rng.random() - 1)]))
        d2 = np.real(np.poly([0.6 * (2 * rng.random() - 1), 0.6 * (2 * rng.random() - 1)]))
        n2 = np.real(np.poly([0.5 * (2 * rng.random() - 1), 0.5 * (2 * rng.random() - 1)]))
        P1, P2 = tf(n1, d1, dt=1.0), tf(n2, d2, dt=1.0)
        R1, R2 = tf(d1, n1, dt=1.0), tf(d2, n2, dt=1.0)
        worst = max(worst, abs(nu_gap(P1, P2) - nu_gap(R1, R2)))
    assert worst < 5e-4


def test_independent_reference_implementation():
    """From-scratch nu-gap on the unit circle vs the package, random systems."""
    rng = np.random.default_rng(1)

    def ref(n1, d1, n2, d2, n=20000, tau=1e-7):
        z = np.exp(1j * np.linspace(-np.pi, np.pi, n, endpoint=False))
        H1, H2 = np.polyval(n1, z) / np.polyval(d1, z), np.polyval(n2, z) / np.polyval(d2, z)
        sup = float(np.nanmax(_chordal(H1, H2)))
        g = 1 + np.conj(H2) * H1
        ccw = np.sum(np.diff(np.unwrap(np.angle(np.append(g, g[0]))))) / (2 * np.pi)
        wno = -int(round(ccw))
        cnt = lambda d: (int(np.sum(np.abs(np.roots(d)) > 1 + tau)),
                         int(np.sum(np.abs(np.abs(np.roots(d)) - 1) <= tau)))
        e1, _ = cnt(d1); e2, e0 = cnt(d2)
        return sup if (wno + e1 - e2 - e0) == 0 else 1.0

    def rand(stable):
        r = (0.6 * (2 * rng.random() - 1)) if stable else (1.1 + 0.3 * rng.random())
        den = np.array([1.0, -r])
        num = np.array([0.0, 0.3 + rng.random()]) if rng.random() < 0.5 else np.array([0.5, 0.0])
        return num, den

    worst = 0.0
    for _ in range(60):
        n1, d1 = rand(rng.random() < 0.7)
        n2, d2 = rand(rng.random() < 0.7)
        worst = max(worst, abs(ref(n1, d1, n2, d2) - nu_gap(tf(n1, d1, dt=1.0), tf(n2, d2, dt=1.0), n=16384)))
    assert worst < 2e-3
