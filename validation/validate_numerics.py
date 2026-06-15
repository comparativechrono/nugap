#!/usr/bin/env python3
r"""
Numerical correctness checks for the nugap package.

Each check is *independent* of the package's own internals, so it tests the
mathematics rather than re-checking a formula against itself:

  A. Closed-form ground truth      exact values computable by hand (constant
                                   gains; stable-vs-unstable winding case).
  B. Algebraic invariance          the chordal metric is invariant under
                                   w -> 1/w, so nu_gap(P1,P2) == nu_gap(1/P1,1/P2)
                                   for stable minimum-phase systems. This is
                                   magnitude-independent and shares no code with
                                   the package.
  C. Independent reference impl.   a deliberately naive, from-scratch evaluation
                                   of the nu-gap (own contour, own winding number,
                                   own pole counts) compared to the package over
                                   many random discrete systems. Same definition,
                                   different code -> catches indexing/contour bugs.
  D. Robust-stability theorem      the *defining property* of the nu-gap
                                   (Vinnicombe): for any C that stabilises both
                                   plants, arcsin b(P2,C) >= arcsin b(P1,C) -
                                   arcsin d_nu(P1,P2). The stability margin b(P,C)
                                   is computed here from the gang-of-four H-inf
                                   norm, with no reference to d_nu. This validates
                                   the *meaning* of the number, not just a formula.

Run:  python validate_numerics.py [--seed N] [--mutate FACTOR]
`--mutate` multiplies the package's nu-gap by FACTOR before the checks, to
demonstrate that a wrong metric is actually caught (a "does the test have teeth"
mutation test). FACTOR=1.0 (default) runs the real package.
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from nugap import tf, nu_gap as _pkg_nu_gap          # noqa: E402
from nugap.systems import LTI                          # noqa: E402

MUT = 1.0
def nu_gap(P1, P2, **kw):
    return float(np.clip(MUT * _pkg_nu_gap(P1, P2, **kw), 0.0, 1.0))


# ----------------------------------------------------------------------------- helpers
def chordal(h1, h2):
    return np.abs(h1 - h2) / np.sqrt((1 + np.abs(h1) ** 2) * (1 + np.abs(h2) ** 2))

def evaltf(num, den, pts):
    return np.polyval(num, pts) / np.polyval(den, pts)

def rand_discrete(rng, order=None, stable=True, minphase=None):
    """Random real discrete proper transfer function (descending coeffs)."""
    order = order or rng.integers(1, 3)
    def roots_in_disk(k, r=0.9):
        rs = []
        while len(rs) < k:
            if k - len(rs) >= 2 and rng.random() < 0.5:
                mag = r * rng.random(); ang = np.pi * rng.random()
                z = mag * np.exp(1j * ang); rs += [z, np.conj(z)]
            else:
                rs.append(r * (2 * rng.random() - 1))
        return rs[:k]
    poles = roots_in_disk(order) if stable else (
        [1.05 + 0.3 * rng.random()] + roots_in_disk(order - 1))
    nz = rng.integers(0, order + 1)
    zeros = roots_in_disk(nz) if (minphase or minphase is None) else [1.2] * nz
    den = np.real(np.poly(poles)); num = np.real(np.poly(zeros)) if nz else np.array([1.0])
    num = num * (0.3 + rng.random())
    num = np.concatenate([np.zeros(len(den) - len(num)), num])  # pad to proper
    return num, den

def report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


# ----------------------------------------------------------------------------- A
def check_closed_form():
    print("A. Closed-form ground truth")
    ok = True
    # constant gains: d_nu = |k1-k2| / sqrt((1+k1^2)(1+k2^2)), exactly
    for k1, k2 in [(0.0, 1.0), (2.0, -0.5), (3.0, 0.25)]:
        exact = abs(k1 - k2) / np.sqrt((1 + k1 ** 2) * (1 + k2 ** 2))
        got = nu_gap(tf([k1], [1.0]), tf([k2], [1.0]))
        ok &= report(f"constant gains ({k1},{k2})", abs(got - exact) < 1e-9,
                      f"pkg={got:.12f} exact={exact:.12f} |diff|={abs(got-exact):.1e}")
    # stable vs unstable: winding condition fails -> d_nu = 1 exactly
    got = nu_gap(tf([1.0], [1.0, 1.0]), tf([1.0], [1.0, -1.0]))   # 1/(s+1) vs 1/(s-1)
    ok &= report("stable vs unstable (winding) -> 1", abs(got - 1.0) < 1e-9, f"pkg={got:.12f}")
    # identity -> 0 exactly
    P = tf([1.0, 0.5], [1.0, 0.7, 0.2])
    ok &= report("identity -> 0", nu_gap(P, P) < 1e-12, f"pkg={nu_gap(P,P):.2e}")
    return ok


# ----------------------------------------------------------------------------- B
def check_reciprocal_invariance(rng, ntrials=200):
    print("B. Algebraic invariance  d_nu(P1,P2) == d_nu(1/P1,1/P2)  (chordal kappa is 1/w-invariant)")
    worst = 0.0
    for _ in range(ntrials):
        # biproper stable minimum-phase (so 1/P is also stable, minimum-phase, proper)
        n1, d1 = rand_discrete(rng, order=2, stable=True, minphase=True)
        n2, d2 = rand_discrete(rng, order=2, stable=True, minphase=True)
        # make biproper: ensure numerator degree == denominator degree, leading coeff != 0
        n1 = n1.copy(); n1[0] = n1[0] or 0.5
        n2 = n2.copy(); n2[0] = n2[0] or 0.5
        P1, P2 = tf(n1, d1, dt=1.0), tf(n2, d2, dt=1.0)
        R1, R2 = tf(d1, n1, dt=1.0), tf(d2, n2, dt=1.0)   # reciprocals
        if not (P1.is_stable and P2.is_stable and R1.is_stable and R2.is_stable):
            continue
        a, b = nu_gap(P1, P2), nu_gap(R1, R2)
        worst = max(worst, abs(a - b))
    ok = worst < 5e-4
    return report("reciprocal invariance", ok, f"max |d_nu(P) - d_nu(1/P)| over trials = {worst:.2e}")


# ----------------------------------------------------------------------------- C
def ref_nu_gap_discrete(num1, den1, num2, den2, n=40000, tau=1e-7):
    """From-scratch nu-gap on the unit circle (independent of package code)."""
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    z = np.exp(1j * th)
    H1, H2 = evaltf(num1, den1, z), evaltf(num2, den2, z)
    sup_kappa = float(np.nanmax(chordal(H1, H2)))
    # winding number of g = 1 + conj(H2) * H1 around the origin (counter-clockwise),
    # then Vinnicombe's clockwise convention negates it
    g = 1 + np.conj(H2) * H1
    ccw = np.sum(np.diff(np.unwrap(np.angle(np.append(g, g[0]))))) / (2 * np.pi)
    wno_cw = -int(round(ccw))
    def counts(den):
        p = np.roots(den)
        eta = int(np.sum(np.abs(p) > 1 + tau))
        eta0 = int(np.sum(np.abs(np.abs(p) - 1) <= tau))
        return eta, eta0
    eta1, _ = counts(den1); eta2, eta0_2 = counts(den2)
    w = wno_cw + eta1 - eta2 - eta0_2
    return sup_kappa if w == 0 else 1.0

def check_reference_impl(rng, ntrials=400):
    print("C. Independent reference implementation (own contour / winding / pole counts)")
    worst = 0.0; nbad = 0; examples = []
    for _ in range(ntrials):
        stab1 = rng.random() < 0.75; stab2 = rng.random() < 0.75
        n1, d1 = rand_discrete(rng, stable=stab1)
        n2, d2 = rand_discrete(rng, stable=stab2)
        P1, P2 = tf(n1, d1, dt=1.0), tf(n2, d2, dt=1.0)
        ref = ref_nu_gap_discrete(n1, d1, n2, d2)
        pkg = nu_gap(P1, P2, n=16384)
        diff = abs(ref - pkg)
        # both saturate to 1 via winding -> agreement is exact; otherwise compare sup
        if diff > worst:
            worst = diff
        if diff > 2e-3:
            nbad += 1
            if len(examples) < 3:
                examples.append((round(pkg, 4), round(ref, 4)))
    ok = nbad == 0
    return report("reference vs package", ok,
                  f"max |ref - pkg| over {ntrials} random systems = {worst:.2e}"
                  + (f"; mismatches={nbad} e.g. {examples}" if nbad else ""))


# ----------------------------------------------------------------------------- D
def gang_of_four_margin(P, C, n=8000):
    """b(P,C) = 1 / sup_w sigma_max(M), M = (1/(1+PC)) [[1,P],[C,PC]] on |z|=1.
    Returns (b, internally_stable)."""
    nump, denp = P; numc, denc = C
    char = np.polyadd(np.polymul(nump, numc), np.polymul(denp, denc))  # den(1+PC)
    stable = np.all(np.abs(np.roots(char)) < 1 - 1e-9)
    th = np.linspace(-np.pi, np.pi, n, endpoint=False); z = np.exp(1j * th)
    p = evaltf(nump, denp, z); c = evaltf(numc, denc, z)
    s = 1.0 / (1 + p * c)
    smax = np.empty(n)
    for i in range(n):
        M = np.array([[s[i], p[i] * s[i]], [c[i] * s[i], p[i] * c[i] * s[i]]])
        smax[i] = np.linalg.svd(M, compute_uv=False)[0]
    return 1.0 / float(np.max(smax)), bool(stable)

def check_robust_stability(rng, npairs=8, want_per_pair=60):
    print("D. Robust-stability theorem  arcsin b(P2,C) >= arcsin b(P1,C) - arcsin d_nu(P1,P2)")
    max_violation = -1.0; max_gap_over_dnu = -1.0; total = 0; viol = 0
    for _ in range(npairs):
        n1, d1 = rand_discrete(rng, order=rng.integers(1, 3), stable=True)
        n2, d2 = rand_discrete(rng, order=rng.integers(1, 3), stable=True)
        P1, P2 = tf(n1, d1, dt=1.0), tf(n2, d2, dt=1.0)
        if not (P1.is_stable and P2.is_stable):
            continue
        dnu = nu_gap(P1, P2)
        got = 0
        for _ in range(2000):
            if got >= want_per_pair:
                break
            nc, dc = rand_discrete(rng, order=rng.integers(1, 3), stable=True)
            b1, st1 = gang_of_four_margin((n1, d1), (nc, dc))
            b2, st2 = gang_of_four_margin((n2, d2), (nc, dc))
            if not (st1 and st2):
                continue
            got += 1; total += 1
            gap = np.arcsin(min(b1, 1)) - np.arcsin(min(b2, 1))           # arcsin b1 - arcsin b2
            v = gap - np.arcsin(min(dnu, 1))                              # must be <= 0
            max_violation = max(max_violation, v)
            max_gap_over_dnu = max(max_gap_over_dnu, gap / max(np.arcsin(min(dnu, 1)), 1e-9))
            if v > 1e-3:
                viol += 1
    ok = viol == 0
    return report("no controller violates the bound", ok,
                  f"tested {total} stabilising controllers; max overshoot of "
                  f"[arcsin b1 - arcsin b2] above arcsin(d_nu) = {max_violation:+.2e}; "
                  f"tightest sampled ratio to arcsin(d_nu) = {max(max_gap_over_dnu,0):.2f}")


# ----------------------------------------------------------------------------- main
def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--mutate", type=float, default=1.0,
                    help="multiply package nu-gap by this factor (mutation test)")
    args = ap.parse_args()
    MUT = args.mutate
    rng = np.random.default_rng(args.seed)
    banner = f"nugap numerical validation  (seed={args.seed}, mutate={MUT})"
    print(banner); print("=" * len(banner))
    results = [
        check_closed_form(),
        check_reciprocal_invariance(rng),
        check_reference_impl(rng),
        check_robust_stability(rng),
    ]
    print("-" * len(banner))
    if all(results):
        print(f"ALL CHECKS PASSED ({sum(results)}/{len(results)})")
        return 0
    print(f"FAILURES: {len(results) - sum(results)}/{len(results)} check groups failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
