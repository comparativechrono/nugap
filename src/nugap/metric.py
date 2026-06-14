"""The Vinnicombe nu-gap metric for SISO systems.

Reference: G. Vinnicombe, "Frequency domain uncertainty and the graph
topology", IEEE Trans. Automatic Control 38 (1993) 1371-1383, and
G. Vinnicombe, "Uncertain Systems and Feedback" (2001).

For two systems P1, P2 the nu-gap is

    delta_nu(P1, P2) = sup_w  kappa(P1, P2)(w)        if the winding-number
                                                        condition holds,
                     = 1                               otherwise,

where the *chordal distance* between two scalar transfer functions at a
frequency point is

                       |P1 - P2|
    kappa = ------------------------------------ ,   in [0, 1].
            sqrt(1+|P1|^2) * sqrt(1+|P2|^2)

The winding-number condition (SISO) is

    wno( 1 + conj(P2) * P1 )  +  eta(P1) - eta(P2) - eta0(P2)  =  0

with eta = number of unstable poles and eta0 = number of boundary poles
(imaginary axis for continuous systems, unit circle for discrete). If the
condition fails the systems are "topologically far" and delta_nu = 1.

The result is symmetric: delta_nu(P1, P2) == delta_nu(P2, P1), and lies in
[0, 1]. 0 means identical dynamics; near 1 means very different.
"""

from __future__ import annotations

import warnings
import numpy as np

from .systems import LTI


def chordal_distance(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Point-wise chordal distance between two frequency responses."""
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    return np.abs(P1 - P2) / (np.sqrt(1.0 + np.abs(P1) ** 2) * np.sqrt(1.0 + np.abs(P2) ** 2))


def _contour(sys1: LTI, sys2: LTI, n: int):
    """Build the integration contour and the matching complex points.

    Returns (param, points, closed) where ``points`` are the complex
    arguments at which to evaluate the transfer functions.

    Continuous: a dense, symmetric grid of frequencies covering the system
    dynamics by several decades; points = j*w.
    Discrete: theta in (-pi, pi]; points = exp(j*theta).
    """
    if sys1.is_discrete() or sys2.is_discrete():
        # unit circle
        theta = np.linspace(-np.pi, np.pi, n, endpoint=True)
        return theta, np.exp(1j * theta), True
    # continuous: choose range from the pole/zero magnitudes
    feats = np.concatenate([
        np.abs(sys1.poles), np.abs(sys2.poles),
        np.abs(sys1.zeros), np.abs(sys2.zeros),
    ])
    feats = feats[feats > 0]
    if feats.size:
        lo = np.log10(feats.min()) - 3
        hi = np.log10(feats.max()) + 3
    else:
        lo, hi = -3.0, 3.0
    w_pos = np.logspace(lo, hi, n // 2)
    w = np.concatenate([-w_pos[::-1], w_pos])
    return w, 1j * w, False


def _winding_number(g_vals: np.ndarray, closed: bool) -> int:
    """Net counter-clockwise encirclements of the origin by g along the
    contour, from the sampled values g_vals (ordered along the contour).

    For the discrete unit circle the contour is already closed. For the
    continuous imaginary axis we sweep a very wide symmetric frequency band;
    because here g(+/-inf) -> a real positive constant, the semicircle at
    infinity adds no net phase, so the imaginary-axis sweep suffices.
    """
    ang = np.unwrap(np.angle(g_vals))
    total = ang[-1] - ang[0]
    return int(np.round(total / (2.0 * np.pi)))


def _g_on_contour(sys1: LTI, sys2: LTI, points: np.ndarray) -> np.ndarray:
    """g = 1 + conj(P2) * P1 evaluated on the contour.

    On the imaginary axis / unit circle the para-conjugate P2~ equals the
    complex conjugate of P2, so this is exactly Vinnicombe's g restricted to
    the contour.
    """
    P1 = sys1.freqresp(points)
    P2 = sys2.freqresp(points)
    return 1.0 + np.conjugate(P2) * P1


def winding_condition(sys1: LTI, sys2: LTI, n: int = 8192, tol: float = 1e-7):
    """Evaluate the integer winding-number condition.

    Returns (ok: bool, value: int, touches_origin: bool).
    """
    _, points, closed = _contour(sys1, sys2, n)
    g = _g_on_contour(sys1, sys2, points)

    # If g touches the origin on the contour, the chordal distance reaches 1
    # somewhere and the systems sit on the boundary -> treat as far.
    touches = bool(np.min(np.abs(g)) < 1e-9 * max(1.0, np.max(np.abs(g))))

    wno_ccw = _winding_number(g, closed)
    # Vinnicombe's condition counts clockwise encirclements of the origin
    # along the Nyquist contour; _winding_number returns the (mathematically
    # standard) counter-clockwise count, so negate.
    wno = -wno_ccw
    eta1, _ = sys1.pole_counts(tol)
    eta2, eta02 = sys2.pole_counts(tol)
    value = wno + eta1 - eta2 - eta02
    ok = (value == 0) and not touches
    return ok, value, touches


def nu_gap(
    sys1: LTI,
    sys2: LTI,
    n: int = 8192,
    tol: float = 1e-7,
    return_details: bool = False,
):
    """Vinnicombe nu-gap metric delta_nu(sys1, sys2) in [0, 1].

    Parameters
    ----------
    sys1, sys2 : LTI
        SISO systems. Must both be continuous or both discrete with the
        same sampling time.
    n : int
        Number of contour samples (resolution of the frequency sweep).
    tol : float
        Tolerance used to classify poles as unstable / on the boundary.
    return_details : bool
        If True, also return a dict with the sup-chordal distance, the
        winding condition value, and the frequency of the maximum.

    Returns
    -------
    float  (or (float, dict) if return_details)
    """
    if sys1.is_discrete() != sys2.is_discrete():
        raise ValueError("cannot compare a continuous system with a discrete one")
    if sys1.is_discrete() and sys2.is_discrete() and sys1.dt != sys2.dt:
        warnings.warn("comparing discrete systems with different sample times")

    param, points, closed = _contour(sys1, sys2, n)
    P1 = sys1.freqresp(points)
    P2 = sys2.freqresp(points)

    kappa = chordal_distance(P1, P2)
    imax = int(np.nanargmax(kappa))
    sup_kappa = float(kappa[imax])

    ok, value, touches = winding_condition(sys1, sys2, n=n, tol=tol)
    delta = sup_kappa if ok else 1.0

    if return_details:
        details = {
            "sup_chordal": sup_kappa,
            "winding_value": value,
            "winding_ok": ok,
            "touches_origin": touches,
            "arg_at_max": float(param[imax]),
            "result": delta,
        }
        return delta, details
    return delta
