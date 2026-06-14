"""Lightweight SISO LTI system representation.

The core of the package only needs numpy/scipy. A system is stored as a
transfer function: numerator and denominator polynomial coefficients
(highest power first, the numpy/scipy convention), plus a sampling time
``dt``.

    dt is None  -> continuous-time system, P(s), poles in the s-plane,
                   metric contour is the imaginary axis s = j*w.
    dt > 0      -> discrete-time system, P(z), poles in the z-plane,
                   metric contour is the unit circle z = exp(j*theta).

Discrete-time is the natural choice for sampled time-course data, so it is
fully supported. python-control is *not* required; if it happens to be
installed you can convert via ``from_control``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LTI:
    num: np.ndarray  # numerator coeffs, highest power first
    den: np.ndarray  # denominator coeffs, highest power first
    dt: float | None = None  # None = continuous, >0 = discrete sample time

    def __post_init__(self):
        self.num = np.atleast_1d(np.asarray(self.num, dtype=float))
        self.den = np.atleast_1d(np.asarray(self.den, dtype=float))
        # strip leading zeros
        self.num = np.trim_zeros(self.num, "f")
        self.den = np.trim_zeros(self.den, "f")
        if self.num.size == 0:
            self.num = np.array([0.0])
        if self.den.size == 0:
            raise ValueError("denominator cannot be all zeros")
        if self.dt is not None and self.dt <= 0:
            raise ValueError("dt must be None (continuous) or a positive float")

    # ---- structural quantities ---------------------------------------
    @property
    def poles(self) -> np.ndarray:
        return np.roots(self.den)

    @property
    def zeros(self) -> np.ndarray:
        return np.roots(self.num)

    def is_discrete(self) -> bool:
        return self.dt is not None

    def freqresp(self, contour: np.ndarray) -> np.ndarray:
        """Evaluate the transfer function on complex points ``contour``.

        For continuous systems pass s = j*w. For discrete systems pass
        z = exp(j*theta).
        """
        num = np.polyval(self.num, contour)
        den = np.polyval(self.den, contour)
        return num / den

    def pole_counts(self, tol: float = 1e-7):
        """Return (eta, eta0): open-instability poles and boundary poles.

        Continuous: eta  = # poles with Re > tol  (open right half plane)
                    eta0 = # poles with |Re| <= tol (on the imaginary axis)
        Discrete:   eta  = # poles with |z| > 1 + tol (outside unit circle)
                    eta0 = # poles with | |z| - 1 | <= tol (on the unit circle)
        """
        p = self.poles
        if self.is_discrete():
            mag = np.abs(p)
            eta = int(np.sum(mag > 1.0 + tol))
            eta0 = int(np.sum(np.abs(mag - 1.0) <= tol))
        else:
            re = p.real
            eta = int(np.sum(re > tol))
            eta0 = int(np.sum(np.abs(re) <= tol))
        return eta, eta0

    def is_stable(self, tol: float = 1e-7) -> bool:
        eta, eta0 = self.pole_counts(tol)
        return eta == 0 and eta0 == 0


# ---- convenience constructors ----------------------------------------
def tf(num, den, dt=None) -> LTI:
    """Build a transfer function. dt=None continuous, dt>0 discrete."""
    return LTI(num, den, dt)


def from_zpk(zeros, poles, gain, dt=None) -> LTI:
    num = gain * np.poly(np.asarray(zeros, dtype=complex)) if len(np.atleast_1d(zeros)) else np.array([gain])
    den = np.poly(np.asarray(poles, dtype=complex))
    return LTI(np.real_if_close(num), np.real_if_close(den), dt)


def from_control(sys) -> LTI:
    """Convert a python-control TransferFunction/StateSpace to LTI (optional)."""
    import control as ct  # noqa: F401

    sys = ct.tf(sys)
    num = np.asarray(sys.num[0][0], dtype=float)
    den = np.asarray(sys.den[0][0], dtype=float)
    dt = None if (sys.dt is None or sys.dt == 0) else float(sys.dt)
    return LTI(num, den, dt)


def to_continuous(sys: LTI) -> LTI:
    """Convert a discrete LTI to continuous time (inverse zero-order hold).

    Useful to mirror MATLAB ``tfest`` continuous transfer functions. Works for
    well-conditioned stable models; the nu-gap *ranking* is essentially
    unchanged by the representation, so this is mainly for matching prior
    continuous-domain results. Experimental.
    """
    from scipy.signal import tf2ss, ss2tf
    from scipy.linalg import logm

    if not sys.is_discrete():
        return sys
    T = sys.dt
    Ad, Bd, Cd, Dd = tf2ss(sys.num, sys.den)
    Ac = np.real(logm(Ad)) / T
    # Bc solves Bd = (Ad - I) A^{-1} Bc  =>  Bc = A (Ad - I)^{-1} Bd
    I = np.eye(Ad.shape[0])
    Bc = Ac @ np.linalg.solve(Ad - I, Bd)
    num, den = ss2tf(Ac, Bc, Cd, Dd)
    return LTI(np.asarray(num).ravel(), np.asarray(den).ravel(), dt=None)
