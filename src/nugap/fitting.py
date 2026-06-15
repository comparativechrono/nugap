"""Fit discrete-time LTI models to time-course data.

This is the part of the pipeline that turns a measured trajectory into an
``LTI`` system that the nu-gap metric can compare. It is deliberately
separated from the metric so you can swap in your own identification routine
(e.g. to mirror exactly what MATLAB's System Identification Toolbox did).

Two cases are handled:

* **Input/output data** (you applied a known stimulus ``u`` and recorded the
  response ``y``): an ARX model is fitted by linear least squares, giving a
  discrete transfer function B(z)/A(z).

* **Output-only data** (you only have the response ``y``, e.g. a signal that
  rises and relaxes after a perturbation): Prony's method fits ``y`` as the
  impulse response of a discrete LTI system, recovering its poles (modes) and
  a numerator.

Sampled data -> discrete-time models is the natural and numerically clean
choice; the nu-gap metric evaluates these on the unit circle.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .systems import LTI


@dataclass
class FitResult:
    model: LTI
    order: int
    r2: float          # fit quality of the simulated/predicted output
    method: str
    sample_time: float


def _sample_time(t) -> float:
    t = np.asarray(t, dtype=float)
    dts = np.diff(t)
    dt = float(np.median(dts))
    if dt <= 0:
        raise ValueError("time vector must be increasing")
    return dt


def _simulate(num, den, u, y0len):
    """Simulate a discrete tf B/A driven by u (impulse if u is None)."""
    import warnings
    from scipy.signal import dlsim, TransferFunction, BadCoefficients

    if u is None:
        u = np.zeros(y0len)
        u[0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BadCoefficients)
        sysd = TransferFunction(num, den, dt=1.0)
        _, yout = dlsim(sysd, u)
    return np.asarray(yout).ravel()


def _r2(y, yhat):
    y = np.asarray(y, dtype=float)
    n = min(len(y), len(yhat))
    y, yhat = y[:n], yhat[:n]
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_prony(t, y, order: int):
    """Fit y as the impulse response of a discrete LTI of given order.

    Returns (num, den) discrete polynomial coefficients (highest power first).
    """
    y = np.asarray(y, dtype=float).ravel()
    N = len(y)
    na = order
    if N <= 2 * na:
        raise ValueError("not enough samples for the requested order")

    # Linear prediction: y[k] = -sum_{j=1..na} a_j y[k-j]
    rows = []
    rhs = []
    for k in range(na, N):
        rows.append([-y[k - j] for j in range(1, na + 1)])
        rhs.append(y[k])
    A = np.asarray(rows)
    b = np.asarray(rhs)
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    den = np.concatenate([[1.0], a])  # [1, a1, ..., a_na]

    # Numerator from the convolution relation treating y as impulse response.
    nb = na
    num = np.zeros(nb + 1)
    for k in range(nb + 1):
        s = 0.0
        for j in range(k + 1):
            if j < len(den):
                s += den[j] * y[k - j]
        num[k] = s
    return num, den


def fit_arx(t, y, u, na: int, nb: int | None = None, nk: int = 1):
    """Fit an ARX model A(z) y = B(z) u by least squares.

    Returns (num, den) of the discrete transfer function B(z)/A(z).
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    N = len(y)
    if nb is None:
        nb = na
    start = max(na, nb + nk)
    rows, rhs = [], []
    for k in range(start, N):
        row = [-y[k - j] for j in range(1, na + 1)]
        row += [u[k - nk - i] for i in range(0, nb + 1)]
        rows.append(row)
        rhs.append(y[k])
    M = np.asarray(rows)
    b = np.asarray(rhs)
    theta, *_ = np.linalg.lstsq(M, b, rcond=None)
    a = theta[:na]
    bb = theta[na:]
    den = np.concatenate([[1.0], a])
    num = np.concatenate([np.zeros(nk), bb])  # account for input delay
    return num, den


def _cap_orders(orders, N: int):
    """Limit candidate orders so we never overfit short trajectories.

    A model of order p has ~2p+1 free parameters; with N samples we keep
    roughly 5 samples per parameter, so p_max ~ N//5 (at least 1).
    """
    p_max = max(1, N // 5)
    capped = [o for o in orders if o <= p_max]
    return capped or [1]


def fit_arx_fast(t, y, u, na: int, nb: int = 0, nk: int = 1, dt=None):
    """Fast ARX fit of arbitrary order with a recursive simulation R^2.

    Fits A(z) y = B(z) u with ``na`` poles, ``nb+1`` numerator taps and input
    delay ``nk`` by linear least squares, then simulates the model from u
    (warm-started on the first samples) to score the fit. Returns (LTI, r2).

    Designed to be called millions of times for pairwise identification, so it
    avoids scipy per call. ``na=1, nb=0`` reproduces the first-order model
    K/(tau s + 1) (MATLAB ``tfest(data, 1, 0)``); ``na=2, nb=0`` is the
    canonical two-pole, no-zero second-order system.
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    if dt is None:
        dt = _sample_time(t)
    na, nb, nk = int(na), int(nb), int(nk)
    N = len(y)
    start = max(na, nb + nk)
    n_params = na + nb + 1
    if N - start < n_params:
        raise ValueError(
            f"not enough samples ({N}) for order na={na}, nb={nb}")

    rows, rhs = [], []
    for k in range(start, N):
        row = [-y[k - i] for i in range(1, na + 1)]
        row += [u[k - nk - i] for i in range(0, nb + 1)]
        rows.append(row)
        rhs.append(y[k])
    theta, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    a = theta[:na]
    b = theta[na:]

    den = np.concatenate([[1.0], a])
    num = np.concatenate([np.zeros(nk), b])  # input delay

    # simulate the model from u, warm-started with measured y on [0, start)
    yhat = np.empty(N)
    yhat[:start] = y[:start]
    for k in range(start, N):
        val = 0.0
        for i in range(1, na + 1):
            val -= a[i - 1] * yhat[k - i]
        for i in range(0, nb + 1):
            val += b[i] * u[k - nk - i]
        yhat[k] = val

    ys, yh = y[start:], yhat[start:]
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = float(1.0 - np.sum((ys - yh) ** 2) / ss_tot) if ss_tot > 0 else 0.0
    return LTI(num, den, dt=dt), r2


def dc_gain(sys) -> float:
    """Steady-state (zero-frequency) input->output gain |H| of an LTI system.

    Discrete: |H(z=1)|. Continuous: |H(s=0)|. This is the magnitude of the
    static transfer from input to output; a value near zero means the input
    barely influences the output's steady state (a candidate "false"
    input-output link). Returns +inf if the system has an integrator
    (pole at the DC point).
    """
    pt = np.array([1.0 + 0j]) if sys.is_discrete() else np.array([0.0 + 0j])
    with np.errstate(divide="ignore", invalid="ignore"):
        val = sys.freqresp(pt)[0]
    return float(np.abs(val))


def fit_first_order(t, y, u, dt=None, min_dc_gain=None):
    """First-order ARX fit (one pole, no zero). Thin wrapper around
    ``fit_arx_fast`` with na=1, nb=0, nk=1. Returns (LTI, r2).

    Parameters
    ----------
    min_dc_gain : float or None
        If given, models whose absolute DC gain is at or below this value are
        treated as failed input-output fits and their r2 is set to 0.0, so
        they are rejected by any downstream reliability gate. This mirrors the
        low-DC-gain filter used by DyDE (Mombaerts et al. 2019) to discard
        spurious links where the input does not drive the output. Default
        ``None`` leaves the fit untouched (the stringent default used in the
        main analysis relies on the simulation-R^2 gate alone).
    """
    sys, r2 = fit_arx_fast(t, y, u, na=1, nb=0, nk=1, dt=dt)
    if min_dc_gain is not None and dc_gain(sys) <= float(min_dc_gain):
        r2 = 0.0
    return sys, r2


def fit_model(
    t,
    y,
    u=None,
    orders=range(1, 5),
    method: str = "auto",
) -> FitResult:
    """Fit a discrete LTI model, selecting the order by AIC.

    Parameters
    ----------
    t : array     time stamps (used only to get the sample time)
    y : array     measured response
    u : array or None   stimulus, if known
    orders : iterable of int   candidate model orders to try. Automatically
        capped to ~N//5 so short trajectories are not overfitted.
    method : 'auto' | 'prony' | 'arx'
        'auto' uses ARX when u is given, Prony otherwise.

    Returns
    -------
    FitResult
    """
    y = np.asarray(y, dtype=float).ravel()
    dt = _sample_time(t)
    N = len(y)
    orders = _cap_orders(list(orders), N)

    if method == "auto":
        method = "arx" if u is not None else "prony"

    best = None
    for order in orders:
        try:
            if method == "prony":
                num, den = fit_prony(t, y, order)
                yhat = _simulate(num, den, None, N)
                nparams = 2 * order + 1
            elif method == "arx":
                num, den = fit_arx(t, y, u, na=order)
                yhat = _simulate(num, den, np.asarray(u, float).ravel(), N)
                nparams = 2 * order + 1
            else:
                raise ValueError(f"unknown method {method!r}")
        except Exception:
            continue

        n = min(len(y), len(yhat))
        sse = np.sum((y[:n] - yhat[:n]) ** 2)
        if sse <= 0 or not np.isfinite(sse):
            continue
        aic = N * np.log(sse / N) + 2 * nparams
        r2 = _r2(y, yhat)
        cand = (aic, order, num, den, r2)
        if best is None or aic < best[0]:
            best = cand

    if best is None:
        raise RuntimeError("model fitting failed for all candidate orders")

    _, order, num, den, r2 = best
    model = LTI(num, den, dt=dt)
    return FitResult(model=model, order=order, r2=r2, method=method, sample_time=dt)
