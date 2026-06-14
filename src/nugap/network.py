"""Pairwise dynamic-network comparison.

Workflow: within each condition, treat every variable as a candidate input for
every other variable and fit a first-order input->output model. Then, for each
ordered pair (i -> j), use the nu-gap to compare the condition-A model with the
condition-B model. Edges with a large nu-gap are interactions whose dynamics
changed between conditions.

With replicates, the within-condition nu-gap (replicate vs replicate of the
same condition) gives a per-edge noise floor; pooled across all edges it forms
a well-estimated null for FDR-controlled discovery across the millions of
edges.

Scale note: N variables -> N*(N-1) ordered edges. Each fit is a trivial
2-parameter least squares; the cost is the metric. First-order systems are
smooth on the unit circle, so a small ``n`` (default 256) is accurate and keeps
this tractable. For thousands of variables, run edge batches in parallel
(the per-edge work is independent) and/or prescreen edges.
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from .fitting import fit_arx_fast, _sample_time
from .metric import nu_gap
from .replicates import _bh_fdr


def _center(arr):
    """Subtract each replicate's mean (these models assume no offset)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr - np.mean(arr)
    return arr - arr.mean(axis=1, keepdims=True)


def _edge_models(reps_in, reps_out, t, dt, na, nb):
    """Fit a model per replicate for one (input->output) edge."""
    models, r2s = [], []
    for u, y in zip(reps_in, reps_out):
        try:
            m, r2 = fit_arx_fast(t, y, u, na=na, nb=nb, nk=1, dt=dt)
            if not m.is_stable() and np.max(np.abs(m.poles)) > 5:
                continue  # discard wildly diverging fits
            models.append(m)
            r2s.append(r2)
        except Exception:
            continue
    return models, r2s


def _gap_pairs(models, n):
    R = len(models)
    g = {}
    for i, j in itertools.combinations(range(R), 2):
        g[(i, j)] = nu_gap(models[i], models[j], n=n)
    return g


def compare_network(
    data_A: dict,
    data_B: dict,
    t,
    order: int = 1,
    n_zeros: int | None = None,
    n: int = 256,
    center: bool = True,
    min_r2: float | None = 0.5,
    gate: str = "either",
    null_from_reliable_only: bool = True,
    include_pairs=None,
    global_null: bool = True,
    progress: bool = False,
):
    """Compare the pairwise interaction network across conditions.

    Parameters
    ----------
    data_A, data_B : dict[str, array]
        variable name -> trajectories. Either 1D (single record) or 2D
        (n_replicates x n_timepoints). With replicates a per-edge noise floor
        is computed.
    t : array
        Common time vector.
    order : int
        Number of poles of the per-edge model. 1 = first-order K/(tau s + 1)
        (default), 2 = second-order. Needs more time points at higher order.
    n_zeros : int or None
        Number of zeros in the numerator. Default None -> 0 (all-pole model:
        first-order = 1 pole/0 zeros, second-order = 2 poles/0 zeros). Set to 1
        for e.g. a two-pole/one-zero model (MATLAB ``tfest(data, 2, 1)``).
    n : int
        Metric contour resolution. 256 is accurate for low-order systems.
    center : bool
        Subtract each trajectory's mean before fitting (recommended).
    min_r2 : float or None
        Fit-quality gate threshold. Only edges where a real relationship holds
        are tested; None disables gating.
    gate : {'either', 'both', 'mean'}
        How the per-condition fit qualities are combined for the gate:
          * 'either' (default) - keep the edge if the best replicate fit in
            EITHER condition reaches min_r2 (catches relationships that appear
            or disappear between conditions).
          * 'both' - require the best replicate fit in BOTH conditions to reach
            min_r2 (a relationship present in both, whose dynamics may differ).
          * 'mean' - use the mean R^2 over all replicate fits of both
            conditions.
    include_pairs : iterable[(src, tgt)] or None
        Restrict to specific ordered edges (e.g. a prescreened candidate set).
        If None, all ordered pairs i != j are tested.
    global_null : bool
        Pool within-condition nu-gaps across all edges into one null and report
        p_global / q_global (BH-FDR).

    Returns
    -------
    pandas.DataFrame, one row per edge, sorted by significance (or nu_gap).
    """
    if gate not in ("either", "both", "mean"):
        raise ValueError("gate must be 'either', 'both', or 'mean'")
    na = int(order)
    nb = 0 if n_zeros is None else int(n_zeros)
    variables = list(data_A.keys())
    dt = _sample_time(t)

    def as_reps(d):
        out = {}
        for k, v in d.items():
            v = np.asarray(v, dtype=float)
            if v.ndim == 1:
                v = v[None, :]
            out[k] = _center(v) if center else v
        return out

    A = as_reps(data_A)
    B = as_reps(data_B)

    if include_pairs is None:
        pairs = [(i, j) for i in variables for j in variables if i != j]
    else:
        pairs = list(include_pairs)

    records = []
    global_within = []
    for e, (src, tgt) in enumerate(pairs):
        if progress and e % 5000 == 0:
            print(f"  {e}/{len(pairs)} edges")

        mA, r2A = _edge_models(A[src], A[tgt], t, dt, na, nb)
        mB, r2B = _edge_models(B[src], B[tgt], t, dt, na, nb)
        if not mA or not mB:
            continue

        # Only test an edge where a real relationship exists; otherwise the
        # meaningless, high-variance fits would pollute the null. How the two
        # conditions' fit qualities are combined is controlled by `gate`.
        best_A = max(r2A, default=-np.inf)
        best_B = max(r2B, default=-np.inf)
        if gate == "either":
            gate_stat = max(best_A, best_B)
        elif gate == "both":
            gate_stat = min(best_A, best_B)
        else:  # 'mean'
            gate_stat = float(np.mean(r2A + r2B)) if (r2A or r2B) else -np.inf
        if min_r2 is not None and gate_stat < min_r2:
            continue
        max_r2 = max(best_A, best_B)

        models = mA + mB
        nA = len(mA)
        idxA, idxB = range(nA), range(nA, len(models))
        gp = _gap_pairs(models, n)

        def get(i, j):
            return gp[(i, j)] if i < j else gp[(j, i)]

        between = [get(a, b) for a in idxA for b in idxB]
        within_A = [get(a, b) for a, b in itertools.combinations(idxA, 2)]
        within_B = [get(a, b) for a, b in itertools.combinations(idxB, 2)]
        within = within_A + within_B

        between_med = float(np.median(between))
        within_med = float(np.median(within)) if within else 0.0
        rec = {
            "source": src, "target": tgt,
            "nu_gap": between_med,
            "within_median": within_med,
            "separation": between_med - within_med,
            "n_reps": min(nA, len(mB)),
            "max_r2": float(max_r2),
            "mean_r2": float(np.mean(r2A + r2B)) if (r2A or r2B) else np.nan,
        }
        records.append(rec)
        if global_null:
            # A within-condition gap belongs in the null only if a real
            # relationship exists in that condition; otherwise we would be
            # pooling the variance of fitting noise to noise, which inflates
            # the null and destroys power (e.g. a condition where the gene
            # went flat). Set null_from_reliable_only=False to pool all.
            thr = -np.inf if (min_r2 is None or not null_from_reliable_only) else min_r2
            if best_A >= thr:
                global_within.extend(within_A)
            if best_B >= thr:
                global_within.extend(within_B)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    if global_null and global_within:
        pool = np.sort(np.asarray(global_within, dtype=float))
        m = pool.size
        ge = m - np.searchsorted(pool, df["nu_gap"].values, side="left")
        df["p_global"] = (ge + 1) / (m + 1)
        df["q_global"] = _bh_fdr(df["p_global"].values)
        df = df.sort_values("q_global", ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("nu_gap", ascending=False).reset_index(drop=True)
    return df
