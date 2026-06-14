"""Replicate-aware two-condition comparison.

With replicates you can measure how much nu-gap arises from noise alone
(comparing replicates *within* a condition) and test whether the
*between*-condition nu-gap is larger than that floor. This is far more
defensible than thresholding a single nu-gap value.

For each variable we:

1. fit a low-order model to every replicate of both conditions,
2. compute the full pairwise nu-gap matrix among all those models,
3. summarise within-condition pairs (the noise floor) vs between-condition
   pairs (the candidate change), and
4. optionally run a label-permutation test for a p-value.

Rank variables by ``separation`` (how far the between-condition nu-gap sits
above the within-condition noise floor), not by raw nu-gap.
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from .fitting import fit_model
from .metric import nu_gap


def _fit_replicates(t, reps, u, orders, method):
    """Fit a model to each replicate; return (models, r2s) skipping failures."""
    models, r2s = [], []
    for y in reps:
        try:
            fr = fit_model(t, y, u=u, orders=orders, method=method)
            models.append(fr.model)
            r2s.append(fr.r2)
        except Exception:
            continue
    return models, r2s


def _pairwise_matrix(models, n):
    R = len(models)
    M = np.full((R, R), np.nan)
    for i in range(R):
        for j in range(i + 1, R):
            d = nu_gap(models[i], models[j], n=n)
            M[i, j] = M[j, i] = d
    return M


def _perm_pvalue(M, nA, nB, observed, rng, n_perm):
    R = nA + nB
    idx = np.arange(R)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        A, B = idx[:nA], idx[nA:]
        between = [M[a, b] for a in A for b in B]
        within = [M[a, b] for a, b in itertools.combinations(A, 2)] + \
                 [M[a, b] for a, b in itertools.combinations(B, 2)]
        stat = np.nanmean(between) - np.nanmean(within)
        if stat >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def compare_variable_replicates(t, reps_A, reps_B, u=None,
                                orders=range(1, 3), method="auto",
                                n=2048, n_perm=0, rng=None):
    """Replicate-aware comparison for a single variable. Returns a dict."""
    if rng is None:
        rng = np.random.default_rng(0)
    mA, r2A = _fit_replicates(t, reps_A, u, orders, method)
    mB, r2B = _fit_replicates(t, reps_B, u, orders, method)
    nA, nB = len(mA), len(mB)
    if nA < 1 or nB < 1:
        raise RuntimeError("no replicate fitted in at least one condition")

    models = mA + mB
    M = _pairwise_matrix(models, n)
    Aidx, Bidx = range(nA), range(nA, nA + nB)

    between = np.array([M[a, b] for a in Aidx for b in Bidx])
    within = np.array(
        [M[a, b] for a, b in itertools.combinations(Aidx, 2)] +
        [M[a, b] for a, b in itertools.combinations(Bidx, 2)]
    )

    between_med = float(np.nanmedian(between))
    within_max = float(np.nanmax(within)) if within.size else 0.0
    within_med = float(np.nanmedian(within)) if within.size else 0.0
    separation = between_med - within_max

    out = {
        "between_median": between_med,
        "within_max": within_max,
        "within_median": within_med,
        "separation": separation,
        "n_reps_A": nA,
        "n_reps_B": nB,
        "mean_r2": float(np.mean(r2A + r2B)) if (r2A or r2B) else np.nan,
        "_between": between,
        "_within": within,
    }
    if n_perm and within.size:
        observed = float(np.nanmean(between) - np.nanmean(within))
        out["p_value"] = _perm_pvalue(M, nA, nB, observed, rng, n_perm)
    return out


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = idx.size
    prev = 1.0
    for rank, j in enumerate(order[::-1]):
        i = m - rank
        val = min(prev, p[j] * m / i)
        q[j] = prev = val
    return q


def compare_conditions_replicates(
    reps_A: dict,
    reps_B: dict,
    t,
    u=None,
    orders=range(1, 3),
    method="auto",
    n=2048,
    n_perm=0,
    global_null: bool = True,
    min_r2: float | None = None,
    seed: int = 0,
    progress: bool = False,
):
    """Replicate-aware comparison across all variables.

    Parameters
    ----------
    reps_A, reps_B : dict[str, array]
        var name -> 2D array (n_replicates x n_timepoints), or list of 1D
        replicate trajectories.
    t : array
        Common time vector.
    u : array or None
        Known stimulus (shared across variables/replicates).
    orders : iterable[int]
        Candidate model orders. Kept low (default 1-2) for short trajectories;
        also auto-capped to ~N//5 inside the fitter.
    n_perm : int
        Per-variable label-permutation iterations (0 = skip). Weak with few
        replicates; ``global_null`` is usually the better choice.
    global_null : bool
        If True (recommended for few replicates), pool the within-condition
        nu-gaps across *all* variables into one null distribution, then score
        each variable's between-condition median against it. Adds columns
        ``p_global`` and ``q_global`` (BH-FDR).
    min_r2 : float or None
        Flag variables whose mean fit R^2 is below this.

    Returns
    -------
    pandas.DataFrame sorted by significance / separation.
    """
    keys = list(reps_A.keys())
    rng = np.random.default_rng(seed)
    records = []
    global_within = []

    for i, key in enumerate(keys):
        if progress and i % 500 == 0:
            print(f"  {i}/{len(keys)}")
        try:
            res = compare_variable_replicates(
                t, np.asarray(reps_A[key]), np.asarray(reps_B[key]),
                u=u, orders=orders, method=method, n=n, n_perm=n_perm, rng=rng,
            )
            res["variable"] = key
            res["error"] = ""
            if global_null:
                w = res["_within"]
                global_within.append(w[~np.isnan(w)])
        except Exception as e:
            res = {"variable": key, "between_median": np.nan, "within_max": np.nan,
                   "within_median": np.nan, "separation": np.nan,
                   "n_reps_A": 0, "n_reps_B": 0, "mean_r2": np.nan,
                   "_between": np.array([]), "_within": np.array([]),
                   "error": repr(e)}
        records.append(res)

    df = pd.DataFrame.from_records(records)

    if global_null and global_within:
        pool = np.concatenate(global_within)
        pool = np.sort(pool[~np.isnan(pool)])
        m = pool.size

        def p_global(row):
            b = row["between_median"]
            if np.isnan(b) or m == 0:
                return np.nan
            # one-sided: P(noise gap >= observed between-condition median)
            ge = m - int(np.searchsorted(pool, b, side="left"))
            return (ge + 1) / (m + 1)

        df["p_global"] = df.apply(p_global, axis=1)
        df["q_global"] = _bh_fdr(df["p_global"].values)

    if min_r2 is not None:
        df["reliable"] = df["mean_r2"] >= min_r2

    df = df.drop(columns=["_between", "_within"], errors="ignore")
    sort_key = "q_global" if "q_global" in df.columns else "separation"
    ascending = sort_key == "q_global"
    front = ["variable", "separation", "between_median", "within_median"]
    for c in ("p_global", "q_global", "p_value"):
        if c in df.columns:
            front.append(c)
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest].sort_values(sort_key, ascending=ascending,
                                       na_position="last")
    return df.reset_index(drop=True)
