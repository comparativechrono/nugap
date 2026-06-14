"""Batch pipeline: compare two conditions across many variables.

Given time-course data for the same set of variables under two conditions,
fit a model to each (variable, condition) trajectory and compute the nu-gap
between the two conditions for every variable. Variables with a large nu-gap
are the ones whose *dynamics* changed most between conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fitting import fit_model, FitResult
from .metric import nu_gap


def compare_variable(t, y_A, y_B, u_A=None, u_B=None, orders=range(1, 5),
                     method="auto", n=8192):
    """Fit a model under each condition for one variable and return the nu-gap.

    Returns a dict with the metric, both fit qualities, and chosen orders.
    """
    fit_A = fit_model(t, y_A, u=u_A, orders=orders, method=method)
    fit_B = fit_model(t, y_B, u=u_B, orders=orders, method=method)
    d, details = nu_gap(fit_A.model, fit_B.model, n=n, return_details=True)
    return {
        "nu_gap": d,
        "r2_A": fit_A.r2,
        "r2_B": fit_B.r2,
        "order_A": fit_A.order,
        "order_B": fit_B.order,
        "winding_ok": details["winding_ok"],
    }


def compare_conditions(
    data_A: dict,
    data_B: dict,
    t,
    u_A=None,
    u_B=None,
    orders=range(1, 5),
    method="auto",
    n=8192,
    min_r2: float | None = None,
    progress: bool = False,
):
    """Compare two conditions across all variables.

    Parameters
    ----------
    data_A, data_B : dict[str, array]
        Mapping variable name -> response trajectory under each condition.
        Both must share the same variable keys. Each trajectory is sampled at
        the times in ``t``. (For replicates, average them first, or pass the
        averaged trajectory.)
    t : array
        Common time vector for the trajectories.
    u_A, u_B : array or None
        Stimulus under each condition, if known (shared across variables).
    orders : iterable[int]
        Candidate model orders.
    method : 'auto' | 'prony' | 'arx'
    min_r2 : float or None
        If set, variables whose fit R^2 under either condition is below this
        are flagged (kept, but marked) so you can filter unreliable fits.
    progress : bool
        Print a counter every 500 variables.

    Returns
    -------
    pandas.DataFrame
        One row per variable, sorted by nu_gap descending. Columns include
        nu_gap, fit qualities, chosen orders, and (if min_r2 set) a
        'reliable' flag.
    """
    keys = list(data_A.keys())
    missing = set(keys) ^ set(data_B.keys())
    if missing:
        raise ValueError(f"variables differ between conditions: {sorted(missing)[:5]} ...")

    records = []
    for i, key in enumerate(keys):
        if progress and i % 500 == 0:
            print(f"  {i}/{len(keys)}")
        try:
            res = compare_variable(
                t, data_A[key], data_B[key], u_A=u_A, u_B=u_B,
                orders=orders, method=method, n=n,
            )
            res["variable"] = key
            res["error"] = ""
        except Exception as e:  # keep going across thousands of variables
            res = {"variable": key, "nu_gap": np.nan, "r2_A": np.nan,
                   "r2_B": np.nan, "order_A": -1, "order_B": -1,
                   "winding_ok": False, "error": repr(e)}
        records.append(res)

    df = pd.DataFrame.from_records(records)
    if min_r2 is not None:
        df["reliable"] = (df["r2_A"] >= min_r2) & (df["r2_B"] >= min_r2)
    cols = ["variable", "nu_gap", "r2_A", "r2_B", "order_A", "order_B",
            "winding_ok"] + (["reliable"] if min_r2 is not None else []) + ["error"]
    df = df[cols].sort_values("nu_gap", ascending=False, na_position="last")
    return df.reset_index(drop=True)
