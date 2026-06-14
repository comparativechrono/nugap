"""Replicate-aware demo matching: few time points, a few replicates per
condition, a known step input. Only some variables truly change; we use the
within-condition nu-gap as the noise floor and a permutation test to flag the
real changes.
"""

import numpy as np
from scipy.signal import lsim, TransferFunction

from nugap import compare_conditions_replicates


def step_response(tau, t, gain=1.0):
    sys = TransferFunction([gain], [tau, 1.0])
    _, y, _ = lsim(sys, U=np.ones_like(t), T=t)
    return y


def main():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 20, 14)      # only 14 time points
    n_reps = 3                       # 3 replicates per condition
    noise = 0.03

    n_vars = 120
    n_changed = 8
    changed = set(rng.choice(n_vars, n_changed, replace=False))

    reps_A, reps_B, truth = {}, {}, {}
    for i in range(n_vars):
        name = f"var_{i:03d}"
        tau, gain = rng.uniform(1.5, 4.0), rng.uniform(0.6, 1.8)
        base = step_response(tau, t, gain)
        reps_A[name] = np.array([base + rng.normal(0, noise, t.shape)
                                 for _ in range(n_reps)])
        if i in changed:
            tau_B = tau * rng.uniform(2.5, 4.0)
            baseB = step_response(tau_B, t, gain * rng.uniform(1.4, 2.0))
        else:
            baseB = base
        reps_B[name] = np.array([baseB + rng.normal(0, noise, t.shape)
                                 for _ in range(n_reps)])
        truth[name] = i in changed

    u = np.ones_like(t)
    # With few time points, FIX the order low. Letting AIC pick higher orders
    # destabilises per-replicate fits and inflates the within-condition floor.
    df = compare_conditions_replicates(
        reps_A, reps_B, t, u=u,
        orders=[1], method="arx",
        n=2048, global_null=True, min_r2=0.8, progress=True,
    )
    df["truly_changed"] = df["variable"].map(truth)

    print("\nTop 12 by significance (global within-condition null):")
    cols = ["variable", "separation", "between_median", "within_median",
            "p_global", "q_global", "truly_changed"]
    print(df[cols].head(12).to_string(index=False))

    hits = df.head(n_changed)["truly_changed"].sum()
    print(f"\n{hits}/{n_changed} truly-changed variables in the top {n_changed}.")
    sig = df[df["q_global"] < 0.1]
    tp = sig["truly_changed"].sum()
    print(f"Flagged at FDR q<0.1: {len(sig)} "
          f"({tp} true positives, {len(sig) - tp} false positives).")


if __name__ == "__main__":
    main()
