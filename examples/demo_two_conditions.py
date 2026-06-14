"""End-to-end demo: simulate time-course data for many variables under two
conditions, where only a handful actually change their dynamics, then recover
them with the nu-gap pipeline.
"""

import numpy as np
from scipy.signal import lsim, TransferFunction

from nugap import compare_conditions


def step_response(tau, t, gain=1.0):
    """Response of a first-order system gain/(tau s + 1) to a unit step."""
    sys = TransferFunction([gain], [tau, 1.0])
    u = np.ones_like(t)
    _, y, _ = lsim(sys, U=u, T=t)
    return y


def main():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 120)

    n_vars = 300
    n_changed = 12
    changed = set(rng.choice(n_vars, n_changed, replace=False))

    data_A, data_B = {}, {}
    truth = {}
    for i in range(n_vars):
        name = f"var_{i:04d}"
        tau = rng.uniform(1.0, 4.0)
        gain = rng.uniform(0.5, 2.0)
        noise = 0.02

        yA = step_response(tau, t, gain) + rng.normal(0, noise, t.shape)
        if i in changed:
            # genuinely different dynamics under condition B
            tau_B = tau * rng.uniform(2.5, 4.0)
            gain_B = gain * rng.uniform(1.5, 2.5)
            yB = step_response(tau_B, t, gain_B) + rng.normal(0, noise, t.shape)
        else:
            # same dynamics, only measurement noise differs
            yB = step_response(tau, t, gain) + rng.normal(0, noise, t.shape)

        data_A[name] = yA
        data_B[name] = yB
        truth[name] = i in changed

    u = np.ones_like(t)  # known step stimulus under both conditions
    df = compare_conditions(
        data_A, data_B, t, u_A=u, u_B=u,
        orders=range(1, 4), method="arx", min_r2=0.9, progress=True,
    )

    df["truly_changed"] = df["variable"].map(truth)
    print("\nTop 15 variables by nu-gap:")
    print(df.head(15).to_string(index=False))

    # How well did the ranking separate changed from unchanged?
    top = df.head(n_changed)
    recovered = top["truly_changed"].sum()
    print(f"\nOf the {n_changed} truly-changed variables, "
          f"{recovered} are in the top {n_changed} by nu-gap.")
    med_changed = df.loc[df["truly_changed"], "nu_gap"].median()
    med_same = df.loc[~df["truly_changed"], "nu_gap"].median()
    print(f"median nu-gap  changed={med_changed:.3f}  unchanged={med_same:.3f}")


if __name__ == "__main__":
    main()
