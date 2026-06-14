"""Pairwise-network demo with *isolated* input->output pairs (no confounding),
to validate the edge-wise pipeline unambiguously.

Setup: sources x0,x2,x4,x6 are independent AR(1) signals. Each target is a
first-order response to its source: x1<-x0, x3<-x2, x5<-x4, x7<-x6. Between
conditions, two of those edge dynamics change (x0->x1 and x4->x5). The pipeline
should flag exactly those two among all 56 ordered pairs.
"""

import numpy as np
from nugap import compare_network


def ar1(n_t, rng, phi=0.6, noise=0.3):
    s = np.zeros(n_t)
    for k in range(1, n_t):
        s[k] = phi * s[k - 1] + rng.normal(0, noise)
    return s


def first_order_response(s, a, b, rng, obs=0.02):
    y = np.zeros_like(s)
    for k in range(1, len(s)):
        y[k] = a * y[k - 1] + b * s[k - 1]
    return y + rng.normal(0, obs, len(s))


def main():
    rng = np.random.default_rng(5)
    n_t, n_reps = 60, 5
    names = [f"x{i}" for i in range(8)]
    edges = {1: 0, 3: 2, 5: 4, 7: 6}  # target -> source

    params_A = {1: (0.6, 0.5), 3: (0.5, 0.6), 5: (0.7, 0.4), 7: (0.4, 0.5)}
    params_B = dict(params_A)
    params_B[1] = (0.2, -0.4)   # x0 -> x1 changes
    params_B[5] = (0.85, 0.7)   # x4 -> x5 changes
    changed_edges = {("x0", "x1"), ("x4", "x5")}

    def make(params):
        d = {nm: [] for nm in names}
        for _ in range(n_reps):
            sig = {}
            for src in set(edges.values()):
                sig[src] = ar1(n_t, rng)
            for tgt, src in edges.items():
                a, b = params[tgt]
                sig[tgt] = first_order_response(sig[src], a, b, rng)
            for i, nm in enumerate(names):
                d[nm].append(sig[i])
        return {nm: np.array(v) for nm, v in d.items()}

    data_A, data_B = make(params_A), make(params_B)
    t = np.arange(n_t) * 1.0

    df = compare_network(data_A, data_B, t, n=256, min_r2=0.5,
                         global_null=True, progress=False)
    df["edge"] = list(zip(df["source"], df["target"]))
    df["truly_changed"] = df["edge"].isin(changed_edges)

    print("Top 8 edges by significance:")
    cols = ["source", "target", "nu_gap", "within_median", "separation",
            "mean_r2", "q_global", "truly_changed"]
    print(df[cols].head(8).to_string(index=False))

    flagged = df[df["q_global"] < 0.1]
    tp = int(flagged["truly_changed"].sum())
    print(f"\nTrue changed edges: {sorted(changed_edges)}")
    print(f"Flagged at FDR q<0.1: {len(flagged)} "
          f"({tp} true positives, {len(flagged) - tp} false positives)")
    print("(real edges that did NOT change should have low nu_gap and not be flagged)")
    for e in [("x0","x1"),("x2","x3"),("x4","x5"),("x6","x7")]:
        row = df[df["edge"] == e]
        if len(row):
            print(f"  {e}: nu_gap {row['nu_gap'].iloc[0]:.3f}  "
                  f"q {row['q_global'].iloc[0]:.3g}  changed={e in changed_edges}")


if __name__ == "__main__":
    main()
