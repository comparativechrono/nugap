"""Synthetic circadian RNA-seq time courses: wild-type vs CLOCK knockout.

Biology encoded
---------------
The core clock is a transcription-translation feedback loop whose positive arm
is the BMAL1:CLOCK activator. Losing functional CLOCK collapses the loop: cells
move from a self-sustained oscillation to a damped approach to a steady state
(a Hopf bifurcation, amplitude death). We model the core pacemaker as a
Stuart-Landau oscillator whose bifurcation parameter lambda is set by CLOCK
functionality:

    WT  : lambda > 0  -> stable ~24 h limit cycle
    KO  : lambda < 0  -> rhythms damp toward a fixed point

CLOCK mRNA is still expressed in both conditions (present but non-functional in
the KO); it simply no longer drives the pacemaker.

Gene classes (50 genes total)
------------------------------
* core clock genes  -> rhythmic in WT with realistic acrophases; flat in KO
* clock output genes -> first-order dynamic responses to a clock-gene driver,
  giving genuine input->output edges that exist in WT and break in KO
* background genes   -> arrhythmic, identical between conditions (controls)

Output is log2-normalised expression, dict[gene] -> array (n_replicates x
n_timepoints), ready for nugap.compare_network.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

PERIOD = 23.8
OMEGA = 2 * np.pi / PERIOD


# ---------------------------------------------------------------- gene table
def gene_table():
    """Return list of (gene, klass, params). Sums to 50 genes."""
    genes = []

    # (name, acrophase in hours). E-box genes ~in phase; Bmal1/Ror antiphase.
    clock = [
        ("Arntl", 20.0, 0.9),   # Bmal1, strong, antiphase
        ("Npas2", 19.0, 0.7),
        ("Clock", 18.0, 0.12),  # expressed, only weakly rhythmic
        ("Per1", 4.0, 0.9),
        ("Per2", 5.0, 1.0),
        ("Per3", 5.5, 0.7),
        ("Cry1", 10.0, 0.8),
        ("Cry2", 7.0, 0.6),
        ("Nr1d1", 4.0, 1.0),    # Rev-erba
        ("Nr1d2", 5.0, 0.8),    # Rev-erbb
        ("Rora", 21.0, 0.5),
        ("Rorb", 22.0, 0.5),
        ("Dbp", 8.0, 1.0),      # canonical output, E-box
        ("Tef", 9.0, 0.7),
        ("Hlf", 9.5, 0.6),
        ("Nfil3", 22.0, 0.7),   # E4bp4, antiphase D-box
    ]
    for name, phi, amp in clock:
        genes.append((name, "clock", {"acrophase": phi, "amp": amp}))

    # output genes: (name, driver clock gene, time constant tau hours, amp)
    outputs = [
        ("Wee1", "Per2", 2.0, 0.9), ("Nampt", "Arntl", 3.0, 0.8),
        ("Pck1", "Dbp", 2.5, 1.0), ("G6pc", "Dbp", 3.0, 0.9),
        ("Hmgcr", "Nr1d1", 2.0, 0.8), ("Cyp7a1", "Dbp", 4.0, 1.1),
        ("Slc2a1", "Cry1", 2.5, 0.6), ("Mfsd2a", "Arntl", 3.0, 0.7),
        ("Avpr1a", "Per1", 2.0, 0.7), ("Gys2", "Dbp", 3.5, 0.8),
        ("Elovl3", "Tef", 2.5, 0.9), ("Cyp2b10", "Tef", 3.0, 0.8),
        ("Abca1", "Nr1d1", 2.5, 0.7), ("Lpl", "Hlf", 3.0, 0.7),
        ("Usp2", "Per2", 2.0, 0.8), ("Por", "Dbp", 3.0, 0.7),
        ("Fmo2", "Nfil3", 2.5, 0.7), ("Slc45a3", "Arntl", 2.0, 0.6),
    ]
    for name, drv, tau, amp in outputs:
        genes.append((name, "output", {"driver": drv, "tau": tau, "amp": amp}))

    # CLOCK-INDEPENDENT rhythmic genes: driven by systemic cues (feeding,
    # temperature, glucocorticoids) that persist in the KO. Rhythmic in BOTH
    # conditions -> their edges should NOT change (specificity control).
    external = [
        ("Hspa1b", 12.0, 0.9), ("Cirbp", 0.0, 0.8), ("Fkbp5", 13.0, 0.9),
        ("Mt1", 11.0, 0.7), ("Pnpla2", 14.0, 0.8),
    ]
    for name, phi, amp in external:
        genes.append((name, "external", {"acrophase": phi, "amp": amp}))

    # background / housekeeping: arrhythmic, identical between conditions
    background = ["Actb", "Gapdh", "Tbp", "Hprt", "B2m", "Ppia", "Rpl13a",
                  "Sdha", "Ubc", "Ywhaz", "Pgk1"]
    for name in background:
        genes.append((name, "background", {}))

    return genes  # 16 clock + 18 output + 5 external + 11 background = 50


# --------------------------------------------------------------- simulation
def _pacemaker(lmbda, t_eval, x0, y0):
    """Stuart-Landau oscillator. Returns x(t), y(t)."""
    def rhs(t, s):
        x, y = s
        rr = x * x + y * y
        return [(lmbda - rr) * x - OMEGA * y,
                (lmbda - rr) * y + OMEGA * x]
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [x0, y0],
                    t_eval=t_eval, rtol=1e-7, atol=1e-9, max_step=0.25)
    return sol.y[0], sol.y[1]


def _projection(x, y, acrophase):
    """Project the 2D pacemaker state onto a gene's acrophase direction."""
    ph = OMEGA * acrophase
    return np.cos(ph) * x + np.sin(ph) * y


def _first_order_filter(t, driver_signal, tau):
    """Low-pass filter the driver to make a phase-lagged output dynamic."""
    g = np.zeros_like(driver_signal)
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        g[k] = g[k - 1] + dt * (driver_signal[k - 1] - g[k - 1]) / tau
    return g


def generate(condition, genes, t_sample, n_reps=4, noise=0.25,
             seed=0, settle=240.0):
    """Generate dict[gene] -> (n_reps x n_timepoints) log2 expression.

    condition: 'WT' or 'KO'.
    """
    rng = np.random.default_rng(seed)
    lmbda = 1.0 if condition == "WT" else -0.3

    # dense grid: long settle for WT to reach the limit cycle, then the window
    t0 = -settle
    t_dense = np.linspace(t0, t_sample[-1], int((t_sample[-1] - t0) / 0.25) + 1)

    if condition == "WT":
        x0, y0 = 1.0, 0.0
    else:
        # KO is induced at t=0 from a point on the former cycle, then damps
        x0, y0 = 1.0, 0.0
        t_dense = np.linspace(0.0, t_sample[-1], int(t_sample[-1] / 0.25) + 1)

    x, y = _pacemaker(lmbda, t_dense, x0, y0)

    # interpolate pacemaker onto a fine grid covering the sample window
    def at(ts, arr):
        return np.interp(ts, t_dense, arr)

    baselines = {}  # stable per-gene baseline log2 level
    rng_base = np.random.default_rng(12345)
    for (name, klass, p) in genes:
        baselines[name] = rng_base.uniform(3.0, 9.0)  # log2 TPM-ish

    # clean (noise-free) trajectories on the sample grid
    clean = {}
    xs, ys = at(t_sample, x), at(t_sample, y)
    # systemic (CLOCK-independent) oscillator: identical in WT and KO
    sx, sy = np.cos(OMEGA * t_sample), np.sin(OMEGA * t_sample)
    for (name, klass, p) in genes:
        if klass == "clock":
            proj = _projection(xs, ys, p["acrophase"])
            clean[name] = baselines[name] + p["amp"] * proj
        elif klass == "external":
            proj = _projection(sx, sy, p["acrophase"])
            clean[name] = baselines[name] + p["amp"] * proj
        elif klass == "background":
            clean[name] = np.full_like(t_sample, baselines[name])
        # outputs handled below (need driver clean signal on dense grid)

    # outputs: filter the driver's clean continuous signal, then sample
    for (name, klass, p) in genes:
        if klass != "output":
            continue
        drv = p["driver"]
        # driver clean signal on the dense grid
        drv_phi = next(g[2]["acrophase"] for g in genes
                       if g[0] == drv and g[1] == "clock")
        drv_dense = _projection(x, y, drv_phi)
        g_dense = _first_order_filter(t_dense, drv_dense, p["tau"])
        clean[name] = baselines[name] + p["amp"] * at(t_sample, g_dense)

    # add replicate noise
    data = {}
    for (name, klass, p) in genes:
        reps = np.array([clean[name] + rng.normal(0, noise, t_sample.shape)
                         for _ in range(n_reps)])
        data[name] = reps
    return data


def make_dataset(n_reps=4, dt_h=2.0, span_h=48.0, noise=0.25, seed=1):
    genes = gene_table()
    t = np.arange(0.0, span_h + 1e-9, dt_h)
    wt = generate("WT", genes, t, n_reps=n_reps, noise=noise, seed=seed)
    ko = generate("KO", genes, t, n_reps=n_reps, noise=noise, seed=seed + 100)
    meta = {name: klass for (name, klass, _) in genes}
    return wt, ko, t, meta, genes
