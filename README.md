# nugap

**Detect condition-specific changes in *dynamics* using the Vinnicombe ν-gap metric.**

[![PyPI](https://img.shields.io/pypi/v/nugap.svg)](https://pypi.org/project/nugap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comparativechrono/nugap/blob/main/tutorial_nugap.ipynb)

`nugap` is a lightweight Python implementation of the **Vinnicombe ν-gap** — a
bounded (0–1) distance between linear dynamical systems from robust control
theory — together with the model identification and statistics needed to ask a
practical question of two-condition time-course data:

> **Which variables, or which relationships between them, change their *dynamical
> behaviour* between conditions?**

(for example wild type vs mutant, or untreated vs treated). Comparisons of
expression *level* answer a different question and are blind to changes in
timescale, gain or phase. The ν-gap is defined directly on dynamical models, so
it is not: two systems close in ν-gap behave the same way under feedback, and a
large ν-gap marks a genuine change in dynamics.

## Installation

```bash
pip install nugap
```

Requires Python ≥ 3.10. The core and pipelines depend only on **NumPy**, **SciPy**
and **pandas**; the plotting helpers additionally use **Matplotlib** and
**NetworkX**.

## Tutorial

A guided, runnable tutorial covers the metric, model fitting, and the two-condition
network comparison end to end:

- **Open in Colab:** https://colab.research.google.com/github/comparativechrono/nugap/blob/main/tutorial_nugap.ipynb
- **View on GitHub:** [`tutorial_nugap.ipynb`](https://github.com/comparativechrono/nugap/blob/main/tutorial_nugap.ipynb)

## Quick start

The ν-gap between two systems (coefficients in descending powers of `s`):

```python
from nugap import tf, nu_gap

P1 = tf([1], [1, 1])     # 1/(s+1)
P2 = tf([1], [1, 3])     # 1/(s+3)
nu_gap(P1, P2)           # -> 0.447   (0 = identical, 1 = maximally different)
```

Comparing a whole interaction network between two conditions, with replicate-based
significance:

```python
from nugap import compare_network

# data_A, data_B: dict {variable_name: array of shape (replicates, timepoints)}
# t: the common time vector
edges = compare_network(data_A, data_B, t, order=1, min_r2=0.5)

edges.query("q_global < 0.1")   # relationships rewired between conditions (FDR < 10%)
```

## What it provides

- **`nu_gap`** — the Vinnicombe ν-gap for SISO systems in continuous or discrete
  time, with an optional frequency-band restriction (`band=`) and a switchable
  winding-number test (`check_winding=`) for oscillatory data.
- **Model identification** — `fit_first_order`, `fit_model`, `fit_arx`,
  `fit_prony`, with simulation-based fit quality and an optional DC-gain floor
  (`min_dc_gain=`); plus the `dc_gain` helper.
- **`compare_conditions`** — per-variable comparison of dynamics between two
  conditions, with a fit-quality reliability flag.
- **`compare_network`** — pairwise interaction-network comparison: a low-order
  model is fitted to every ordered pair of variables in each condition, the ν-gap
  is taken between conditions, and significance comes from a replicate-derived
  empirical null with Benjamini–Hochberg FDR control.
- **Plotting** — `nugap.viz` (volcano plot, hub network, hub bar plot).

## How it works

For each variable or pairwise interaction, `nugap` fits a low-order linear
input–output model under each condition, then measures the ν-gap between the
fitted models. Because models are compared on mean-centred trajectories, the
metric reflects changes in the *relationship* — timescale, gain or phase — rather
than in absolute level. With biological replicates, the spread of within-condition
ν-gaps provides an empirical null and a per-edge noise floor, against which
between-condition changes are tested and FDR-controlled.

The models are single-input single-output, so an edge captures a pairwise
input–output relationship, not proven causation.

## Correctness

The ν-gap implementation is verified by several independent routes — exact
closed-form values, an algebraic invariance of the chordal metric, an independent
reference implementation, and the Vinnicombe robust-stability theorem — and is
cross-checked against MATLAB's `gapmetric` (Robust Control Toolbox), which it
reproduces to within ~10⁻⁶. The scripts are in [`validation/`](validation/), with
a fast subset run on every commit.

## Citing

If you use `nugap` in your work, please cite the software and the accompanying
methods paper:

```
Hearn, T. J. nugap: condition-specific changes in dynamical relationships via the
Vinnicombe ν-gap. Software archive: Zenodo, DOI: 10.5281/zenodo.20693443.

<methods paper reference — to be added>
```

## License

Released under the **MIT License**. Copyright © 2026 Tim Hearn. See
[`LICENSE`](LICENSE).
