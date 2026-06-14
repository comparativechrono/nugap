# nugap

A Python implementation of the **Vinnicombe nu-gap metric** (δν) and a
pipeline that uses it to find which variables change their *dynamics* between
two experimental conditions, from time-course data.

There is no nu-gap implementation in the standard Python control ecosystem —
it has lived almost exclusively in MATLAB's Robust Control Toolbox
(`gapmetric`). This package provides a tested, dependency-light one
(numpy + scipy + pandas only).

## What the nu-gap metric is

For two linear systems P1 and P2, δν(P1, P2) is a number in **[0, 1]**:

* **0** — identical dynamics,
* **near 1** — very different dynamics.

It is computed from the *chordal distance* between the two frequency responses,
gated by a winding-number (topological) condition. Unlike a naive comparison of
fitted parameters, it is a true metric, it is bounded, and it correctly treats
systems that look very different on paper but behave similarly (and vice versa).

Reference: G. Vinnicombe, *Frequency domain uncertainty and the graph
topology*, IEEE TAC 38 (1993) 1371–1383.

## Install

```bash
pip install nugap
```

## Usage

The metric on two systems directly:

```python
from nugap import tf, nu_gap

P1 = tf([1.0], [1.0, 1.0])      # 1/(s+1), continuous
P2 = tf([1.0], [1.0, 1.2])      # 1/(s+1.2)
print(nu_gap(P1, P2))           # ~0.07

# discrete systems use dt; the metric uses the unit circle automatically
Pd = tf([0.5], [1.0, -0.5], dt=0.1)
```

The full two-condition comparison:

```python
from nugap import compare_conditions

# data_A, data_B: dict mapping variable name -> trajectory (sampled at t)
# u_A, u_B: the known stimulus, if you have one (else omit -> Prony fit)
df = compare_conditions(
    data_A, data_B, t,
    u_A=u, u_B=u,           # drop these for output-only data
    orders=range(1, 5),     # candidate model orders (AIC-selected)
    method="arx",           # or "prony" (output-only), or "auto"
    min_r2=0.9,             # flag variables with poor fits
)
# df is sorted by nu_gap descending, with fit quality per condition
```

## Why might you like to use the nu-gap metirc?

In Omics such as transcriptomics it is normal to  have time-course data for thousands of variables under two conditions. The
analysis is a **pairwise dynamic network**: within each condition, every
variable is treated as a candidate input for every other variable, and a
first-order input->output model is fitted for each ordered pair (i -> j). Then
the nu-gap compares condition A's model with condition B's model for each edge.
Edges with a large nu-gap are interactions whose dynamics changed.

```python
from nugap import compare_network

# data_A, data_B: dict  variable name -> array (n_replicates x n_timepoints)
edges = compare_network(
    data_A, data_B, t,
    order=1,          # model poles: 1 = first-order, 2 = second-order
    n_zeros=None,     # numerator zeros (default 0 -> all-pole model)
    n=256,            # contour resolution; 256 is plenty for low order
    min_r2=0.5,       # only test pairs with a real relationship
    gate="either",    # how to combine the two conditions' fit quality
    global_null=True, # pool within-condition nu-gaps -> p_global, q_global
)
# one row per edge (source, target, nu_gap, within_median, separation,
# max_r2, q_global), sorted by significance. Flag changes with q_global < 0.1.
```

**`order` / `n_zeros`** choose the per-edge model. `order=1, n_zeros=0`
(default) is the first-order K/(τs+1); `order=2` is a two-pole system, with
`n_zeros=1` if you want a zero (the discrete analogue of MATLAB
`tfest(data, 2, 1)`). Higher order needs more time points per trajectory.

**`gate`** controls the fit-quality gate across the two conditions:
`"either"` keeps an edge if the relationship is well fit in at least one
condition (so relationships that appear or disappear are tested); `"both"`
requires a good fit in both conditions; `"mean"` uses the mean R² over all
replicate fits.

### Two things that matter a lot here

1. **Gate on fit quality (`min_r2`).** Most variable pairs have *no* real
   first-order relationship; those fits are meaningless and produce large,
   high-variance nu-gaps that would swamp the null. `compare_network` only
   tests an edge if a first-order relationship actually holds (R^2 above
   `min_r2`) in at least one condition. This is essential — without it nothing
   is detectable.

2. **Confounding.** Pairwise first-order identification assumes the i->j
   relationship is approximately self-contained. In a densely coupled system
   each output depends on many inputs, so a single pairwise model is
   misspecified and the within-condition noise floor rises. This is a property
   of the method, not the metric. Sparse or modular systems behave well; dense ones need care.

### Scale

N variables -> N*(N-1) ordered edges (a million at N=1000). Each fit is a
2-parameter least squares; the cost is the metric, kept cheap by the small
`n`. The per-edge work is independent, so wrap the edge loop in
`joblib.Parallel` / `multiprocessing` for real datasets, and/or pass
`include_pairs=` to test only a prescreened candidate set.


See `examples/demo_network.py` for a small validated example, and
`examples/demo_clock.py` for a full biological test case: synthetic circadian
RNA-seq for 50 genes (core clock genes, clock outputs, CLOCK-independent
rhythmic genes, and background) under wild-type vs **CLOCK knockout**. The KO
collapses the cell-autonomous oscillation (a Hopf bifurcation to a damped fixed
point) while CLOCK mRNA is still expressed; `compare_network` recovers the
collapse, flagging ~90% of clock/output edges as changed while leaving the
CLOCK-independent rhythmic edges (rhythmic in both conditions) and background
alone. The generator is `examples/clock_sim.py`.

### The null and short/flat conditions

`null_from_reliable_only=True` (default) builds the within-condition null only
from edges where a real relationship exists in that condition. This matters
whenever one condition loses dynamics (e.g. genes go flat in a knockout):
fitting noise-to-noise there produces large, meaningless within-condition
nu-gaps that would otherwise inflate the null and hide the real changes.

## Visualising results

`nugap.viz` (needs matplotlib + networkx: `pip install nugap[viz]`) provides
three views of a `compare_network` edge table:

```python
from nugap.viz import volcano, hub_barplot, hub_network

volcano(df, q_thresh=0.1)        # effect size (nu_gap) vs -log10 FDR; best overview
hub_barplot(df, top=20)          # genes ranked by # of significant changed edges
hub_network(df, top_hubs=15,     # directed graph of the most-rewired genes;
            node_groups=classes) # nodes sized by degree, edges coloured by nu_gap
```

The volcano is the recommended default (scale-independent, shows everything at
once). `hub_barplot` gives the node-level summary that is usually the most
interpretable. `hub_network` draws the most-rewired genes and the significant
changed edges among them as a directed graph (pass `node_groups`, a dict
gene -> label, to colour by class); `top_hubs` controls how many genes appear,
so you can keep it readable on dense results. `changed_edge_counts(df)` returns
the per-gene counts behind both hub views. Each plotting function returns a
matplotlib Axes so you can compose or restyle. `examples/demo_clock.py` writes
`clock_viz.png` (volcano + hubs) and `clock_hub_network.png`.

## Modules

* `nugap.metric` — the nu-gap metric, chordal distance, winding condition.
* `nugap.systems` — lightweight SISO LTI type (`tf`, `from_zpk`,
  `from_control`).
* `nugap.fitting` — identify discrete LTI models from data (ARX / Prony, with
  AIC order selection). **Swap this out** to match your MATLAB procedure.
* `nugap.network` — **pairwise (input->output) network comparison across
  conditions** with fit-quality gating and FDR; the main entry point for your
  application. `compare_network`, `fit_first_order`.
* `nugap.pipeline` / `nugap.replicates` — single-variable comparison (one model
  per variable), with and without replicates.

## Comparing conditions with replicates (recommended)

If you have replicates, use the replicate-aware pipeline. It fits a model to
every replicate, then uses *within*-condition nu-gaps (replicate vs replicate
of the same condition) as a noise floor and compares the *between*-condition
nu-gap against it:

```python
from nugap import compare_conditions_replicates

# reps_A[var], reps_B[var]: 2D array (n_replicates x n_timepoints)
df = compare_conditions_replicates(
    reps_A, reps_B, t, u=u,
    orders=[1],          # fix the order low for short series (see below)
    method="arx",
    global_null=True,    # pool within-condition gaps across all variables
)
# columns include between_median, within_median, p_global, q_global (BH-FDR)
# sorted by q_global ascending; flag changes with e.g. q_global < 0.1
```

`global_null=True` pools the within-condition noise across all of your
variables into one well-estimated null, which is far more powerful than the
~3 within-pairs a single variable provides.

### The most important knob: model order

With few time points, **fix the model order low (1, sometimes 2)** rather than
letting AIC roam. A too-high order makes the per-replicate fit unstable, so
replicates of *identical* dynamics produce a large nu-gap — that variance lands
directly in your noise floor and destroys sensitivity. In the bundled demo,
order 2 on 14 points gives a within-condition median nu-gap of ~0.16 and
detects nothing; order 1 gives ~0.04 and recovers every true change at
FDR < 0.1.

**Diagnostic:** look at `within_median` / the within-condition null. If it is
large (say > ~0.1), your fits are too unstable — lower the order, average
replicates, or get more points before trusting the between-condition results.

### Matching the common MATLAB `tfest` workflow

`tfest` needs an input and a response, and returns a *continuous* transfer
function; you then ran `gapmetric` on those. Here, fit with `method="arx"`
(input/output) and, if you want continuous-domain numbers to match MATLAB, map
each fitted model with `nugap.to_continuous` before comparing. The *ranking* of
variables is essentially the same in discrete or continuous form, so for
discovery you can stay discrete.

## Known limitations (v0.1)

* SISO only (one signal per variable). MIMO would need the determinant form of
  the winding condition.
* Systems with poles *exactly* on the stability boundary (pure integrators /
  undamped oscillators) are an edge case in the winding condition; fitted
  models from real data essentially never hit this.
* The fitting layer is intentionally basic. For best results, match the model
  class and order you have previosuly used in MATLAB.

## License

MIT — see the [LICENSE](LICENSE) file. 
