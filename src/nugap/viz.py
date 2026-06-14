"""Visualisations for nu-gap results.

These work on the edge table returned by ``compare_network`` (and, for the
volcano, on the per-variable tables too). matplotlib is required; install with
``pip install nugap[viz]``.

Functions
---------
volcano(df)        effect size (nu_gap) vs significance (-log10 q) overview
hub_network(df)    directed graph of the most-rewired hub genes (networkx)
hub_barplot(df)    genes ranked by number of significant changed edges
changed_edge_counts(df)   the per-gene counts behind hub_barplot
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for nugap.viz. Install with "
            "`pip install nugap[viz]` or `pip install matplotlib`.") from e


def _significant(df, q_col, q_thresh, nu_col, nu_thresh):
    """Boolean mask of 'changed' edges, by FDR if available else by nu_gap."""
    if q_col in df.columns and q_thresh is not None:
        return df[q_col] < q_thresh
    if nu_thresh is not None:
        return df[nu_col] > nu_thresh
    raise ValueError(
        f"need either '{q_col}' with q_thresh, or nu_thresh on '{nu_col}'")


# --------------------------------------------------------------- volcano
def volcano(df, value_col="nu_gap", q_col="q_global", q_thresh=0.1,
            label_top=10, ax=None, title="nu-gap volcano"):
    """Scatter of effect size (x) vs -log10(FDR) (y).

    Significant edges (q < q_thresh) are highlighted; the top edges by effect
    size are labelled. Returns the matplotlib Axes.
    """
    plt = _plt()
    if q_col not in df.columns:
        raise ValueError(f"'{q_col}' not in dataframe; run compare_network "
                         f"with global_null=True")
    d = df.dropna(subset=[value_col, q_col]).copy()
    qsafe = d[q_col].clip(lower=d[q_col][d[q_col] > 0].min() / 10 if (d[q_col] > 0).any() else 1e-12)
    d["_y"] = -np.log10(qsafe)
    sig = d[q_col] < q_thresh

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d.loc[~sig, value_col], d.loc[~sig, "_y"], s=10,
               c="#bdbdbd", label="n.s.", edgecolors="none")
    ax.scatter(d.loc[sig, value_col], d.loc[sig, "_y"], s=14,
               c="#c2185b", label=f"q<{q_thresh}", edgecolors="none")
    ax.axhline(-np.log10(q_thresh), ls="--", lw=0.8, c="grey")
    ax.set_xlabel(f"{value_col}  (effect size, 0=unchanged \u2192 1=very different)")
    ax.set_ylabel(r"$-\log_{10}$ FDR ($q$)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)

    if label_top and "source" in d.columns and "target" in d.columns:
        top = d[sig].nlargest(label_top, value_col)
        for _, r in top.iterrows():
            ax.annotate(f"{r['source']}\u2192{r['target']}",
                        (r[value_col], r["_y"]), fontsize=6,
                        xytext=(2, 2), textcoords="offset points")
    return ax


# --------------------------------------------------------------- hub network
def hub_network(df, top_hubs=15, q_col="q_global", q_thresh=0.1,
                nu_col="nu_gap", nu_thresh=None, node_groups=None,
                layout="spring", seed=0, ax=None,
                title="Rewired hub network"):
    """Directed graph of the most-rewired genes and the significant changed
    edges among them.

    Nodes are the ``top_hubs`` genes with the most significant changed edges,
    sized by that count. Directed edges are the significant changes among those
    hubs, coloured by nu_gap (effect size). Pass ``node_groups`` (a dict
    gene -> group label, e.g. gene class) to colour and label node groups.

    Requires networkx. Returns the matplotlib Axes.
    """
    plt = _plt()
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise ImportError("hub_network needs networkx: pip install networkx") from e

    sig = _significant(df, q_col, q_thresh, nu_col, nu_thresh)
    s = df.loc[sig]
    counts = (pd.concat([s["source"], s["target"]]).value_counts())
    hubs = list(counts.head(top_hubs).index)
    sub = s[s["source"].isin(hubs) & s["target"].isin(hubs)]

    G = nx.DiGraph()
    for h in hubs:
        G.add_node(h, degree=int(counts.get(h, 0)))
    for _, r in sub.iterrows():
        G.add_edge(r["source"], r["target"], nu=float(r[nu_col]))

    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed, k=1.2)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    degs = np.array([G.nodes[n]["degree"] for n in G.nodes])
    sizes = 150 + 60 * degs

    # node colours: by group if provided, else uniform
    if node_groups is not None:
        groups = [node_groups.get(n, "other") for n in G.nodes]
        uniq = sorted(set(groups))
        palette = plt.get_cmap("tab10")
        cmap_g = {g: palette(i % 10) for i, g in enumerate(uniq)}
        node_color = [cmap_g[g] for g in groups]
    else:
        node_color = "#1b7837"

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_color,
                           edgecolors="white", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    nus = np.array([G[u][v]["nu"] for u, v in G.edges])
    ecmap = plt.get_cmap("plasma")
    if len(G.edges):
        nx.draw_networkx_edges(
            G, pos, edge_color=nus, edge_cmap=ecmap, edge_vmin=0, edge_vmax=1,
            width=1.0 + 2.0 * nus, alpha=0.7, arrows=True, arrowsize=9,
            connectionstyle="arc3,rad=0.08", ax=ax)
        sm = plt.cm.ScalarMappable(cmap=ecmap,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("nu_gap (edge change)")

    if node_groups is not None:
        handles = [plt.Line2D([0], [0], marker="o", ls="", markersize=8,
                              markerfacecolor=cmap_g[g], markeredgecolor="white",
                              label=g) for g in uniq]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")

    ax.set_title(title)
    ax.axis("off")
    return ax


# --------------------------------------------------------------- hubs
def changed_edge_counts(df, q_col="q_global", q_thresh=0.1, nu_col="nu_gap",
                        nu_thresh=None):
    """Per-gene count of significant changed edges (as source or target)."""
    sig = _significant(df, q_col, q_thresh, nu_col, nu_thresh)
    s = df.loc[sig]
    return (pd.concat([s["source"], s["target"]]).value_counts()
            .rename("changed_edges"))


def hub_barplot(df, top=20, q_col="q_global", q_thresh=0.1, nu_col="nu_gap",
                nu_thresh=None, ax=None, title="Most-rewired genes"):
    """Horizontal bar chart of genes with the most significant changed edges."""
    plt = _plt()
    counts = changed_edge_counts(df, q_col, q_thresh, nu_col, nu_thresh).head(top)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.32 * len(counts) + 1))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#1b7837")
    ax.set_xlabel("number of significant changed edges")
    ax.set_title(title)
    return ax
