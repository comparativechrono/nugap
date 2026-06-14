"""Generate the synthetic WT vs CLOCK-KO circadian RNA-seq dataset, save it to
CSV, plot representative genes, and run the nu-gap network comparison.

Outputs (written next to this script, under ./clock_output/):
    clock_rnaseq_WT.csv, clock_rnaseq_KO.csv   long-format log2 expression
    clock_gene_metadata.csv                    gene -> class (+ params)
    clock_edges_results.csv                    compare_network output
    clock_tracks.png                           example tracks, WT vs KO
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clock_sim import make_dataset
from nugap import compare_network

OUT = os.path.join(os.path.dirname(__file__), "clock_output")
os.makedirs(OUT, exist_ok=True)


def to_long(data, condition, t):
    rows = []
    for gene, reps in data.items():
        for r in range(reps.shape[0]):
            for k, tk in enumerate(t):
                rows.append((gene, condition, r + 1, float(tk), float(reps[r, k])))
    return pd.DataFrame(rows, columns=["gene", "condition", "replicate",
                                       "time_h", "log2_expr"])


def main():
    wt, ko, t, meta, genes = make_dataset(n_reps=4, dt_h=2.0, span_h=48.0)

    # save data + metadata
    to_long(wt, "WT", t).to_csv(f"{OUT}/clock_rnaseq_WT.csv", index=False)
    to_long(ko, "KO", t).to_csv(f"{OUT}/clock_rnaseq_KO.csv", index=False)
    meta_rows = [(name, klass, p.get("acrophase", ""), p.get("driver", ""))
                 for (name, klass, p) in genes]
    pd.DataFrame(meta_rows, columns=["gene", "class", "acrophase_h", "driver"]
                 ).to_csv(f"{OUT}/clock_gene_metadata.csv", index=False)

    # plot a few representative genes, WT vs KO (replicate means)
    examples = ["Arntl", "Per2", "Nr1d1", "Dbp", "Clock",  # clock
                "Pck1", "Cyp7a1",                          # outputs
                "Hspa1b", "Fkbp5",                          # external (preserved)
                "Actb", "Gapdh"]                            # background
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    for ax, g in zip(axes.ravel(), examples):
        ax.plot(t, wt[g].mean(0), "-o", ms=3, label="WT", color="#1b7837")
        ax.plot(t, ko[g].mean(0), "-s", ms=3, label="CLOCK-KO", color="#c2185b")
        ax.fill_between(t, wt[g].mean(0) - wt[g].std(0), wt[g].mean(0) + wt[g].std(0),
                        color="#1b7837", alpha=0.15)
        ax.fill_between(t, ko[g].mean(0) - ko[g].std(0), ko[g].mean(0) + ko[g].std(0),
                        color="#c2185b", alpha=0.15)
        ax.set_title(f"{g}  [{meta[g]}]", fontsize=9)
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (h)")
    fig.suptitle("Synthetic circadian RNA-seq: WT vs CLOCK knockout", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/clock_tracks.png", dpi=130)

    # run the network comparison
    print("Running pairwise nu-gap network comparison (50 genes)...")
    df = compare_network(wt, ko, t, order=1, min_r2=0.5, gate="either", n=256)
    df["src_class"] = df["source"].map(meta)
    df["tgt_class"] = df["target"].map(meta)
    df.to_csv(f"{OUT}/clock_edges_results.csv", index=False)

    # overview figure: volcano + hub bar chart
    from nugap.viz import volcano, hub_barplot, hub_network
    vfig = plt.figure(figsize=(13, 5))
    volcano(df, q_thresh=0.1, label_top=0, ax=vfig.add_subplot(1, 2, 1))
    hub_barplot(df, top=18, ax=vfig.add_subplot(1, 2, 2))
    vfig.tight_layout()
    vfig.savefig(f"{OUT}/clock_viz.png", dpi=130, bbox_inches="tight")

    # separate network figure: most-rewired hubs, nodes coloured by gene class
    nfig, nax = plt.subplots(figsize=(9, 9))
    hub_network(df, top_hubs=15, q_thresh=0.1, node_groups=meta,
                layout="spring", ax=nax,
                title="CLOCK-KO: most-rewired hubs")
    nfig.savefig(f"{OUT}/clock_hub_network.png", dpi=130, bbox_inches="tight")

    flagged = df[df["q_global"] < 0.1]
    involves_clock = (df["src_class"].isin(["clock", "output"]) |
                      df["tgt_class"].isin(["clock", "output"]))
    ext_ext = (df["src_class"] == "external") & (df["tgt_class"] == "external")
    f_ext_ext = (flagged["src_class"] == "external") & (flagged["tgt_class"] == "external")

    print(f"\nEdges passing the fit-quality gate: {len(df)}")
    print(f"Edges flagged as changed (FDR q<0.1): {len(flagged)}")
    print(f"  ... involving clock/output genes: "
          f"{int((flagged['src_class'].isin(['clock','output']) | flagged['tgt_class'].isin(['clock','output'])).sum())}")
    print(f"SENSITIVITY (clock-network edges that collapse in KO): "
          f"{int(flagged[involves_clock.loc[flagged.index]].shape[0])} flagged")
    print(f"SPECIFICITY control (external-external, rhythmic in BOTH conditions): "
          f"{int(f_ext_ext.sum())} flagged of {int(ext_ext.sum())} tested  "
          f"(should be near the {0.1:.0%} FDR level)")
    print(f"\nFiles written to {OUT}/")


if __name__ == "__main__":
    main()
