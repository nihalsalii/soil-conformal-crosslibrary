"""
asama08 — Is the coverage failure explained by the loss of point-prediction accuracy?

Fits no models. Reads the per-cell tables written by asama07 (the matched
protocol), which record rmse/std in the source and in the target for every
cell, and relates that accuracy change to the coverage of the source-calibrated
interval C0.

Output:
  figures/Figure8_accuracy_vs_coverage.png / .pdf
  out/asama08_nokta_vs_kapsama.csv

Usage:
  python scripts\\asama08_nokta_vs_kapsama.py
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

C_BLUE, C_ORANGE, C_GREEN, C_YELLOW, C_PINK = ("#0072B2", "#D55E00", "#009E73",
                                               "#E69F00", "#CC79A7")
INK, INK2, GRID = "#1a1a1a", "#595959", "#d9d9d9"
ARMS = [("mir_clean", "MIR (method-matched)"), ("visnir_naive", "Vis-NIR (method-mixed)")]
NOM = 0.90


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def save(fig, figdir, name, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def cells(out, tag):
    p = os.path.join(out, f"asama07_hucreler_{tag}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d["method"] == "C0"]
    g = (d.groupby(["prop", "model", "target"], as_index=False)
           [["rmse_std_source", "rmse_std_target", "coverage"]].mean())
    g["ratio"] = g["rmse_std_target"] / g["rmse_std_source"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)

    rows, panels = [], []
    for tag, label in ARMS:
        g = cells(a.out, tag)
        if g is None:
            print(f"  {tag}: asama07 table missing, skipped")
            continue
        rho_r, p_r = spearmanr(g["ratio"], g["coverage"])
        rho_t, p_t = spearmanr(g["rmse_std_target"], g["coverage"])
        panels.append((label, g, rho_r, p_r))
        rows.append(dict(arm=label, cells=len(g),
                         rho_ratio=round(float(rho_r), 3), p_ratio=round(float(p_r), 4),
                         rho_target_rmse=round(float(rho_t), 3), p_target_rmse=round(float(p_t), 4),
                         median_ratio=round(float(g["ratio"].median()), 3),
                         cells_accuracy_kept_coverage_below_080=int(
                             ((g["ratio"] <= 1.25) & (g["coverage"] < 0.80)).sum())))
        print(f"{label}: {len(g)} cells | rho(ratio, coverage) = {rho_r:+.3f} (p = {p_r:.3f})")

    if not panels:
        print("nothing to plot")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(9.6, 4.0), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, g, rho, p) in zip(axes, panels):
        ax.axhline(NOM, ls="--", lw=1.1, color=INK2, zorder=1)
        ax.axvline(1.0, ls=":", lw=1.0, color=GRID, zorder=1)
        ax.scatter(g["ratio"], g["coverage"], s=38, color=C_BLUE,
                   edgecolor="white", linewidth=0.6, zorder=3)

        # highlight the cell whose accuracy is best preserved
        best = g.nsmallest(1, "ratio").iloc[0]
        ax.scatter([best["ratio"]], [best["coverage"]], s=90, facecolor="none",
                   edgecolor=C_ORANGE, linewidth=1.8, zorder=4)

        pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.03, 0.06, f"ρ = {rho:+.2f}  ({pstr})", transform=ax.transAxes,
                fontsize=9, color=INK2)
        style(ax, xlabel="target rmse / std  ÷  source rmse / std",
              ylabel="empirical coverage of C0" if ax is axes[0] else "",
              title=label)
        ax.set_ylim(-0.02, 1.02)

    handles = [
        Line2D([], [], marker="o", ls="", color=C_BLUE, markeredgecolor="white",
               markersize=7, label="one library × property × model cell"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=C_ORANGE, markeredgewidth=1.8, markersize=9,
               label="cell with the best-preserved accuracy"),
        Line2D([], [], ls="--", color=INK2, label="nominal 0.90"),
        Line2D([], [], ls=":", color=GRID, label="accuracy unchanged from source"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Figure 8. Coverage failure is not a restatement of accuracy loss",
                 x=0.012, ha="left", fontsize=11, color=INK)
    save(fig, a.figdir, "Figure8_accuracy_vs_coverage", rect=(0, 0.04, 1, 0.94))

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "asama08_nokta_vs_kapsama.csv"), index=False)
    print()
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
