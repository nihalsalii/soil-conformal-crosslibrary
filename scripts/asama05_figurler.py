#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aşama 05 — Güvenilirlik tablosu ve makale şekilleri (şekil metinleri İNGİLİZCE)
==============================================================================
Yeni model eğitilmez; asama02, asama03 ve asama04'ün CSV çıktılarını okur.

Şekiller (figures\ klasörüne, 300 dpi PNG + PDF), tümü İngilizce:
  Figure1  per-cell coverage by method (alpha = 0.10)
  Figure2  reliability: share of cells with coverage >= nominal - 0.05
  Figure3  label budget: coverage and interval width vs k
  Figure4  width-coverage trade-off
  Figure5  domain separability (AUC) vs weighted-conformal breakdown and coverage loss
  Figure6  sensitivity to the nominal level (80 / 90 / 95 %)
  Figure7  model families under one matched protocol (asama06 + asama07)

Tablolar (out\ klasörüne):
  asama05_guvenilirlik.csv, asama05_etiket_butcesi.csv

Kullanım:
  python scripts\\asama05_figurler.py
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# Okabe-Ito alt kümesi; CVD ayrım doğrulamasından geçti (sabit sıra, döngü yok)
C_BLUE, C_ORANGE, C_GREEN, C_YELLOW, C_PINK = ("#0072B2", "#D55E00", "#009E73",
                                               "#E69F00", "#CC79A7")
INK, INK2, GRID = "#1a1a1a", "#595959", "#d9d9d9"
ARMS = [("mir_clean", "MIR (method-matched)"), ("visnir_naive", "Vis-NIR (method-mixed)")]
NOM = 0.90

METHOD_LABEL = {
    "C0": "C0  source conformal",
    "C0n": "C0n  normalized",
    "C1": "C1  weighted",
    "C1c": "C1c  weighted (clipped)",
    "C4": "C4  CQR",
    "C2": "C2  k = 25 target labels",
    "C3n": "C3n  affine + normalized, k = 25",
}
METHOD_SHORT = {"C0": "C0\nsource", "C0n": "C0n\nnormalized", "C1": "C1\nweighted",
                "C1c": "C1c\nweighted\n(clipped)", "C4": "C4\nCQR",
                "C2": "C2\nk = 25", "C3n": "C3n\nk = 25"}
METHOD_ORDER = ["C0", "C0n", "C1", "C1c", "C4", "C2", "C3n"]
METHOD_COLOR = {"C0": INK, "C0n": C_BLUE, "C1": C_ORANGE, "C1c": C_ORANGE,
                "C4": C_GREEN, "C2": C_PINK, "C3n": C_YELLOW}


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


def load(out, pattern, tag):
    p = os.path.join(out, pattern.format(tag=tag))
    return pd.read_csv(p) if os.path.exists(p) else None


def method_table(out, tag):
    frames = []
    a2 = load(out, "asama02_hucreler_{tag}.csv", tag)
    if a2 is not None:
        sel = a2[((a2["method"].isin(["C0", "C0n", "C1"])) & (a2["k"] == 0)) |
                 ((a2["method"].isin(["C2", "C3n"])) & (a2["k"] == 25))]
        frames.append(sel[["prop", "model", "target", "method", "coverage",
                           "width_sd", "frac_inf"]])
    a3 = load(out, "asama03_hucreler_{tag}.csv", tag)
    if a3 is not None:
        c1c = (a3[a3["method"] == "C1c"]
               .groupby(["prop", "model", "target", "method"], as_index=False)
               [["coverage", "width_sd", "frac_inf"]].mean())
        frames.append(c1c)
    a4 = load(out, "asama04_hucreler_{tag}.csv", tag)
    if a4 is not None:
        c4 = a4[(a4["alpha"] == 0.10) & (a4["method"] == "C4")].copy()
        c4["model"], c4["frac_inf"] = "hgb", 0.0
        frames.append(c4[["prop", "model", "target", "method", "coverage",
                          "width_sd", "frac_inf"]])
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    return d[d["method"].isin(METHOD_ORDER)]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    print("figures:")

    tables = {tag: method_table(a.out, tag) for tag, _ in ARMS}
    tables = {k: v for k, v in tables.items() if v is not None and len(v)}
    if not tables:
        raise SystemExit(f"'{a.out}' icinde asama02/03/04 CSV'leri bulunamadi")
    arms_present = [t for t in ARMS if t[0] in tables]

    # ---------- Figure 1 ----------
    fig, axes = plt.subplots(1, len(arms_present), figsize=(5.0 * len(arms_present), 4.4),
                             sharex=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, (tag, label) in zip(axes, arms_present):
        d = tables[tag]
        meths = [m for m in METHOD_ORDER if m in set(d["method"])]
        for i, m in enumerate(meths):
            v = d.loc[d["method"] == m, "coverage"].to_numpy()
            ax.scatter(v, i + rng.uniform(-0.16, 0.16, len(v)), s=15,
                       color=METHOD_COLOR[m], alpha=0.75, linewidths=0, zorder=3)
            ax.plot([np.median(v)], [i], marker="|", ms=18, mew=2.2, color=INK, zorder=4)
        ax.axvline(NOM, color=INK2, lw=1.2, ls="--", zorder=2)
        ax.set_yticks(range(len(meths)))
        ax.set_yticklabels([METHOD_LABEL[m] for m in meths], fontsize=8, color=INK)
        ax.set_ylim(len(meths) - 0.4, -0.9)
        ax.set_xlim(0, 1.02)
        ax.text(NOM - 0.02, -0.75, "nominal 0.90", fontsize=8, color=INK2, ha="right")
        style(ax, xlabel="empirical coverage in the held-out library", title=label)
    fig.suptitle("Figure 1. Coverage per cell (property × model × target library); "
                 "vertical bar is the median",
                 fontsize=10, color=INK, x=0.005, ha="left")
    save(fig, a.figdir, "Figure1_coverage_by_method", rect=(0, 0, 1, 0.95))

    # ---------- reliability table ----------
    rel_rows = []
    for tag, label in ARMS:
        if tag not in tables:
            continue
        d = tables[tag]
        for m in METHOD_ORDER:
            v = d.loc[d["method"] == m, "coverage"]
            if not len(v):
                continue
            w = d.loc[d["method"] == m, "width_sd"].replace(np.inf, np.nan)
            rel_rows.append(dict(
                arm=label, method=m, description=METHOD_LABEL[m], cells=len(v),
                median=round(v.median(), 3), q25=round(v.quantile(.25), 3),
                q75=round(v.quantile(.75), 3), worst=round(v.min(), 3),
                reliable_share=round(float((v >= NOM - 0.05).mean()), 3),
                below_075_share=round(float((v < 0.75).mean()), 3),
                median_width=round(float(w.median()), 3),
                infinite_share=round(float(d.loc[d["method"] == m, "frac_inf"].mean()), 3)))
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(os.path.join(a.out, "asama05_guvenilirlik.csv"), index=False)

    # ---------- Figure 2 ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    meths = [m for m in METHOD_ORDER if m in set(rel["method"])]
    x = np.arange(len(meths))
    width = 0.38
    for j, (tag, label) in enumerate(arms_present):
        vals = []
        for m in meths:
            r = rel[(rel["arm"] == label) & (rel["method"] == m)]["reliable_share"]
            vals.append(float(r.iloc[0]) if len(r) else np.nan)
        ax.bar(x + (j - 0.5) * width, vals, width * 0.92,
               color=(C_BLUE if j == 0 else C_ORANGE), label=label, zorder=3)
        for xi, v in zip(x + (j - 0.5) * width, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_SHORT[m] for m in meths], fontsize=8, color=INK)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    style(ax, ylabel="share of cells with coverage ≥ 0.85",
          title="Figure 2. Reliability: in how many libraries the method actually works")
    save(fig, a.figdir, "Figure2_reliability")

    # ---------- Figure 3 ----------
    budget_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    meth_lbl = {"C2": "C2  threshold only", "C3": "C3  affine", "C3n": "C3n  affine + normalized"}
    for tag, label in ARMS:
        a2 = load(a.out, "asama02_hucreler_{tag}.csv", tag)
        if a2 is None:
            continue
        ls = "-" if tag == "mir_clean" else "--"
        for m, c in (("C2", C_PINK), ("C3", C_BLUE), ("C3n", C_YELLOW)):
            d = a2[(a2["method"] == m) & (a2["k"] > 0)]
            if not len(d):
                continue
            g = d.groupby("k")[["coverage", "width_sd"]].median().reset_index()
            for _, r in g.iterrows():
                budget_rows.append(dict(arm=label, method=m, k=int(r["k"]),
                                        coverage=round(r["coverage"], 3),
                                        width=round(r["width_sd"], 3)))
            axes[0].plot(g["k"], g["coverage"], ls, color=c, marker="o", ms=4.5, lw=2)
            axes[1].plot(g["k"], g["width_sd"], ls, color=c, marker="o", ms=4.5, lw=2)
    # sonlu örneklem beklentisi: ceil((k+1)(1-alpha)) / (k+1)
    kk = np.array([10, 25, 50, 100])
    theo = np.ceil((kk + 1) * NOM) / (kk + 1)
    axes[0].plot(kk, theo, color=INK2, lw=1.2, ls=":", marker="", zorder=1)
    axes[0].axhline(NOM, color=GRID, lw=1.0, zorder=0)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(kk)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(axes[0], xlabel="number of target labels k", ylabel="coverage (median)",
          title="Figure 3a. Coverage")
    style(axes[1], xlabel="number of target labels k",
          ylabel="interval width / target SD", title="Figure 3b. Width")
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=4.5, label=meth_lbl[m])
               for m, c in (("C2", C_PINK), ("C3", C_BLUE), ("C3n", C_YELLOW))]
    handles += [Line2D([], [], color=INK2, lw=2, ls="-", label=ARMS[0][1]),
                Line2D([], [], color=INK2, lw=2, ls="--", label=ARMS[1][1]),
                Line2D([], [], color=INK2, lw=1.2, ls=":", label="finite-sample expectation")]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    save(fig, a.figdir, "Figure3_label_budget", rect=(0, 0.12, 1, 1))
    pd.DataFrame(budget_rows).to_csv(os.path.join(a.out, "asama05_etiket_butcesi.csv"),
                                     index=False)

    # ---------- Figure 4 ----------
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for tag, label in arms_present:
        d = tables[tag]
        mk = "o" if tag == "mir_clean" else "s"
        for m in [x for x in METHOD_ORDER if x in set(d["method"])]:
            dd = d[d["method"] == m]
            w = dd["width_sd"].replace(np.inf, np.nan).median()
            c = dd["coverage"].median()
            if not np.isfinite(w):
                continue
            ax.scatter([w], [c], s=70, marker=mk, color=METHOD_COLOR[m],
                       edgecolor="white", linewidth=1.2, zorder=3)
            ax.annotate(m, (w, c), xytext=(7, 4), textcoords="offset points",
                        fontsize=8, color=INK)
    ax.axhline(NOM, color=INK2, lw=1.2, ls="--")
    ax.scatter([], [], marker="o", color=INK2, label=ARMS[0][1])
    ax.scatter([], [], marker="s", color=INK2, label=ARMS[1][1])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, xlabel="interval width / target SD (median)", ylabel="coverage (median)",
          title="Figure 4. Valid intervals are paid for in width")
    save(fig, a.figdir, "Figure4_width_coverage")

    # ---------- Figure 5 ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    txt = []
    for tag, label in ARMS:
        a3 = load(a.out, "asama03_hucreler_{tag}.csv", tag)
        if a3 is None:
            continue
        cell = (a3.groupby(["prop", "model", "target", "method"], as_index=False)
                  [["coverage", "frac_inf", "auc"]].mean())
        c1 = cell[cell["method"] == "C1"]
        c0 = cell[cell["method"] == "C0"]
        col = C_BLUE if tag == "mir_clean" else C_ORANGE
        mk = "o" if tag == "mir_clean" else "s"
        axes[0].scatter(c1["auc"], c1["frac_inf"], s=26, marker=mk, color=col,
                        alpha=0.8, linewidths=0, label=label)
        loss = NOM - c0["coverage"]
        axes[1].scatter(c0["auc"], loss, s=26, marker=mk, color=col, alpha=0.8,
                        linewidths=0, label=label)
        r1 = spearmanr(c1["auc"], c1["frac_inf"], nan_policy="omit")
        r2 = spearmanr(c0["auc"], loss, nan_policy="omit")
        txt.append((label, r1.correlation, r1.pvalue, r2.correlation, r2.pvalue))
    style(axes[0], xlabel="domain-classifier AUC",
          title="Figure 5a. Separability and the breakdown of weighted conformal")
    style(axes[1], xlabel="domain-classifier AUC",
          title="Figure 5b. Does separability predict the coverage loss?")
    axes[0].set_ylabel("share of infinite intervals (C1)", color=INK, fontsize=9)
    axes[1].set_ylabel("coverage loss (0.90 − C0)", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for i, (label, r1, p1, r2, p2) in enumerate(txt):
        axes[0].text(0.03, 0.60 - 0.09 * i, f"{label}: ρ = {r1:+.2f} (p = {p1:.3f})",
                     transform=axes[0].transAxes, fontsize=7.5, color=INK2)
        axes[1].text(0.03, 0.93 - 0.09 * i, f"{label}: ρ = {r2:+.2f} (p = {p2:.3f})",
                     transform=axes[1].transAxes, fontsize=7.5, color=INK2)
    save(fig, a.figdir, "Figure5_separability")

    # ---------- Figure 6 ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    sel = {"C0": (C_BLUE, "C0  source conformal"),
           "C0n": (C_GREEN, "C0n  normalized"),
           "C4": (C_ORANGE, "C4  CQR"),
           "C2_k25": (C_PINK, "C2  k = 25 target labels")}
    for ax, (tag, label) in zip(axes, ARMS):
        a4 = load(a.out, "asama04_hucreler_{tag}.csv", tag)
        if a4 is None:
            continue
        for m, (c, _) in sel.items():
            d = a4[a4["method"] == m]
            if not len(d):
                continue
            g = d.groupby("alpha")["coverage"].median().reset_index()
            g["nominal"] = 1 - g["alpha"]
            g = g.sort_values("nominal")
            ax.plot(g["nominal"], g["coverage"] - g["nominal"], color=c, marker="o",
                    ms=5, lw=2)
        ax.axhline(0, color=INK2, lw=1.2, ls="--")
        ax.set_xticks([0.80, 0.90, 0.95])
        style(ax, xlabel="nominal level", title=label)
    axes[0].set_ylabel("coverage − nominal", color=INK, fontsize=9)
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=5, label=lab)
               for _, (c, lab) in sel.items()]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 6. Narrow (80 %) intervals are the most misleading",
                 fontsize=10, color=INK, x=0.005, ha="left")
    save(fig, a.figdir, "Figure6_nominal_level", rect=(0, 0.06, 1, 0.95))

    # ---------- Figure 7: model families, matched protocol ----------
    fam_rows = []
    for tag, label in ARMS:
        a7 = load(a.out, "asama07_hucreler_{tag}.csv", tag)
        a6 = load(a.out, "asama06_hucreler_cnn_{tag}.csv", tag)
        for d in (a7, a6):
            if d is None:
                continue
            sel = d[((d["method"] == "C0") & (d["k"] == 0)) |
                    ((d["method"].isin(["C2", "C3n"])) & (d["k"] == 25))]
            for _, r in sel.iterrows():
                fam_rows.append(dict(arm=label, family=r["model"], method=r["method"],
                                     coverage=r["coverage"], width=r["width_sd"]))
    if fam_rows:
        fam = pd.DataFrame(fam_rows)
        fam_lbl = {"plsr": "PLSR\n(linear projection)", "hgb": "Gradient boosting\n(trees)",
                   "cnn": "1D-CNN\n(deep)"}
        order = [f for f in ("plsr", "hgb", "cnn") if f in set(fam["family"])]
        colors = {"plsr": C_BLUE, "hgb": C_PINK, "cnn": C_GREEN}
        arms_f = [t for t in ARMS if t[1] in set(fam["arm"])]
        fig, axes = plt.subplots(1, len(arms_f), figsize=(4.8 * len(arms_f), 4.0),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        jit = np.random.default_rng(1)
        for ax, (tag, label) in zip(axes, arms_f):
            d = fam[fam["arm"] == label]
            for i, f in enumerate(order):
                v = d[(d["family"] == f) & (d["method"] == "C0")]["coverage"].to_numpy()
                ax.scatter(i + jit.uniform(-0.14, 0.14, len(v)), v, s=18,
                           color=colors[f], alpha=0.75, linewidths=0, zorder=3)
                if len(v):
                    ax.plot([i], [np.median(v)], marker="_", ms=26, mew=2.4,
                            color=INK, zorder=4)
                w = d[(d["family"] == f) & (d["method"] == "C3n")]["coverage"]
                if len(w):
                    ax.scatter([i], [w.median()], marker="D", s=34, color=C_YELLOW,
                               edgecolor="white", linewidth=1.0, zorder=5)
            ax.axhline(NOM, color=INK2, lw=1.2, ls="--", zorder=2)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([fam_lbl[f] for f in order], fontsize=8, color=INK)
            ax.set_ylim(0, 1.03)
            style(ax, title=label)
        axes[0].set_ylabel("empirical coverage in the held-out library", color=INK, fontsize=9)
        handles = [Line2D([], [], color=INK, lw=0, marker="_", ms=14, mew=2.2,
                          label="median, source-calibrated (C0)"),
                   Line2D([], [], color=C_YELLOW, lw=0, marker="D", ms=6,
                          label="median with 25 target labels (C3n)")]
        fig.legend(handles=handles, frameon=False, fontsize=8, ncol=2,
                   loc="lower center", bbox_to_anchor=(0.5, -0.04))
        fig.suptitle("Figure 7. Same protocol, three model families: the direction is "
                     "universal, the magnitude is not",
                     fontsize=10, color=INK, x=0.005, ha="left")
        save(fig, a.figdir, "Figure7_model_families", rect=(0, 0.08, 1, 0.95))

        famsum = (fam[fam["method"] == "C0"].groupby(["arm", "family"])["coverage"]
                  .agg(median="median", worst="min",
                       reliable=lambda v: float((v >= NOM - 0.05).mean()),
                       cells="size").round(3).reset_index())
        famsum.to_csv(os.path.join(a.out, "asama05_model_aileleri.csv"), index=False)
        print("\nMODEL FAMILIES (matched protocol, C0)")
        print(famsum.to_string(index=False))

    # ---------- console ----------
    print("\nRELIABILITY TABLE (alpha = 0.10)")
    print(rel[["arm", "method", "cells", "median", "q25", "q75", "worst",
               "reliable_share", "median_width"]].to_string(index=False))
    print(f"\ntables:  {os.path.abspath(a.out)}")
    print(f"figures: {os.path.abspath(a.figdir)}")


if __name__ == "__main__":
    main()
