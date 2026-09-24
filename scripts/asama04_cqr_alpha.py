#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aşama 04 — CQR karşılaştırması ve güven düzeyine duyarlılık
===========================================================
İki açık soruyu kapatır:
  1. Conformalized Quantile Regression (CQR, Romano ve ark. 2019) uyarlanabilir
     aralıkların standart yöntemidir. Kütüphane değişince o da bozuluyor mu?
  2. Bulgular yalnızca %90 düzeyinde mi geçerli? %80 ve %95'te ne oluyor?

Yöntemler (her alpha için, aynı hedef değerlendirme kümesinde):
  C0   split conformal (mutlak artık), kaynakta kalibre
  C0n  normalize conformal, kaynakta kalibre
  C4   CQR, kaynakta kalibre (iki kantil modeli + conformal düzeltme)
  C4t  CQR, k hedef etiketiyle yeniden kalibre
  C2   mutlak artık, k hedef etiketiyle yeniden kalibre
  C3n  afin düzeltme + normalize conformal, k hedef etiketi

Not: CQR kantil kaybı gerektirdiği için yalnızca HGB ile çalışır; bu betik
bu yüzden tek model ailesi (hgb) kullanır. PLSR karşılaştırmaları asama02
ve asama03'te zaten var.

Önceden belirtilen beklentiler (betik çalıştırılmadan yazıldı):
  F1 C4'ün sıfır atış kapsaması üç alpha düzeyinde de nominalin altında
     (bootstrap GA üst sınırı nominalin altında)
  F2 C4 kapsaması C0'dan yüksek (uyarlanabilirlik bir miktar yardımcı olur)
  F3 C4t, C2 ve C3n k=25'te nominale ulaşır (GA nominali içerir ya da üstünde)
  F4 Kapsama açığı alpha küçüldükçe büyür (%95 hedefi %80'den zor)
  F5 MIR'de C3n genişliği <= C4t genişliği (k=25)

Kullanım:
  python scripts\\asama04_cqr_alpha.py --arm mir --track clean
  python scripts\\asama04_cqr_alpha.py --arm visnir --track naive
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

SEED = 42
PROPS = ["clay.tot", "oc", "n.tot", "ph.h2o"]
LOGPROPS = {"oc", "n.tot"}
DEFAULT_DATA = r"C:\Users\nihal\Documents\cross-library-transfer\data\processed"


def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def to_num(sr):
    x = sr.astype(str).str.strip().str.replace("<", "", regex=False)
    x = x.str.replace(">", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def conf_q(scores, alpha):
    n = len(scores)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    if r > n:
        return np.inf
    return float(np.sort(scores)[r - 1])


def fit_sigma(P_di, absres_di, n_feat=20):
    d = int(min(n_feat, P_di.shape[1]))
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                      random_state=SEED).fit(P_di[:, :d], absres_di)
    floor = max(float(np.percentile(absres_di, 5)), 1e-6)
    return lambda P: np.maximum(m.predict(P[:, :d]), floor)


def affine_loo(pred_k, y_k, pred_eval):
    A = np.c_[pred_k, np.ones_like(pred_k)]
    coef, *_ = np.linalg.lstsq(A, y_k, rcond=None)
    res = y_k - A @ coef
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    h = np.clip(np.diag(H), 0, 0.999)
    return coef[0] * pred_eval + coef[1], res / (1 - h)


def boot_median_ci(x, rng, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    b = [np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--arm", default="mir", choices=["mir", "visnir"])
    ap.add_argument("--track", default="clean", choices=["clean", "naive"])
    ap.add_argument("--libcol", default="dataset.code_ascii_txt")
    ap.add_argument("--props", default=",".join(PROPS))
    ap.add_argument("--alphas", default="0.20,0.10,0.05")
    ap.add_argument("--ks", default="25,50")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--min-eval", type=int, default=30)
    ap.add_argument("--n-pc", type=int, default=120)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 230)
    t0 = time.time()
    alphas = [float(x) for x in a.alphas.split(",")]
    ks_all = [int(k) for k in a.ks.split(",")]

    X = np.load(os.path.join(a.data_dir, f"spectra_{a.arm}.npy"))
    bad = ~np.isfinite(X).all(axis=0)
    if bad.any():
        print(f"  {int(bad.sum())} bant NaN iceriyor, cikariliyor")
        X = X[:, ~bad]
    meta = pd.read_csv(os.path.join(a.data_dir, f"meta_{a.arm}.csv"), low_memory=False)
    lib_all = meta[a.libcol].astype(str).to_numpy()
    rng = np.random.default_rng(a.seed)
    print(f"{a.arm} / {a.track}: {X.shape[0]} spektrum | alpha {alphas} | k {ks_all} | model hgb")

    rows = []
    for prop in a.props.split(","):
        col = f"{prop}__{a.track}"
        if col not in meta.columns:
            continue
        y_all = to_num(meta[col]).to_numpy(float)
        if prop in LOGPROPS:
            y_all = np.log1p(np.clip(y_all, 0, None))
        ok = np.isfinite(y_all) & np.isfinite(X).all(axis=1)
        vc = pd.Series(lib_all[ok]).value_counts()
        libs = vc[vc >= a.min_n].index.tolist()
        if len(libs) < 3:
            continue
        print(f"\n=== {prop}  ({int(ok.sum())} ornek, {len(libs)} kutuphane)")

        for tgt in libs:
            m = ok & (lib_all == tgt)
            s = ok & (lib_all != tgt)
            Xt, yt = X[m], y_all[m]
            nt = len(yt)
            ks = [k for k in ks_all if nt - k >= a.min_eval]
            if not ks:
                continue
            kmax = max(ks)
            sd_t = yt.std(ddof=1)

            si = rng.permutation(np.flatnonzero(s))
            n10, n20 = int(0.10 * len(si)), int(0.20 * len(si))
            ca = si[n10:n10 + n20]
            di = si[n10 + n20:2 * n10 + n20]
            tr = si[2 * n10 + n20:]

            # ortak izdüşüm ve orta (nokta) model
            Ztr = snv(X[tr])
            npc = int(min(a.n_pc, Ztr.shape[0] - 1, Ztr.shape[1]))
            pca = PCA(n_components=npc, random_state=a.seed).fit(Ztr)
            Ptr = pca.transform(Ztr)
            P_ca, P_di, P_t = (pca.transform(snv(X[ca])), pca.transform(snv(X[di])),
                               pca.transform(snv(Xt)))
            est = HistGradientBoostingRegressor(random_state=a.seed).fit(Ptr, y_all[tr])
            yh_ca, yh_di, yh_t = est.predict(P_ca), est.predict(P_di), est.predict(P_t)
            sig = fit_sigma(P_di, np.abs(y_all[di] - yh_di))
            s_ca, s_t = sig(P_ca), sig(P_t)
            sc_abs = np.abs(y_all[ca] - yh_ca)

            for alpha in alphas:
                # CQR: iki kantil modeli (alpha/2 ve 1-alpha/2)
                qlo_m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=alpha / 2, random_state=a.seed).fit(Ptr, y_all[tr])
                qhi_m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=1 - alpha / 2, random_state=a.seed).fit(Ptr, y_all[tr])
                lo_ca, hi_ca = qlo_m.predict(P_ca), qhi_m.predict(P_ca)
                lo_t, hi_t = qlo_m.predict(P_t), qhi_m.predict(P_t)
                E_ca = np.maximum(lo_ca - y_all[ca], y_all[ca] - hi_ca)
                E_t = np.maximum(lo_t - yt, yt - hi_t)

                q0 = conf_q(sc_abs, alpha)
                q0n = conf_q(sc_abs / s_ca, alpha)
                q4 = conf_q(E_ca, alpha)

                acc = {}

                def add(key, hit, width):
                    d = acc.setdefault(key, dict(cov=[], w=[]))
                    d["cov"].append(float(np.mean(hit)))
                    d["w"].append(float(np.median(width) / sd_t))

                for _ in range(a.reps):
                    perm = rng.permutation(nt)
                    pool, ev = perm[:kmax], perm[kmax:]
                    y_ev, p_ev = yt[ev], yh_t[ev]
                    ae = np.abs(y_ev - p_ev)

                    add("C0", ae <= q0, np.full(len(ev), 2 * q0))
                    add("C0n", ae <= q0n * s_t[ev], 2 * q0n * s_t[ev])
                    lo_e, hi_e = lo_t[ev] - q4, hi_t[ev] + q4
                    add("C4", (y_ev >= lo_e) & (y_ev <= hi_e), hi_e - lo_e)

                    for k in ks:
                        ck = pool[:k]
                        q2 = conf_q(np.abs(yt[ck] - yh_t[ck]), alpha)
                        add(f"C2_k{k}", ae <= q2, np.full(len(ev), 2 * q2))

                        q4t = conf_q(E_t[ck], alpha)
                        lo_k, hi_k = lo_t[ev] - q4t, hi_t[ev] + q4t
                        add(f"C4t_k{k}", (y_ev >= lo_k) & (y_ev <= hi_k), hi_k - lo_k)

                        p3, loo = affine_loo(yh_t[ck], yt[ck], p_ev)
                        q3n = conf_q(np.abs(loo) / s_t[ck], alpha)
                        add(f"C3n_k{k}", np.abs(y_ev - p3) <= q3n * s_t[ev],
                            2 * q3n * s_t[ev])

                for meth, d in acc.items():
                    rows.append(dict(prop=prop, target=tgt, n_target=nt, alpha=alpha,
                                     method=meth,
                                     coverage=float(np.mean(d["cov"])),
                                     width_sd=float(np.median(d["w"]))))
                print(f"  {tgt:>18s} a={alpha:.2f} | C0 {np.mean(acc['C0']['cov']):.3f} "
                      f"C0n {np.mean(acc['C0n']['cov']):.3f} C4 {np.mean(acc['C4']['cov']):.3f} | "
                      f"k=25: C2 {np.mean(acc['C2_k25']['cov']):.3f} "
                      f"C4t {np.mean(acc['C4t_k25']['cov']):.3f} "
                      f"C3n {np.mean(acc['C3n_k25']['cov']):.3f}  [{time.time() - t0:.0f} sn]")

    if not rows:
        raise SystemExit("hic hucre kosmadi")
    res = pd.DataFrame(rows)
    tag = f"{a.arm}_{a.track}"
    res.to_csv(os.path.join(a.out, f"asama04_hucreler_{tag}.csv"), index=False)

    brng = np.random.default_rng(a.seed)
    out = []
    for (alpha, meth), d in res.groupby(["alpha", "method"]):
        lo, hi = boot_median_ci(d["coverage"], brng)
        out.append(dict(alpha=alpha, method=meth, hucre=len(d),
                        nominal=1 - alpha, kapsama=d["coverage"].median(),
                        ga_alt=lo, ga_ust=hi, acik=(1 - alpha) - d["coverage"].median(),
                        genislik=d["width_sd"].median()))
    order = {"C0": 0, "C0n": 1, "C4": 2, "C2_k25": 3, "C4t_k25": 4, "C3n_k25": 5,
             "C2_k50": 6, "C4t_k50": 7, "C3n_k50": 8}
    summ = (pd.DataFrame(out).assign(_o=lambda x: x["method"].map(order))
              .sort_values(["alpha", "_o"]).drop(columns="_o").round(3))
    summ.to_csv(os.path.join(a.out, f"asama04_ozet_{tag}.csv"), index=False)

    print("\n\n" + "=" * 92)
    print(f"OZET — CQR ve guven duzeyi duyarliligi ({tag})")
    print("=" * 92)
    print(summ.to_string(index=False))

    S = summ.set_index(["alpha", "method"])
    print("\nONCEDEN BELIRTILEN BEKLENTILER")
    f1 = all(S.loc[(al, "C4"), "ga_ust"] < 1 - al for al in alphas)
    print(f"  F1 C4 sifir atis kapsamasi tum alpha'larda nominalin altinda? "
          f"{'EVET' if f1 else 'HAYIR'}")
    for al in alphas:
        print(f"     alpha={al:.2f}: C4 {S.loc[(al, 'C4'), 'kapsama']:.3f} "
              f"[{S.loc[(al, 'C4'), 'ga_alt']:.3f}, {S.loc[(al, 'C4'), 'ga_ust']:.3f}]  "
              f"nominal {1 - al:.2f}")
    f2 = all(S.loc[(al, "C4"), "kapsama"] > S.loc[(al, "C0"), "kapsama"] for al in alphas)
    print(f"  F2 C4 > C0 tum alpha'larda? {'EVET' if f2 else 'HAYIR'}")
    for al in alphas:
        print(f"     alpha={al:.2f}: C0 {S.loc[(al, 'C0'), 'kapsama']:.3f}  "
              f"C0n {S.loc[(al, 'C0n'), 'kapsama']:.3f}  C4 {S.loc[(al, 'C4'), 'kapsama']:.3f}")
    f3 = all(S.loc[(al, m), "ga_ust"] >= 1 - al for al in alphas
             for m in ("C2_k25", "C4t_k25", "C3n_k25"))
    print(f"  F3 k=25 yontemleri nominale ulasiyor? {'EVET' if f3 else 'HAYIR'}")
    acik = {al: (1 - al) - S.loc[(al, "C0"), "kapsama"] for al in alphas}
    f4 = acik[min(alphas)] > acik[max(alphas)]
    print(f"  F4 kapsama acigi alpha kuculunce buyuyor? {'EVET' if f4 else 'HAYIR'}")
    print("     " + "   ".join(f"a={al:.2f}: {acik[al]:+.3f}" for al in sorted(alphas, reverse=True)))
    f5 = all(S.loc[(al, "C3n_k25"), "genislik"] <= S.loc[(al, "C4t_k25"), "genislik"]
             for al in alphas)
    print(f"  F5 C3n genisligi <= C4t genisligi (k=25)? {'EVET' if f5 else 'HAYIR'}")
    for al in alphas:
        print(f"     alpha={al:.2f}: C3n {S.loc[(al, 'C3n_k25'), 'genislik']:.3f}  "
              f"C4t {S.loc[(al, 'C4t_k25'), 'genislik']:.3f}  "
              f"C2 {S.loc[(al, 'C2_k25'), 'genislik']:.3f}")
    print(f"\ncikti: {os.path.abspath(a.out)}   ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
