#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aşama 02 — TAM ANALİZ: kütüphaneler arası tahmin aralıklarının geçerliliği
==========================================================================
Pilot (asama01) sonrası eklenenler:
  * Kaynak eğitim kümesi alt örneklenmez (varsayılan --max-train 0 = hepsi)
  * C0w  genişlik eşitlenmiş kontrol: C0 aralığı, C1'in medyan genişliğine
         kadar açılır. C1'in kazancı ağırlıklardan mı, yalnızca aralığın
         genişlemesinden mi geliyor?
  * C0n  normalize (uyarlanabilir) conformal: kaynakta bir zorluk modeli
         sigma(x) eğitilir, aralık yhat +/- q*sigma(x). Hakemin "uyarlanabilir
         yöntem denediniz mi?" sorusunun cevabı.
  * C3n  afin düzeltme + normalize conformal (k hedef etiketi, LOO)
  * Hücreler üzerinden bootstrap güven aralıkları (medyan kapsama)
  * Kaynak aralığı içi / dışı kapsama tüm yöntemler için

Yöntemler:
  C0   split conformal, kaynakta kalibre, hedefe olduğu gibi
  C0w  C0, C1'in medyan yarı-genişliğiyle (genişlik kontrolü)
  C0n  normalize conformal, kaynakta kalibre
  C1   weighted conformal (alan sınıflandırıcısı, hedef etiketi kullanılmaz)
  C2   k hedef etiketiyle eşik yeniden kalibre
  C3   k hedef etiketiyle afin + LOO conformal
  C3n  k hedef etiketiyle afin + normalize LOO conformal

Önceden belirtilen beklentiler (pilot sonrası, tam analiz öncesi yazıldı):
  B1 C0 medyan kapsamasının %95 GA üst sınırı < 0.90
  B2 C1 < C2(k=25) (GA'lar ayrık)
  B3 C0w kapsaması C1'e yakın (fark < 0.03) -> C1 kazancı büyük ölçüde genişlikten
  B4 C3 genişliği < C2 genişliği (k >= 25)
  B5 C0n, C0'dan daha iyi kapsar ama nominale ulaşmaz (koşullu kayma sürer)

Kullanım (NEW ARTİCLE klasöründen, ossl ortamında):
  python scripts\\asama02_conformal_full.py --arm mir --track clean
  python scripts\\asama02_conformal_full.py --arm visnir --track naive
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

SEED = 42
PROPS = ["clay.tot", "oc", "n.tot", "ph.h2o"]
LOGPROPS = {"oc", "n.tot"}
DEFAULT_DATA = r"C:\Users\nihal\Documents\cross-library-transfer\data\processed"


# ---------------------------------------------------------------------------
def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def to_num(sr):
    x = sr.astype(str).str.strip().str.replace("<", "", regex=False)
    x = x.str.replace(">", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def fit_model(Xtr, ytr, model, n_pc=120, n_plsr=20):
    Z = snv(Xtr)
    npc = int(min(n_pc, Z.shape[0] - 1, Z.shape[1]))
    pca = PCA(n_components=npc, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    if model == "plsr":
        k = int(min(n_plsr, P.shape[1], max(2, P.shape[0] - 1)))
        est = PLSRegression(n_components=k).fit(P, ytr)
    else:
        est = HistGradientBoostingRegressor(random_state=SEED).fit(P, ytr)
    return pca, est


def predict(pca, est, X):
    P = pca.transform(snv(X))
    return np.asarray(est.predict(P)).ravel(), P


# ---------------------------------------------------------------------------
def conf_q(scores, alpha):
    """Sıralı skorların ceil((n+1)(1-alpha))'ıncı değeri; n yetmezse sonsuz."""
    n = len(scores)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    if r > n:
        return np.inf
    return float(np.sort(scores)[r - 1])


def weighted_q(scores, w_cal, w_test, alpha):
    order = np.argsort(scores)
    s, w = scores[order], w_cal[order]
    cw = np.cumsum(w)
    thr = (1 - alpha) * (cw[-1] + w_test)
    idx = np.searchsorted(cw, thr, side="left")
    return np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)


def density_ratio(P_cal, P_tgt, n_feat=20):
    d = int(min(n_feat, P_cal.shape[1]))
    Xd = np.vstack([P_cal[:, :d], P_tgt[:, :d]])
    yd = np.r_[np.zeros(len(P_cal)), np.ones(len(P_tgt))]
    sc = StandardScaler().fit(Xd)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xd), yd)

    def w(P):
        p = clf.predict_proba(sc.transform(P[:, :d]))[:, 1]
        p = np.clip(p, 0.01, 0.99)
        return (p / (1 - p)) * (len(P_cal) / len(P_tgt))
    return w


def fit_sigma(P_di, absres_di, n_feat=20):
    """Zorluk modeli: |artık| ~ f(ilk PC'ler). Alt sınır: |artık|'ın %5'lik değeri."""
    d = int(min(n_feat, P_di.shape[1]))
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                      random_state=SEED).fit(P_di[:, :d], absres_di)
    floor = max(float(np.percentile(absres_di, 5)), 1e-6)

    def sigma(P):
        return np.maximum(m.predict(P[:, :d]), floor)
    return sigma


def affine_loo(pred_k, y_k, pred_eval):
    A = np.c_[pred_k, np.ones_like(pred_k)]
    coef, *_ = np.linalg.lstsq(A, y_k, rcond=None)
    res = y_k - A @ coef
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    h = np.clip(np.diag(H), 0, 0.999)
    loo = res / (1 - h)
    return coef[0] * pred_eval + coef[1], loo


def boot_median_ci(x, n=2000, rng=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    rng = rng or np.random.default_rng(SEED)
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
    ap.add_argument("--models", default="plsr,hgb")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--ks", default="10,25,50,100")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--min-eval", type=int, default=30)
    ap.add_argument("--max-train", type=int, default=0, help="0 = tüm kaynak")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 230)
    t0 = time.time()
    ks_all = [int(k) for k in a.ks.split(",")]

    X = np.load(os.path.join(a.data_dir, f"spectra_{a.arm}.npy"))
    bad = ~np.isfinite(X).all(axis=0)
    if bad.any():
        print(f"  {int(bad.sum())} bant NaN iceriyor, cikariliyor")
        X = X[:, ~bad]
    meta = pd.read_csv(os.path.join(a.data_dir, f"meta_{a.arm}.csv"), low_memory=False)
    lib_all = meta[a.libcol].astype(str).to_numpy()
    rng = np.random.default_rng(a.seed)
    print(f"{a.arm} / {a.track}: {X.shape[0]} spektrum, {X.shape[1]} bant, "
          f"hedef kapsama {1 - a.alpha:.2f}, max-train={a.max_train or 'hepsi'}")

    rows = []
    for prop in a.props.split(","):
        col = f"{prop}__{a.track}"
        if col not in meta.columns:
            print(f"  {prop}: {col} yok, atlandi")
            continue
        y_all = to_num(meta[col]).to_numpy(float)
        if prop in LOGPROPS:
            y_all = np.log1p(np.clip(y_all, 0, None))
        ok = np.isfinite(y_all) & np.isfinite(X).all(axis=1)
        vc = pd.Series(lib_all[ok]).value_counts()
        libs = vc[vc >= a.min_n].index.tolist()
        if len(libs) < 3:
            print(f"  {prop}: {len(libs)} kutuphane, yetersiz")
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

            # kaynak: %10 test, %20 kalibrasyon, %10 zorluk modeli, kalan eğitim
            si = rng.permutation(np.flatnonzero(s))
            n10, n20 = int(0.10 * len(si)), int(0.20 * len(si))
            te = si[:n10]
            ca = si[n10:n10 + n20]
            di = si[n10 + n20:2 * n10 + n20]
            tr = si[2 * n10 + n20:]
            if a.max_train and len(tr) > a.max_train:
                tr = tr[:a.max_train]
            lo, hi = np.percentile(y_all[tr], [10, 90])

            for model in a.models.split(","):
                pca, est = fit_model(X[tr], y_all[tr], model)
                yh_ca, P_ca = predict(pca, est, X[ca])
                yh_te, P_te = predict(pca, est, X[te])
                yh_di, P_di = predict(pca, est, X[di])
                yh_t, P_t = predict(pca, est, Xt)

                # C0
                sc_ca = np.abs(y_all[ca] - yh_ca)
                q0 = conf_q(sc_ca, a.alpha)
                cov_src = float(np.mean(np.abs(y_all[te] - yh_te) <= q0))

                # C0n: normalize
                sig = fit_sigma(P_di, np.abs(y_all[di] - yh_di))
                s_ca, s_te, s_t = sig(P_ca), sig(P_te), sig(P_t)
                q0n = conf_q(sc_ca / s_ca, a.alpha)
                cov_src_n = float(np.mean(np.abs(y_all[te] - yh_te) <= q0n * s_te))

                # C1: weighted
                wfun = density_ratio(P_ca, P_t)
                w_ca, w_t = wfun(P_ca), wfun(P_t)
                ess = float(w_ca.sum() ** 2 / (w_ca ** 2).sum())
                q1_all = weighted_q(sc_ca, w_ca, w_t, a.alpha)

                acc = {}

                def add(key, hit, halfw, inside, inf=0.0):
                    d = acc.setdefault(key, dict(cov=[], w=[], inf=[], cin=[], cout=[]))
                    d["cov"].append(hit.mean())
                    hw = np.asarray(halfw, float)
                    fin = np.isfinite(hw)
                    d["w"].append(float(np.median(2 * hw[fin]) / sd_t) if fin.any() else np.inf)
                    d["inf"].append(float(inf))
                    d["cin"].append(hit[inside].mean() if inside.any() else np.nan)
                    d["cout"].append(hit[~inside].mean() if (~inside).any() else np.nan)

                for r in range(a.reps):
                    perm = rng.permutation(nt)
                    pool, ev = perm[:kmax], perm[kmax:]
                    y_ev, p_ev, sg_ev = yt[ev], yh_t[ev], s_t[ev]
                    ae = np.abs(y_ev - p_ev)
                    inside = (y_ev >= lo) & (y_ev <= hi)

                    add(("C0", 0), ae <= q0, np.full(len(ev), q0), inside)
                    add(("C0n", 0), ae <= q0n * sg_ev, q0n * sg_ev, inside)

                    q1 = q1_all[ev]
                    fin1 = np.isfinite(q1)
                    add(("C1", 0), ae <= q1, q1, inside, inf=(~fin1).mean())
                    hw_match = float(np.median(q1[fin1])) if fin1.any() else np.inf
                    add(("C0w", 0), ae <= hw_match, np.full(len(ev), hw_match), inside)

                    for k in ks:
                        ck = pool[:k]
                        # C2
                        q2 = conf_q(np.abs(yt[ck] - yh_t[ck]), a.alpha)
                        add(("C2", k), ae <= q2, np.full(len(ev), q2), inside,
                            inf=float(np.isinf(q2)))
                        # C3
                        p3, loo = affine_loo(yh_t[ck], yt[ck], p_ev)
                        q3 = conf_q(np.abs(loo), a.alpha)
                        add(("C3", k), np.abs(y_ev - p3) <= q3, np.full(len(ev), q3),
                            inside, inf=float(np.isinf(q3)))
                        # C3n: afin + normalize
                        q3n = conf_q(np.abs(loo) / s_t[ck], a.alpha)
                        add(("C3n", k), np.abs(y_ev - p3) <= q3n * sg_ev, q3n * sg_ev,
                            inside, inf=float(np.isinf(q3n)))

                for (meth, k), d in acc.items():
                    wv = np.asarray(d["w"], float)
                    rows.append(dict(
                        prop=prop, model=model, target=tgt, n_target=nt,
                        method=meth, k=k,
                        coverage=round(float(np.nanmean(d["cov"])), 4),
                        coverage_sd=round(float(np.nanstd(d["cov"])), 4),
                        width_sd=round(float(np.nanmedian(wv[np.isfinite(wv)])), 4)
                        if np.isfinite(wv).any() else np.inf,
                        frac_inf=round(float(np.mean(d["inf"])), 4),
                        cov_inside=round(float(np.nanmean(d["cin"])), 4),
                        cov_outside=round(float(np.nanmean(d["cout"])), 4),
                        frac_outside_target=round(float(np.mean((yt < lo) | (yt > hi))), 4),
                        cov_source_test=round(cov_src, 4),
                        cov_source_test_norm=round(cov_src_n, 4),
                        ess_c1=round(ess, 1),
                        n_train=len(tr), n_cal_source=len(ca),
                    ))
                g = lambda key: np.mean(acc[key]["cov"]) if key in acc else np.nan
                k25 = 25 if 25 in ks else ks[0]
                print(f"  {tgt:>18s} {model:>4s} n={nt:6d} | kaynak {cov_src:.3f} | "
                      f"C0 {g(('C0', 0)):.3f} C0n {g(('C0n', 0)):.3f} "
                      f"C1 {g(('C1', 0)):.3f} C0w {g(('C0w', 0)):.3f} | "
                      f"C2/C3/C3n(k={k25}) {g(('C2', k25)):.3f} {g(('C3', k25)):.3f} "
                      f"{g(('C3n', k25)):.3f}  [{time.time() - t0:.0f} sn]")

    if not rows:
        raise SystemExit("hic hucre kosmadi")
    res = pd.DataFrame(rows)
    tag = f"{a.arm}_{a.track}"
    res.to_csv(os.path.join(a.out, f"asama02_hucreler_{tag}.csv"), index=False)

    # ---------------- özet ----------------
    brng = np.random.default_rng(a.seed)
    out_rows = []
    for (meth, k), d in res.groupby(["method", "k"]):
        lo_ci, hi_ci = boot_median_ci(d["coverage"], rng=brng)
        out_rows.append(dict(
            method=meth, k=k, hucre=len(d),
            kapsama_medyan=d["coverage"].median(), ga_alt=lo_ci, ga_ust=hi_ci,
            kapsama_min=d["coverage"].min(),
            genislik_medyan=d["width_sd"].replace(np.inf, np.nan).median(),
            sonsuz_oran=d["frac_inf"].mean(),
            kapsama_ici=d["cov_inside"].median(),
            kapsama_disi=d["cov_outside"].median()))
    order = {"C0": 0, "C0w": 1, "C0n": 2, "C1": 3, "C2": 4, "C3": 5, "C3n": 6}
    summ = (pd.DataFrame(out_rows)
              .assign(_o=lambda x: x["method"].map(order))
              .sort_values(["_o", "k"]).drop(columns="_o").round(3))
    summ.to_csv(os.path.join(a.out, f"asama02_ozet_{tag}.csv"), index=False)

    byprop = (res[res["method"].isin(["C0", "C0n", "C1", "C0w"]) | (res["k"] == 25)]
              .pivot_table(index="prop", columns=["method", "k"],
                           values="coverage", aggfunc="median").round(3))
    bymodel = (res[res["method"].isin(["C0", "C1"]) | (res["k"] == 25)]
               .pivot_table(index="model", columns=["method", "k"],
                            values="coverage", aggfunc="median").round(3))

    print("\n\n" + "=" * 90)
    print(f"OZET — hedef kapsama {1 - a.alpha:.2f}  ({tag})")
    print("=" * 90)
    print(summ.to_string(index=False))
    print("\nozellik bazinda kapsama medyani:")
    print(byprop.to_string())
    print("\nmodel bazinda kapsama medyani:")
    print(bymodel.to_string())

    # ---------------- beklentiler ----------------
    S = summ.set_index(["method", "k"])
    src = res.drop_duplicates(["prop", "model", "target"])["cov_source_test"].median()
    srcn = res.drop_duplicates(["prop", "model", "target"])["cov_source_test_norm"].median()

    def val(mk, col):
        return S.loc[mk, col] if mk in S.index else np.nan

    print("\nHAT KONTROLU")
    print(f"  kaynak-ici kapsama medyani: C0 {src:.3f}   C0n {srcn:.3f}   (beklenen ~0.90)")
    print("\nONCEDEN BELIRTILEN BEKLENTILER")
    b1 = val(("C0", 0), "ga_ust") < 1 - a.alpha
    print(f"  B1 C0 medyan {val(('C0', 0), 'kapsama_medyan'):.3f} "
          f"[{val(('C0', 0), 'ga_alt'):.3f}, {val(('C0', 0), 'ga_ust'):.3f}]  "
          f"ust sinir < {1 - a.alpha:.2f}? {'EVET' if b1 else 'HAYIR'}")
    b2 = val(("C1", 0), "ga_ust") < val(("C2", 25), "ga_alt")
    print(f"  B2 C1 {val(('C1', 0), 'kapsama_medyan'):.3f} "
          f"[{val(('C1', 0), 'ga_alt'):.3f}, {val(('C1', 0), 'ga_ust'):.3f}] vs "
          f"C2(25) {val(('C2', 25), 'kapsama_medyan'):.3f} "
          f"[{val(('C2', 25), 'ga_alt'):.3f}, {val(('C2', 25), 'ga_ust'):.3f}]  "
          f"ayrik? {'EVET' if b2 else 'HAYIR'}")
    dw = val(("C1", 0), "kapsama_medyan") - val(("C0w", 0), "kapsama_medyan")
    print(f"  B3 C1 - C0w = {dw:+.3f}  (< 0.03 ise C1 kazanci buyuk olcude genislikten) "
          f"-> {'EVET' if abs(dw) < 0.03 else 'HAYIR'}")
    for k in (25, 50, 100):
        w2, w3 = val(("C2", k), "genislik_medyan"), val(("C3", k), "genislik_medyan")
        print(f"  B4 k={k:3d}: genislik C2 {w2:.3f}  C3 {w3:.3f}  C3n "
              f"{val(('C3n', k), 'genislik_medyan'):.3f}  -> C3 daha dar? "
              f"{'EVET' if w3 < w2 else 'HAYIR'}")
    c0, c0n = val(("C0", 0), "kapsama_medyan"), val(("C0n", 0), "kapsama_medyan")
    print(f"  B5 C0 {c0:.3f} -> C0n {c0n:.3f}  (iyilesir ama nominale ulasmaz?) -> "
          f"{'EVET' if (c0n > c0 and val(('C0n', 0), 'ga_ust') < 1 - a.alpha) else 'HAYIR'}")
    print("\nKAYNAK ARALIGI DISI KAPSAMA (k=50):")
    for meth in ("C2", "C3", "C3n"):
        print(f"  {meth}: ici {val((meth, 50), 'kapsama_ici'):.3f}   "
              f"disi {val((meth, 50), 'kapsama_disi'):.3f}")
    print(f"\ncikti: {os.path.abspath(a.out)}   ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
