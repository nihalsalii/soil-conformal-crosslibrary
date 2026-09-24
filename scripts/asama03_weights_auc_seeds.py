#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aşama 03 — Weighted conformal'a adil test, etiketsiz uyarı işareti, tohum sağlamlığı
====================================================================================
asama02'den sonra açık kalan dört soru:
  1. Ağırlıkları kırpılmış weighted conformal (C1c) sonsuz aralıkları giderip
     kapsamayı nominale taşıyabiliyor mu? (Hakem: "C1'e haksızlık ettiniz")
  2. C1'in yalnızca SONLU aralıklarındaki kapsama nedir? (bilgi taşıyan kısmı)
  3. Alan sınıflandırıcısının AUC'si (kaynak ve hedef ne kadar ayrışıyor?)
     hedef etiketi gelmeden kapsama kaybını öngörebiliyor mu?
  4. Sonuçlar kaynak bölmesinin tohumuna ne kadar duyarlı?

Yöntemler (asama02 ile aynı tanımlar):
  C0   split conformal, kaynakta kalibre
  C0w  C0, C1'in medyan (sonlu) yarı-genişliğiyle
  C1   weighted conformal, olasılıklar [0.01, 0.99]'a kırpılı (asama02 ile aynı)
  C1c  weighted conformal, olasılıklar [0.05, 0.95]'e kırpılı (en büyük ağırlık
       oranı 19 ile sınırlı; yaygın kararlı uygulama)
  C2   k=25 hedef etiketiyle eşik yeniden kalibre (referans)
  C3n  k=25 hedef etiketiyle afin + normalize conformal (önerilen yöntem, referans)

Önceden belirtilen beklentiler (asama03 çalıştırılmadan yazıldı):
  E1 C1c sonsuz oranı < 0.05, ama kapsama medyanı C3n(25)'in GA alt sınırının altında
  E2 C1'in yalnızca sonlu aralıklarındaki kapsama medyanı < 0.90
  E3 AUC ile C1 sonsuz oranı arasında Spearman rho > 0.5
  E4 (keşifsel) AUC, C0 kapsama kaybını (0.90 - C0) öngörür: rho > 0.4 ve p < 0.05
  E5 C0 kapsama medyanı tohumlar arasında 0.03'ten az değişir

Kullanım (NEW ARTİCLE klasöründen, ossl ortamında):
  python scripts\\asama03_weights_auc_seeds.py --arm mir --track clean
  python scripts\\asama03_weights_auc_seeds.py --arm visnir --track naive
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

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


def fit_model(Xtr, ytr, model, seed, n_pc=120, n_plsr=20):
    Z = snv(Xtr)
    npc = int(min(n_pc, Z.shape[0] - 1, Z.shape[1]))
    pca = PCA(n_components=npc, random_state=seed).fit(Z)
    P = pca.transform(Z)
    if model == "plsr":
        k = int(min(n_plsr, P.shape[1], max(2, P.shape[0] - 1)))
        est = PLSRegression(n_components=k).fit(P, ytr)
    else:
        est = HistGradientBoostingRegressor(random_state=seed).fit(P, ytr)
    return pca, est


def predict(pca, est, X):
    P = pca.transform(snv(X))
    return np.asarray(est.predict(P)).ravel(), P


def conf_q(scores, alpha):
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


def domain_classifier(P_cal, P_tgt, seed, n_feat=20):
    """Alan sınıflandırıcısı + 5 katlı çapraz geçerli AUC (hedef etiketi kullanılmaz).
    Döndürür: p(x) fonksiyonu (tüm veriyle fit) ve CV AUC."""
    d = int(min(n_feat, P_cal.shape[1]))
    Xd = np.vstack([P_cal[:, :d], P_tgt[:, :d]])
    yd = np.r_[np.zeros(len(P_cal)), np.ones(len(P_tgt))]
    sc = StandardScaler().fit(Xd)
    Xs = sc.transform(Xd)
    oof = np.zeros(len(yd))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, yd):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], yd[tr])
        oof[te] = m.predict_proba(Xs[te])[:, 1]
    auc = float(roc_auc_score(yd, oof))
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, yd)

    def prob(P):
        return clf.predict_proba(sc.transform(P[:, :d]))[:, 1]
    return prob, auc


def to_weight(p, lo, hi, n_cal, n_tgt):
    p = np.clip(p, lo, hi)
    return (p / (1 - p)) * (n_cal / n_tgt)


def fit_sigma(P_di, absres_di, seed, n_feat=20):
    d = int(min(n_feat, P_di.shape[1]))
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                      random_state=seed).fit(P_di[:, :d], absres_di)
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
def run_seed(X, meta, lib_all, a, seed, t0):
    rng = np.random.default_rng(seed)
    rows = []
    k = a.k
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

        for tgt in libs:
            m = ok & (lib_all == tgt)
            s = ok & (lib_all != tgt)
            Xt, yt = X[m], y_all[m]
            nt = len(yt)
            if nt - k < a.min_eval:
                continue
            sd_t = yt.std(ddof=1)

            si = rng.permutation(np.flatnonzero(s))
            n10, n20 = int(0.10 * len(si)), int(0.20 * len(si))
            ca = si[n10:n10 + n20]
            di = si[n10 + n20:2 * n10 + n20]
            tr = si[2 * n10 + n20:]

            for model in a.models.split(","):
                pca, est = fit_model(X[tr], y_all[tr], model, seed)
                yh_ca, P_ca = predict(pca, est, X[ca])
                yh_di, P_di = predict(pca, est, X[di])
                yh_t, P_t = predict(pca, est, Xt)

                sc_ca = np.abs(y_all[ca] - yh_ca)
                q0 = conf_q(sc_ca, a.alpha)
                sig = fit_sigma(P_di, np.abs(y_all[di] - yh_di), seed)
                s_t = sig(P_t)

                prob, auc = domain_classifier(P_ca, P_t, seed)
                p_ca, p_t = prob(P_ca), prob(P_t)
                w1c, w1t = (to_weight(p_ca, .01, .99, len(ca), nt),
                            to_weight(p_t, .01, .99, len(ca), nt))
                wcc, wct = (to_weight(p_ca, .05, .95, len(ca), nt),
                            to_weight(p_t, .05, .95, len(ca), nt))
                ess1 = float(w1c.sum() ** 2 / (w1c ** 2).sum())
                essc = float(wcc.sum() ** 2 / (wcc ** 2).sum())
                q1_all = weighted_q(sc_ca, w1c, w1t, a.alpha)
                q1c_all = weighted_q(sc_ca, wcc, wct, a.alpha)

                acc = {key: dict(cov=[], w=[], inf=[], covfin=[]) for key in
                       ("C0", "C0w", "C1", "C1c", "C2", "C3n")}

                def add(key, hit, halfw, fin=None):
                    d = acc[key]
                    hw = np.asarray(halfw, float)
                    f = np.isfinite(hw) if fin is None else fin
                    d["cov"].append(hit.mean())
                    d["w"].append(float(np.median(2 * hw[f]) / sd_t) if f.any() else np.inf)
                    d["inf"].append(float((~f).mean()))
                    d["covfin"].append(hit[f].mean() if f.any() else np.nan)

                for _ in range(a.reps):
                    perm = rng.permutation(nt)
                    ck, ev = perm[:k], perm[k:]
                    y_ev, p_ev = yt[ev], yh_t[ev]
                    ae = np.abs(y_ev - p_ev)

                    add("C0", ae <= q0, np.full(len(ev), q0))
                    q1 = q1_all[ev]
                    add("C1", ae <= q1, q1)
                    fin = np.isfinite(q1)
                    hw = float(np.median(q1[fin])) if fin.any() else np.inf
                    add("C0w", ae <= hw, np.full(len(ev), hw))
                    q1c = q1c_all[ev]
                    add("C1c", ae <= q1c, q1c)

                    q2 = conf_q(np.abs(yt[ck] - yh_t[ck]), a.alpha)
                    add("C2", ae <= q2, np.full(len(ev), q2))
                    p3, loo = affine_loo(yh_t[ck], yt[ck], p_ev)
                    q3n = conf_q(np.abs(loo) / s_t[ck], a.alpha)
                    add("C3n", np.abs(y_ev - p3) <= q3n * s_t[ev], q3n * s_t[ev])

                for meth, d in acc.items():
                    wv = np.asarray(d["w"], float)
                    rows.append(dict(
                        seed=seed, prop=prop, model=model, target=tgt, n_target=nt,
                        method=meth, coverage=float(np.nanmean(d["cov"])),
                        coverage_finite=float(np.nanmean(d["covfin"])),
                        width_sd=float(np.nanmedian(wv[np.isfinite(wv)]))
                        if np.isfinite(wv).any() else np.inf,
                        frac_inf=float(np.mean(d["inf"])),
                        auc=auc, ess_c1=ess1, ess_c1c=essc,
                        n_cal_source=len(ca)))
                cv = {mm: np.mean(acc[mm]["cov"]) for mm in acc}
                print(f"  tohum {seed:>4d} {prop:>8s} {tgt:>18s} {model:>4s} | AUC {auc:.3f} | "
                      f"C0 {cv['C0']:.3f} C1 {cv['C1']:.3f} (sonsuz "
                      f"{np.mean(acc['C1']['inf']):.2f}) C1c {cv['C1c']:.3f} | "
                      f"C3n {cv['C3n']:.3f}  [{time.time() - t0:.0f} sn]")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--arm", default="mir", choices=["mir", "visnir"])
    ap.add_argument("--track", default="clean", choices=["clean", "naive"])
    ap.add_argument("--libcol", default="dataset.code_ascii_txt")
    ap.add_argument("--props", default=",".join(PROPS))
    ap.add_argument("--models", default="plsr,hgb")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seeds", default="42,7,2026")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--min-eval", type=int, default=30)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 230)
    t0 = time.time()

    X = np.load(os.path.join(a.data_dir, f"spectra_{a.arm}.npy"))
    bad = ~np.isfinite(X).all(axis=0)
    if bad.any():
        print(f"  {int(bad.sum())} bant NaN iceriyor, cikariliyor")
        X = X[:, ~bad]
    meta = pd.read_csv(os.path.join(a.data_dir, f"meta_{a.arm}.csv"), low_memory=False)
    lib_all = meta[a.libcol].astype(str).to_numpy()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"{a.arm} / {a.track}: {X.shape[0]} spektrum | tohumlar {seeds} | k={a.k}")

    rows = []
    for seed in seeds:
        rows += run_seed(X, meta, lib_all, a, seed, t0)
    if not rows:
        raise SystemExit("hic hucre kosmadi")
    res = pd.DataFrame(rows)
    tag = f"{a.arm}_{a.track}"
    res.to_csv(os.path.join(a.out, f"asama03_hucreler_{tag}.csv"), index=False)

    # hücre = özellik x model x hedef; tohumlar üzerinden ortalama
    cell = (res.groupby(["prop", "model", "target", "method"])
               [["coverage", "coverage_finite", "width_sd", "frac_inf", "auc"]]
               .mean().reset_index())
    brng = np.random.default_rng(42)
    order = ["C0", "C0w", "C1", "C1c", "C2", "C3n"]
    summ = []
    for meth in order:
        d = cell[cell["method"] == meth]
        lo, hi = boot_median_ci(d["coverage"], brng)
        lof, hif = boot_median_ci(d["coverage_finite"], brng)
        summ.append(dict(method=meth, hucre=len(d),
                         kapsama=d["coverage"].median(), ga_alt=lo, ga_ust=hi,
                         kapsama_sonlu=d["coverage_finite"].median(),
                         sonlu_ga_alt=lof, sonlu_ga_ust=hif,
                         genislik=d["width_sd"].replace(np.inf, np.nan).median(),
                         sonsuz_oran=d["frac_inf"].mean()))
    summ = pd.DataFrame(summ).round(3)
    summ.to_csv(os.path.join(a.out, f"asama03_ozet_{tag}.csv"), index=False)

    wide = cell.pivot_table(index=["prop", "model", "target"], columns="method",
                            values="coverage").reset_index()
    inf1 = cell[cell["method"] == "C1"].set_index(["prop", "model", "target"])["frac_inf"]
    aucs = cell[cell["method"] == "C0"].set_index(["prop", "model", "target"])["auc"]
    wide = wide.set_index(["prop", "model", "target"])
    wide["auc"], wide["c1_inf"] = aucs, inf1
    wide["kayip_c0"] = (1 - a.alpha) - wide["C0"]
    wide["c1_bilgi"] = wide["C1"] - wide["C0w"]
    wide.round(4).to_csv(os.path.join(a.out, f"asama03_auc_{tag}.csv"))

    print("\n\n" + "=" * 92)
    print(f"OZET — hedef kapsama {1 - a.alpha:.2f}  ({tag}, {len(seeds)} tohum ortalamasi)")
    print("=" * 92)
    print(summ.to_string(index=False))

    print("\nAUC dagilimi (alan siniflandiricisi, 5 katli CV):")
    print(f"  medyan {wide['auc'].median():.3f}   min {wide['auc'].min():.3f}   "
          f"max {wide['auc'].max():.3f}   AUC>0.95 olan hucre orani "
          f"{(wide['auc'] > 0.95).mean():.2f}")

    r_inf, p_inf = spearmanr(wide["auc"], wide["c1_inf"], nan_policy="omit")
    r_loss, p_loss = spearmanr(wide["auc"], wide["kayip_c0"], nan_policy="omit")
    r_info, p_info = spearmanr(wide["auc"], wide["c1_bilgi"], nan_policy="omit")
    print(f"  AUC ~ C1 sonsuz orani       rho {r_inf:+.3f}  p={p_inf:.4f}")
    print(f"  AUC ~ C0 kapsama kaybi      rho {r_loss:+.3f}  p={p_loss:.4f}")
    print(f"  AUC ~ C1 bilgi (C1 - C0w)   rho {r_info:+.3f}  p={p_info:.4f}")

    per_seed = (res[res["method"] == "C0"].groupby("seed")["coverage"].median())
    print("\nC0 kapsama medyani, tohum bazinda:")
    print("  " + "   ".join(f"{s}: {v:.3f}" for s, v in per_seed.items()))
    spread = per_seed.max() - per_seed.min()

    S = summ.set_index("method")
    print("\nONCEDEN BELIRTILEN BEKLENTILER")
    e1 = (S.loc["C1c", "sonsuz_oran"] < 0.05) and (S.loc["C1c", "kapsama"] < S.loc["C3n", "ga_alt"])
    print(f"  E1 C1c sonsuz {S.loc['C1c', 'sonsuz_oran']:.3f}, kapsama {S.loc['C1c', 'kapsama']:.3f} "
          f"vs C3n GA alt {S.loc['C3n', 'ga_alt']:.3f}  -> {'EVET' if e1 else 'HAYIR'}")
    e2 = S.loc["C1", "kapsama_sonlu"] < 1 - a.alpha
    print(f"  E2 C1 sonlu aralik kapsamasi {S.loc['C1', 'kapsama_sonlu']:.3f} "
          f"[{S.loc['C1', 'sonlu_ga_alt']:.3f}, {S.loc['C1', 'sonlu_ga_ust']:.3f}] < 0.90? "
          f"{'EVET' if e2 else 'HAYIR'}")
    print(f"  E3 AUC ~ C1 sonsuz rho {r_inf:+.3f} > 0.5? {'EVET' if r_inf > 0.5 else 'HAYIR'}")
    e4 = (r_loss > 0.4) and (p_loss < 0.05)
    print(f"  E4 AUC ~ C0 kaybi rho {r_loss:+.3f} p={p_loss:.4f} (rho>0.4, p<0.05)? "
          f"{'EVET' if e4 else 'HAYIR'}")
    print(f"  E5 C0 tohumlar arasi fark {spread:.3f} < 0.03? {'EVET' if spread < 0.03 else 'HAYIR'}")
    print(f"\ncikti: {os.path.abspath(a.out)}   ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
