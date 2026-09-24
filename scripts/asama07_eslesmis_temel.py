#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aşama 07 — Eşleştirilmiş temel çizgi: CNN protokolüyle PLSR ve HGB
==================================================================
asama06'da CNN, asama02'den farklı bir protokolle koştu (kaynak 8000 örnekle
sınırlı, spektrum 4'lü ortalamayla indirgenmiş, tek tohum). Bu yüzden CNN ile
PLSR/HGB arasındaki kapsama farkı model ailesine mi yoksa protokole mi bağlı,
ayırt edilemiyor.

Bu betik PLSR ve HGB'yi TAM OLARAK CNN'in protokolüyle çalıştırır: aynı
--bin, aynı --max-train, aynı --seed, aynı bölme mantığı, aynı yöntemler
(C0, C0n, C2, C3n) ve aynı k değerleri. Böylece üç aile birebir
karşılaştırılabilir.

Önceden belirtilen beklentiler (betik çalıştırılmadan yazıldı):
  H1 Eşleştirilmiş protokolde her üç ailede de C0 medyanı nominalin altında
  H2 Eşleştirilmiş PLSR/HGB medyanı ile CNN medyanı arasındaki fark, asama02
     ile CNN arasındaki farktan (MIR'de 0.091) küçük -> farkın bir kısmı
     protokolden geliyor
  H3 k = 25'te üç aile de nominale ulaşır

Kullanım (asama06 ile aynı parametreler verilmeli):
  python scripts\\asama07_eslesmis_temel.py --arm mir --track clean
  python scripts\\asama07_eslesmis_temel.py --arm visnir --track naive
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


def binned(X, b):
    if b <= 1:
        return X
    n = (X.shape[1] // b) * b
    return X[:, :n].reshape(X.shape[0], n // b, b).mean(axis=2)


def conf_q(scores, alpha):
    n = len(scores)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if r > n else float(np.sort(scores)[r - 1])


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


def fit_predict(model, Xtr, ytr, seed, n_pc, n_plsr):
    """CNN ile aynı girdi (binlenmiş SNV spektrumu); PCA yalnızca kaynakta."""
    npc = int(min(n_pc, Xtr.shape[0] - 1, Xtr.shape[1]))
    pca = PCA(n_components=npc, random_state=seed).fit(Xtr)
    P = pca.transform(Xtr)
    if model == "plsr":
        k = int(min(n_plsr, P.shape[1], max(2, P.shape[0] - 1)))
        est = PLSRegression(n_components=k).fit(P, ytr)
        return lambda X: np.asarray(est.predict(pca.transform(X))).ravel(), pca
    est = HistGradientBoostingRegressor(random_state=seed).fit(P, ytr)
    return lambda X: est.predict(pca.transform(X)), pca


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
    ap.add_argument("--ks", default="25,50")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--min-eval", type=int, default=30)
    ap.add_argument("--max-train", type=int, default=8000, help="asama06 ile aynı olmalı")
    ap.add_argument("--bin", type=int, default=4, help="asama06 ile aynı olmalı")
    ap.add_argument("--n-pc", type=int, default=120)
    ap.add_argument("--n-plsr", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED, help="asama06 ile aynı olmalı")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 220)
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
    Xb = binned(snv(X), a.bin).astype(np.float32)
    print(f"{a.arm}/{a.track}: {X.shape[1]} -> {Xb.shape[1]} bant (bin={a.bin}) | "
          f"eslestirilmis protokol, max-train={a.max_train}, tohum={a.seed}")

    rows = []
    for prop in a.props.split(","):
        col = f"{prop}__{a.track}"
        if col not in meta.columns:
            continue
        y_all = to_num(meta[col]).to_numpy(float)
        if prop in LOGPROPS:
            y_all = np.log1p(np.clip(y_all, 0, None))
        ok = np.isfinite(y_all) & np.isfinite(Xb).all(axis=1)
        vc = pd.Series(lib_all[ok]).value_counts()
        libs = vc[vc >= a.min_n].index.tolist()
        if len(libs) < 3:
            continue
        print(f"\n=== {prop} ({int(ok.sum())} ornek, {len(libs)} kutuphane)")

        for tgt in libs:
            m = ok & (lib_all == tgt)
            s = ok & (lib_all != tgt)
            Xt, yt = Xb[m], y_all[m]
            nt = len(yt)
            ks = [k for k in ks_all if nt - k >= a.min_eval]
            if not ks:
                continue
            kmax = max(ks)
            sd_t = yt.std(ddof=1)

            # asama06 ile AYNI bölme mantığı ve aynı tohum
            si = rng.permutation(np.flatnonzero(s))
            n10, n20 = int(0.10 * len(si)), int(0.20 * len(si))
            te, ca = si[:n10], si[n10:n10 + n20]
            di, tr = si[n10 + n20:2 * n10 + n20], si[2 * n10 + n20:]
            if a.max_train and len(tr) > a.max_train:
                tr = tr[:a.max_train]
            if len(ca) > 4000:
                ca = ca[:4000]
            if len(te) > 2000:
                te = te[:2000]
            if len(di) > 4000:
                di = di[:4000]

            for model in a.models.split(","):
                pred, pca = fit_predict(model, Xb[tr], y_all[tr], a.seed, a.n_pc, a.n_plsr)
                yh_ca, yh_te, yh_di, yh_t = (pred(Xb[ca]), pred(Xb[te]),
                                             pred(Xb[di]), pred(Xt))

                sig_m = HistGradientBoostingRegressor(
                    max_iter=200, learning_rate=0.05, random_state=a.seed).fit(
                    pca.transform(Xb[di])[:, :20], np.abs(y_all[di] - yh_di))
                floor = max(float(np.percentile(np.abs(y_all[di] - yh_di), 5)), 1e-6)
                sg = lambda Z: np.maximum(sig_m.predict(pca.transform(Z)[:, :20]), floor)
                s_ca, s_te, s_t = sg(Xb[ca]), sg(Xb[te]), sg(Xt)

                sc = np.abs(y_all[ca] - yh_ca)
                q0, q0n = conf_q(sc, a.alpha), conf_q(sc / s_ca, a.alpha)
                cov_src = float(np.mean(np.abs(y_all[te] - yh_te) <= q0))
                rmse_src = float(np.sqrt(np.mean((y_all[te] - yh_te) ** 2)) / y_all[te].std())
                rmse_tgt = float(np.sqrt(np.mean((yt - yh_t) ** 2)) / sd_t)

                acc = {}

                def add(key, hit, width):
                    d = acc.setdefault(key, dict(cov=[], w=[]))
                    d["cov"].append(float(np.mean(hit)))
                    d["w"].append(float(np.median(width) / sd_t))

                for _ in range(a.reps):
                    perm = rng.permutation(nt)
                    pool, ev = perm[:kmax], perm[kmax:]
                    y_ev, p_ev, sge = yt[ev], yh_t[ev], s_t[ev]
                    ae = np.abs(y_ev - p_ev)
                    add(("C0", 0), ae <= q0, np.full(len(ev), 2 * q0))
                    add(("C0n", 0), ae <= q0n * sge, 2 * q0n * sge)
                    for k in ks:
                        ck = pool[:k]
                        q2 = conf_q(np.abs(yt[ck] - yh_t[ck]), a.alpha)
                        add(("C2", k), ae <= q2, np.full(len(ev), 2 * q2))
                        p3, loo = affine_loo(yh_t[ck], yt[ck], p_ev)
                        q3n = conf_q(np.abs(loo) / s_t[ck], a.alpha)
                        add(("C3n", k), np.abs(y_ev - p3) <= q3n * sge, 2 * q3n * sge)

                for (meth, k), d in acc.items():
                    rows.append(dict(prop=prop, model=model, target=tgt, n_target=nt,
                                     method=meth, k=k,
                                     coverage=float(np.mean(d["cov"])),
                                     width_sd=float(np.median(d["w"])),
                                     cov_source_test=cov_src,
                                     rmse_std_source=rmse_src,
                                     rmse_std_target=rmse_tgt, n_train=len(tr)))
                print(f"  {tgt:>18s} {model:>4s} n={nt:6d} | kaynak-ici {cov_src:.3f} "
                      f"(rmse/std {rmse_src:.3f}) | C0 {np.mean(acc[('C0', 0)]['cov']):.3f} "
                      f"C0n {np.mean(acc[('C0n', 0)]['cov']):.3f} "
                      f"C2({ks[0]}) {np.mean(acc[('C2', ks[0])]['cov']):.3f} "
                      f"C3n({ks[0]}) {np.mean(acc[('C3n', ks[0])]['cov']):.3f}"
                      f"  [{time.time() - t0:.0f} sn]")

    if not rows:
        raise SystemExit("hic hucre kosmadi")
    res = pd.DataFrame(rows)
    tag = f"{a.arm}_{a.track}"
    res.to_csv(os.path.join(a.out, f"asama07_hucreler_{tag}.csv"), index=False)

    brng = np.random.default_rng(a.seed)
    out = []
    for (model, meth, k), d in res.groupby(["model", "method", "k"]):
        lo, hi = boot_median_ci(d["coverage"], brng)
        out.append(dict(model=model, method=meth, k=k, cells=len(d),
                        coverage=d["coverage"].median(), ci_low=lo, ci_high=hi,
                        worst=d["coverage"].min(), width=d["width_sd"].median()))
    summ = pd.DataFrame(out).round(3).sort_values(["method", "k", "model"])
    summ.to_csv(os.path.join(a.out, f"asama07_ozet_{tag}.csv"), index=False)

    print("\n\n" + "=" * 88)
    print(f"OZET — eslestirilmis protokol ({tag}), hedef kapsama {1 - a.alpha:.2f}")
    print("=" * 88)
    print(summ.to_string(index=False))

    # CNN ile karşılaştırma (asama06 çıktısı varsa)
    cnnp = os.path.join(a.out, f"asama06_ozet_cnn_{tag}.csv")
    S = summ.set_index(["model", "method", "k"])
    print("\nUC AILE, AYNI PROTOKOL (C0 medyani)")
    vals = {}
    for model in a.models.split(","):
        if (model, "C0", 0) in S.index:
            vals[model] = S.loc[(model, "C0", 0), "coverage"]
            print(f"  {model:>5s}: {vals[model]:.3f} "
                  f"[{S.loc[(model, 'C0', 0), 'ci_low']:.3f}, "
                  f"{S.loc[(model, 'C0', 0), 'ci_high']:.3f}]  "
                  f"en kotu {S.loc[(model, 'C0', 0), 'worst']:.3f}")
    if os.path.exists(cnnp):
        cnn = pd.read_csv(cnnp).set_index(["method", "k"])
        c = float(cnn.loc[("C0", 0), "coverage"])
        vals["cnn"] = c
        print(f"    cnn: {c:.3f} "
              f"[{float(cnn.loc[('C0', 0), 'ci_low']):.3f}, "
              f"{float(cnn.loc[('C0', 0), 'ci_high']):.3f}]  "
              f"en kotu {float(cnn.loc[('C0', 0), 'worst']):.3f}")
        spread = max(vals.values()) - min(vals.values())
        print(f"\n  aileler arasi en buyuk fark: {spread:.3f}")
        print("\nONCEDEN BELIRTILEN BEKLENTILER")
        h1 = all(S.loc[(m, "C0", 0), "ci_high"] < 1 - a.alpha
                 for m in a.models.split(",") if (m, "C0", 0) in S.index) \
            and float(cnn.loc[("C0", 0), "ci_high"]) < 1 - a.alpha
        print(f"  H1 uc ailede de C0 nominalin altinda? {'EVET' if h1 else 'HAYIR'}")
        print(f"  H2 eslestirilmis fark {spread:.3f} < asama02-CNN farki "
              f"(mir 0.091 / visnir 0.002)? -> karsilastir")
        h3 = all(S.loc[(m, "C2", ks_all[0]), "ci_high"] >= 1 - a.alpha
                 for m in a.models.split(",") if (m, "C2", ks_all[0]) in S.index)
        print(f"  H3 k={ks_all[0]}'te tum aileler nominale ulasiyor? "
              f"{'EVET' if h3 else 'HAYIR'}")
    else:
        print(f"\n  (asama06 ozeti bulunamadi: {cnnp} — CNN karsilastirmasi atlandi)")
    print(f"\ncikti: {os.path.abspath(a.out)}   ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
