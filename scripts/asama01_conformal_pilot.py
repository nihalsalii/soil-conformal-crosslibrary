#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aşama 01 — PİLOT: Tahmin aralıkları kütüphane değişince geçerli kalıyor mu?
============================================================================
Soru:
  Kaynak kütüphanelerde kalibre edilen %90'lık bir tahmin aralığı, hiç
  görülmemiş bir hedef kütüphanede gerçekten örneklerin %90'ını kapsıyor mu?
  Kapsamıyorsa, bunu hangi yöntem geri getirir?

Teorik kanca (cross-library makalesinden):
  Kütüphaneler arası kayma KOŞULLU (P(y|x) değişiyor). Weighted conformal
  (Tibshirani ve ark. 2019) yalnızca KOVARYAT kaymayı düzeltmek için
  tasarlanmıştır. Öngörü: C1 kapsamayı geri getiremez, hedeften birkaç
  etiketle yeniden kalibrasyon (C2/C3) getirir.

Yöntemler (aynı hedef değerlendirme kümesinde):
  C0  split conformal, kaynakta kalibre, hedefe olduğu gibi (zero-shot)
  C1  weighted conformal: kaynak kalibrasyon örnekleri, hedefe benzerliğe
      göre ağırlıklı (alan sınıflandırıcısı ile yoğunluk oranı; hedef
      ETİKETİ kullanılmaz)
  C2  k hedef etiketiyle yeniden kalibrasyon (yalnızca aralık genişliği)
  C3  k hedef etiketiyle afin düzeltme + LOO artıklarıyla conformal
      (nokta tahmini ve aralık birlikte düzeltilir)

Düzen asama18 ile aynıdır: leave-one-library-out (kaynak = diğer tüm
kütüphaneler), SNV, PCA(120) yalnızca kaynak eğitimde fit, PLSR + HGB,
oc ve n.tot log1p ölçeğinde.

Önceden yazılmış durdurma kuralları (sonuç görülmeden sabitlendi):
  K0 (hat kontrolü) Kaynak içi test kapsaması medyanı 0.87-0.93 dışında
     ise hatta sorun var; sonuç yorumlanmaz.
  K1 C0 hedef kapsaması medyanı >= 0.85 ise aralıklar zaten sağlam: DUR.
  K2 C1 medyanı, C2(k=25) medyanının 0.02 altından yüksekse ve C1'de
     sonsuz aralık oranı < 0.20 ise koşullu kayma kancası çöker: çerçeve
     değişir.
  Aksi halde tam analize geç.

Kullanım (Windows, cross-library ortamında):
  python asama01_conformal_pilot.py --arm mir --track clean
  python asama01_conformal_pilot.py --arm mir --track clean --max-train 0   (tüm kaynak)
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
# asama18 ile aynı yardımcılar
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
# conformal yardımcıları
def conf_q(scores, alpha):
    """Split conformal eşiği: sıralı artıkların ceil((n+1)(1-alpha))'ıncı değeri.
    n küçükse (ör. %90'da n<9) sonsuz döner.
    Düzeltme (22.09.2026): önceki sürüm np.quantile(method='higher') ile bir
    sonraki sıralı değeri alıyordu (k=25'te maksimumu); aralıklar gereğinden
    genişti. C0/C1'de (n binlerce) etkisi ihmal edilebilir."""
    n = len(scores)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    if r > n:
        return np.inf
    return float(np.sort(scores)[r - 1])


def weighted_q(scores, w_cal, w_test, alpha):
    """Weighted conformal (Tibshirani ve ark. 2019). Her test noktası için
    ayrı eşik: kalibrasyon ağırlıkları + test noktasının kendi ağırlığı
    (sonsuzda bir kütle olarak)."""
    order = np.argsort(scores)
    s, w = scores[order], w_cal[order]
    cw = np.cumsum(w)
    thr = (1 - alpha) * (cw[-1] + w_test)
    idx = np.searchsorted(cw, thr, side="left")
    return np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)


def density_ratio(P_cal, P_tgt, n_feat=20):
    """Alan sınıflandırıcısıyla w(x) = p_T(x)/p_S(x). Hedef etiketi kullanılmaz."""
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


def affine_loo(pred_k, y_k, pred_eval):
    """k etiketle y = a*pred + b; LOO artıkları (hat matrisi ile kapalı form)."""
    A = np.c_[pred_k, np.ones_like(pred_k)]
    coef, *_ = np.linalg.lstsq(A, y_k, rcond=None)
    res = y_k - A @ coef
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    h = np.clip(np.diag(H), 0, 0.999)
    loo = np.abs(res / (1 - h))
    return coef[0] * pred_eval + coef[1], loo


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
    ap.add_argument("--max-train", type=int, default=30000,
                    help="pilot hızı için kaynak eğitim alt örneklemi (0 = hepsi)")
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
    rng = np.random.default_rng(SEED)
    print(f"{a.arm} / {a.track}: {X.shape[0]} spektrum, {X.shape[1]} bant, "
          f"alpha={a.alpha} (hedef kapsama {1 - a.alpha:.2f})")

    rows = []
    for prop in a.props.split(","):
        col = f"{prop}__{a.track}"
        if col not in meta.columns:
            print(f"  {prop}: {col} yok, atlandi")
            continue
        y_all = to_num(meta[col]).to_numpy(float)
        if prop in LOGPROPS:
            y_all = np.log1p(np.clip(y_all, 0, None))
        ok = np.isfinite(y_all)
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

            # kaynak bölmesi: %10 test (kaynak içi kontrol), %20 kalibrasyon, kalan eğitim
            si = rng.permutation(np.flatnonzero(s))
            n_te, n_ca = int(0.10 * len(si)), int(0.20 * len(si))
            te, ca, tr = si[:n_te], si[n_te:n_te + n_ca], si[n_te + n_ca:]
            if a.max_train and len(tr) > a.max_train:
                tr = tr[:a.max_train]
            lo, hi = np.percentile(y_all[tr], [10, 90])

            for model in a.models.split(","):
                pca, est = fit_model(X[tr], y_all[tr], model)
                yh_ca, P_ca = predict(pca, est, X[ca])
                yh_te, _ = predict(pca, est, X[te])
                yh_t, P_t = predict(pca, est, Xt)

                sc_ca = np.abs(y_all[ca] - yh_ca)
                q0 = conf_q(sc_ca, a.alpha)
                cov_src = float(np.mean(np.abs(y_all[te] - yh_te) <= q0))

                wfun = density_ratio(P_ca, P_t)
                w_ca, w_t = wfun(P_ca), wfun(P_t)
                ess = float(w_ca.sum() ** 2 / (w_ca ** 2).sum())
                q1_all = weighted_q(sc_ca, w_ca, w_t, a.alpha)

                acc = {}  # (yöntem, k) -> listeler

                def add(key, cov, width, inf, cin=np.nan, cout=np.nan):
                    d = acc.setdefault(key, dict(cov=[], w=[], inf=[], cin=[], cout=[]))
                    d["cov"].append(cov); d["w"].append(width); d["inf"].append(inf)
                    d["cin"].append(cin); d["cout"].append(cout)

                for r in range(a.reps):
                    perm = rng.permutation(nt)
                    pool, ev = perm[:kmax], perm[kmax:]
                    y_ev, p_ev = yt[ev], yh_t[ev]
                    inside = (y_ev >= lo) & (y_ev <= hi)

                    # C0
                    hit0 = np.abs(y_ev - p_ev) <= q0
                    add(("C0", 0), hit0.mean(), 2 * q0 / sd_t, 0.0,
                        hit0[inside].mean() if inside.any() else np.nan,
                        hit0[~inside].mean() if (~inside).any() else np.nan)

                    # C1
                    q1 = q1_all[ev]
                    hit1 = np.abs(y_ev - p_ev) <= q1
                    fin = np.isfinite(q1)
                    add(("C1", 0), hit1.mean(),
                        float(np.median(2 * q1[fin]) / sd_t) if fin.any() else np.inf,
                        float((~fin).mean()))

                    for k in ks:
                        ck = pool[:k]
                        # C2: yalnızca eşik hedefte yeniden
                        q2 = conf_q(np.abs(yt[ck] - yh_t[ck]), a.alpha)
                        hit2 = np.abs(y_ev - p_ev) <= q2
                        add(("C2", k), hit2.mean(), 2 * q2 / sd_t, float(np.isinf(q2)),
                            hit2[inside].mean() if inside.any() else np.nan,
                            hit2[~inside].mean() if (~inside).any() else np.nan)
                        # C3: afin + LOO conformal
                        p3, loo = affine_loo(yh_t[ck], yt[ck], p_ev)
                        q3 = conf_q(loo, a.alpha)
                        hit3 = np.abs(y_ev - p3) <= q3
                        add(("C3", k), hit3.mean(), 2 * q3 / sd_t, float(np.isinf(q3)),
                            hit3[inside].mean() if inside.any() else np.nan,
                            hit3[~inside].mean() if (~inside).any() else np.nan)

                for (meth, k), d in acc.items():
                    wv = np.asarray(d["w"], float)
                    rows.append(dict(
                        prop=prop, model=model, target=tgt, n_target=nt,
                        method=meth, k=k,
                        coverage=round(float(np.nanmean(d["cov"])), 4),
                        width_sd=round(float(np.nanmedian(wv[np.isfinite(wv)])), 4)
                        if np.isfinite(wv).any() else np.inf,
                        frac_inf=round(float(np.mean(d["inf"])), 4),
                        cov_inside=round(float(np.nanmean(d["cin"])), 4),
                        cov_outside=round(float(np.nanmean(d["cout"])), 4),
                        cov_source_test=round(cov_src, 4),
                        ess_c1=round(ess, 1) if meth == "C1" else np.nan,
                        n_cal_source=len(ca),
                    ))
                c0 = np.mean(acc[("C0", 0)]["cov"])
                c1 = np.mean(acc[("C1", 0)]["cov"])
                k25 = 25 if 25 in ks else ks[0]
                c2 = np.mean(acc[("C2", k25)]["cov"])
                print(f"  {tgt:>18s} {model:>4s}  n={nt:6d}  kaynak-ici {cov_src:.3f}  "
                      f"C0 {c0:.3f}  C1 {c1:.3f} (ESS {ess:.0f})  C2(k={k25}) {c2:.3f}"
                      f"   [{time.time() - t0:.0f} sn]")

    if not rows:
        raise SystemExit("hic hucre kosmadi")
    res = pd.DataFrame(rows)
    tag = f"{a.arm}_{a.track}"
    res.to_csv(os.path.join(a.out, f"asama01_hucreler_{tag}.csv"), index=False)

    # ---------------- özet ve durdurma kuralları ----------------
    summ = (res.groupby(["method", "k"])
               .agg(kapsama_medyan=("coverage", "median"),
                    kapsama_min=("coverage", "min"),
                    genislik_medyan=("width_sd", "median"),
                    sonsuz_oran=("frac_inf", "mean"),
                    kapsama_ici=("cov_inside", "median"),
                    kapsama_disi=("cov_outside", "median"),
                    hucre=("coverage", "size"))
               .round(3))
    summ.to_csv(os.path.join(a.out, f"asama01_ozet_{tag}.csv"))
    byprop = (res[res["method"].isin(["C0", "C1"]) | (res["k"] == 25)]
              .pivot_table(index="prop", columns=["method", "k"],
                           values="coverage", aggfunc="median").round(3))

    print("\n\n" + "=" * 78)
    print(f"OZET — hedef kapsama {1 - a.alpha:.2f}  ({tag}, hucre = ozellik x model x hedef)")
    print("=" * 78)
    print(summ.to_string())
    print("\nozellik bazinda kapsama medyani:")
    print(byprop.to_string())

    src = res.drop_duplicates(["prop", "model", "target"])["cov_source_test"].median()
    c0 = summ.loc[("C0", 0), "kapsama_medyan"]
    c1 = summ.loc[("C1", 0), "kapsama_medyan"]
    c1inf = summ.loc[("C1", 0), "sonsuz_oran"]
    c2 = summ.loc[("C2", 25), "kapsama_medyan"] if ("C2", 25) in summ.index else np.nan

    print("\nDURDURMA KURALLARI (onceden yazildi)")
    print(f"  K0 kaynak-ici kapsama medyani   {src:.3f}   (beklenen 0.87-0.93)")
    print(f"  K1 C0 hedef kapsama medyani     {c0:.3f}   (>= 0.85 ise DUR)")
    print(f"  K2 C1 medyani {c1:.3f} vs C2(k=25) {c2:.3f}, C1 sonsuz orani {c1inf:.3f}")
    if not (0.87 <= src <= 0.93):
        print("\n  -> K0 BASARISIZ: hat kontrolu gecmedi, sonuclari yorumlama.")
    elif c0 >= 0.85:
        print("\n  -> K1: aralıklar kütüphane değişince de sağlam. DUR.")
    elif (c1 >= c2 - 0.02) and (c1inf < 0.20):
        print("\n  -> K2: weighted conformal hedef etiketi kadar iyi. Koşullu kayma "
              "kancası çöktü, çerçeve değişmeli.")
    else:
        print("\n  -> Kurallar geçildi: kapsama bozuluyor, weighted conformal "
              "düzeltemiyor. TAM ANALİZE GEÇ.")
    print(f"\ncikti: {os.path.abspath(a.out)}   ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
