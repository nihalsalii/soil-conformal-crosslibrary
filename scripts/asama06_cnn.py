#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aşama 06 — Üçüncü model ailesi: 1B evrişimli sinir ağı (1D-CNN)
===============================================================
Amaç dar: bulgunun model ailesine bağlı olup olmadığını göstermek. asama02'de
PLSR (doğrusal izdüşüm) ve HGB (ağaç topluluğu) vardı; burada spektrumu
doğrudan işleyen bir derin model ekleniyor (SOIL 2025 çalışması CNN kullandığı
için hakemin soracağı soru budur).

Kapsam (bilinçli olarak sınırlı):
  * alpha = 0.10, k = 25 ve 50
  * yöntemler C0, C0n, C2, C3n  (weighted ve CQR burada tekrarlanmaz)
  * kaynak eğitim kümesi alt örneklenir (--max-train, varsayılan 8000)
  * spektrum --bin kadar bantta ortalanarak indirgenir (varsayılan 4;
    MIR 1701 -> ~425 bant, yaklaşık 8 cm-1; Weerasekara ve ark. 2026 da
    10 cm-1'e indirgemişti)

Önceden belirtilen beklentiler (betik çalıştırılmadan yazıldı):
  G1 CNN ile C0 kapsamasının medyanı nominalin altında (GA üst sınırı < 0.90)
  G2 CNN ile k = 25 nominale ulaşır (GA nominali içerir ya da üstündedir)
  G3 CNN'in C0 medyanı, asama02'deki PLSR/HGB medyanına +/- 0.07 içinde
     (yani sonuç model ailesine bağlı değil)

Kurulum (bir kez):
  pip install torch --index-url https://download.pytorch.org/whl/cpu

Kullanım:
  python scripts\\asama06_cnn.py --arm mir --track clean
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

try:
    import torch
    import torch.nn as nn
except Exception:
    raise SystemExit(
        "PyTorch bulunamadi. Kurulum:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu")

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


# ---------------------------------------------------------------------------
class SpecCNN(nn.Module):
    """Ng ve ark. (2019) hattındaki sade bir 1B evrişimli ağ."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


def train_cnn(Xtr, ytr, seed, epochs, patience, batch, lr, device):
    torch.manual_seed(seed)
    n = len(ytr)
    idx = np.random.default_rng(seed).permutation(n)
    nval = max(200, int(0.1 * n))
    va, tr = idx[:nval], idx[nval:]
    mu, sd = ytr[tr].mean(), ytr[tr].std() or 1.0
    Xt = torch.tensor(Xtr[tr], dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor((ytr[tr] - mu) / sd, dtype=torch.float32)
    Xv = torch.tensor(Xtr[va], dtype=torch.float32).unsqueeze(1).to(device)
    yv = torch.tensor((ytr[va] - mu) / sd, dtype=torch.float32).to(device)

    model = SpecCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), batch):
            b = perm[i:i + batch]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(lossf(model(Xv), yv))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                xb = torch.tensor(X[i:i + 1024], dtype=torch.float32).unsqueeze(1).to(device)
                out.append(model(xb).cpu().numpy())
        return np.concatenate(out) * sd + mu
    return predict


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--arm", default="mir", choices=["mir", "visnir"])
    ap.add_argument("--track", default="clean", choices=["clean", "naive"])
    ap.add_argument("--libcol", default="dataset.code_ascii_txt")
    ap.add_argument("--props", default=",".join(PROPS))
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--ks", default="25,50")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--min-eval", type=int, default=30)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--bin", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 220)
    t0 = time.time()
    ks_all = [int(k) for k in a.ks.split(",")]
    device = torch.device(a.device)

    X = np.load(os.path.join(a.data_dir, f"spectra_{a.arm}.npy"))
    bad = ~np.isfinite(X).all(axis=0)
    if bad.any():
        print(f"  {int(bad.sum())} bant NaN iceriyor, cikariliyor")
        X = X[:, ~bad]
    meta = pd.read_csv(os.path.join(a.data_dir, f"meta_{a.arm}.csv"), low_memory=False)
    lib_all = meta[a.libcol].astype(str).to_numpy()
    rng = np.random.default_rng(a.seed)
    Xb = binned(snv(X), a.bin).astype(np.float32)
    print(f"{a.arm}/{a.track}: {X.shape[0]} spektrum, {X.shape[1]} -> {Xb.shape[1]} bant "
          f"(bin={a.bin}) | CNN, cihaz={a.device}, max-train={a.max_train}")

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

            pred = train_cnn(Xb[tr], y_all[tr], a.seed, a.epochs, a.patience,
                             a.batch, a.lr, device)
            yh_ca, yh_te, yh_di, yh_t = (pred(Xb[ca]), pred(Xb[te]),
                                         pred(Xb[di]), pred(Xt))

            # zorluk modeli (sigma): kaynak PCA(20) uzerinde HGB
            pca = PCA(n_components=20, random_state=a.seed).fit(Xb[tr])
            sig_m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                  random_state=a.seed).fit(
                pca.transform(Xb[di]), np.abs(y_all[di] - yh_di))
            floor = max(float(np.percentile(np.abs(y_all[di] - yh_di), 5)), 1e-6)
            s_ca = np.maximum(sig_m.predict(pca.transform(Xb[ca])), floor)
            s_te = np.maximum(sig_m.predict(pca.transform(Xb[te])), floor)
            s_t = np.maximum(sig_m.predict(pca.transform(Xt)), floor)

            sc = np.abs(y_all[ca] - yh_ca)
            q0, q0n = conf_q(sc, a.alpha), conf_q(sc / s_ca, a.alpha)
            cov_src = float(np.mean(np.abs(y_all[te] - yh_te) <= q0))
            rmse_std_src = float(np.sqrt(np.mean((y_all[te] - yh_te) ** 2)) / y_all[te].std())
            rmse_std_tgt = float(np.sqrt(np.mean((yt - yh_t) ** 2)) / sd_t)

            acc = {}

            def add(key, hit, width):
                d = acc.setdefault(key, dict(cov=[], w=[]))
                d["cov"].append(float(np.mean(hit)))
                d["w"].append(float(np.median(width) / sd_t))

            for _ in range(a.reps):
                perm = rng.permutation(nt)
                pool, ev = perm[:kmax], perm[kmax:]
                y_ev, p_ev, sg = yt[ev], yh_t[ev], s_t[ev]
                ae = np.abs(y_ev - p_ev)
                add(("C0", 0), ae <= q0, np.full(len(ev), 2 * q0))
                add(("C0n", 0), ae <= q0n * sg, 2 * q0n * sg)
                for k in ks:
                    ck = pool[:k]
                    q2 = conf_q(np.abs(yt[ck] - yh_t[ck]), a.alpha)
                    add(("C2", k), ae <= q2, np.full(len(ev), 2 * q2))
                    p3, loo = affine_loo(yh_t[ck], yt[ck], p_ev)
                    q3n = conf_q(np.abs(loo) / s_t[ck], a.alpha)
                    add(("C3n", k), np.abs(y_ev - p3) <= q3n * sg, 2 * q3n * sg)

            for (meth, k), d in acc.items():
                rows.append(dict(prop=prop, model="cnn", target=tgt, n_target=nt,
                                 method=meth, k=k,
                                 coverage=float(np.mean(d["cov"])),
                                 width_sd=float(np.median(d["w"])),
                                 cov_source_test=cov_src,
                                 rmse_std_source=rmse_std_src,
                                 rmse_std_target=rmse_std_tgt,
                                 n_train=len(tr)))
            print(f"  {tgt:>18s} n={nt:6d} | kaynak-ici {cov_src:.3f} "
                  f"(rmse/std {rmse_std_src:.3f}) | hedef rmse/std {rmse_std_tgt:.3f} | "
                  f"C0 {np.mean(acc[('C0', 0)]['cov']):.3f} "
                  f"C0n {np.mean(acc[('C0n', 0)]['cov']):.3f} "
                  f"C2(25) {np.mean(acc[('C2', ks[0])]['cov']):.3f} "
                  f"C3n(25) {np.mean(acc[('C3n', ks[0])]['cov']):.3f}  [{time.time() - t0:.0f} sn]")

    if not rows:
        raise SystemExit("hic hucre kosmadi")
    res = pd.DataFrame(rows)
    tag = f"{a.arm}_{a.track}"
    res.to_csv(os.path.join(a.out, f"asama06_hucreler_cnn_{tag}.csv"), index=False)

    brng = np.random.default_rng(a.seed)
    out = []
    for (meth, k), d in res.groupby(["method", "k"]):
        lo, hi = boot_median_ci(d["coverage"], brng)
        out.append(dict(method=meth, k=k, cells=len(d),
                        coverage=d["coverage"].median(), ci_low=lo, ci_high=hi,
                        worst=d["coverage"].min(),
                        width=d["width_sd"].median()))
    summ = pd.DataFrame(out).round(3)
    summ.to_csv(os.path.join(a.out, f"asama06_ozet_cnn_{tag}.csv"), index=False)

    print("\n\n" + "=" * 84)
    print(f"OZET — 1D-CNN, hedef kapsama {1 - a.alpha:.2f} ({tag})")
    print("=" * 84)
    print(summ.to_string(index=False))
    src = res.drop_duplicates(["prop", "target"])["cov_source_test"].median()
    print(f"\nkaynak-ici kapsama medyani: {src:.3f}   "
          f"kaynak-ici rmse/std medyani: "
          f"{res.drop_duplicates(['prop', 'target'])['rmse_std_source'].median():.3f}")

    S = summ.set_index(["method", "k"])
    print("\nONCEDEN BELIRTILEN BEKLENTILER")
    g1 = S.loc[("C0", 0), "ci_high"] < 1 - a.alpha
    print(f"  G1 C0 {S.loc[('C0', 0), 'coverage']:.3f} "
          f"[{S.loc[('C0', 0), 'ci_low']:.3f}, {S.loc[('C0', 0), 'ci_high']:.3f}] "
          f"nominalin altinda? {'EVET' if g1 else 'HAYIR'}")
    k1 = ks_all[0]
    g2 = S.loc[("C2", k1), "ci_high"] >= 1 - a.alpha
    print(f"  G2 C2(k={k1}) {S.loc[('C2', k1), 'coverage']:.3f} "
          f"[{S.loc[('C2', k1), 'ci_low']:.3f}, {S.loc[('C2', k1), 'ci_high']:.3f}] "
          f"nominale ulasiyor? {'EVET' if g2 else 'HAYIR'}")
    print("  G3 asama02 ozetindeki PLSR/HGB C0 medyani ile karsilastir "
          "(mir_clean 0.754, visnir_naive 0.848): fark +/- 0.07 icinde mi?")
    print(f"     CNN C0 medyani {S.loc[('C0', 0), 'coverage']:.3f}")
    print(f"\ncikti: {os.path.abspath(a.out)}   ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
