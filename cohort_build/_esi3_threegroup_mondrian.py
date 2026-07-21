# -*- coding: utf-8 -*-
"""_esi3_threegroup_mondrian.py — three-group Mondrian CRC isolating ESI-3
(Online Resource 19).

Motivation: the two-group Mondrian design (high = ESI 1-2, alpha=0.05 / low =
ESI 3-5, alpha=0.15) merges ESI 3 — the largest-volume, diagnostically hardest
ED stratum — with the mild ESI 4-5 patients under the loosest budget.

Cohort ESI distribution (key fact): ESI1=390, ESI2=2860, ESI3=4515, ESI4=58,
ESI5=0 => ESI 4-5 is only 58 stays (0.7%), so the pooled "low" group is
effectively ESI-3 itself.
Three groups: high{1,2} alpha=0.05 / mid{3} alpha=0.10 (tighter than the
two-group 0.15) / low{4,5} alpha=0.15. Each group is calibrated independently
via select_lambda = the e0_crc.py Mondrian extended to three groups.
Key questions: does isolating ESI-3 under the tighter 0.10 keep usable commit
with joint <= budget; and is low (ESI 4-5, n~58) certifiable at all (data
limit expected)?
Primary qwen14b 30 runs + 8-backbone summary. select_lambda / hb_pvalue / _kl
copied verbatim from e0_crc.py.
"""
import glob
import math
import os
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

W = "full"
LAYER_POS = 3
N_PCA = 50
DELTA = 0.10
SPLIT_SEEDS = list(range(42, 52))
GROUPS = {"high(ESI1-2)": ({1, 2}, 0.05),
          "mid(ESI3)":    ({3}, 0.10),
          "low(ESI4-5)":  ({4, 5}, 0.15)}


def _kl(a, b):
    a = min(max(a, 1e-12), 1 - 1e-12); b = min(max(b, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hb_pvalue(rhat, n, alpha):
    if n <= 0 or rhat >= alpha:
        return 1.0
    hoeff = math.exp(-n * _kl(rhat, alpha))
    try:
        from scipy.stats import binom
        bent = math.e * float(binom.cdf(math.ceil(n * rhat), n, alpha))
    except Exception:
        bent = hoeff
    return float(min(hoeff, bent))


def select_lambda(s_cal, loss_cal, alpha, delta):
    lam = None
    for c in np.sort(np.unique(s_cal)):
        rhat = float(np.mean(loss_cal * (s_cal <= c)))
        if hb_pvalue(rhat, len(s_cal), alpha) <= delta:
            lam = float(c)
        else:
            break
    return lam


def fit_scores(Xtr, ytr, Xev, seed):
    n_pca = int(min(N_PCA, Xtr.shape[1], Xtr.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xev)[:, 1]


demo = pd.read_csv("cohort_v2.csv", usecols=["stay_id", "triage"])
tmap = demo.set_index("stay_id")["triage"].to_dict()
gold = pd.read_csv("cohort_v2.csv", usecols=lambda c: c == "stay_id" or c.startswith(f"y_{W}__"))
gmap = {int(r["stay_id"]): r for r in gold.to_dict("records")}


def load_backbone(path):
    z = np.load(path, allow_pickle=True)
    stay_ids, valid, pred_bits = z["stay_ids"], z["valid"], z["pred_bits"]
    LABELS = [str(x) for x in z["labels"]]
    CT = [L for L in LABELS if L.startswith("CT_")]
    last = z["last"]
    keep, l_ct, esi = [], [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        g = gmap.get(int(sid)); t = tmap.get(int(sid))
        if g is None or t is None:
            continue
        gset = {L for L in LABELS if int(g.get(f"y_{W}__{L}", 0) or 0) == 1}
        pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j]) == 1}
        ctg = {L for L in CT if L in gset}
        keep.append(i); l_ct.append(int(len(ctg - pred) > 0) if ctg else 0); esi.append(int(t))
    keep = np.array(keep)
    return last[keep][:, LAYER_POS, :].astype(np.float32), np.array(l_ct), np.array(esi)


def run_backbone(path):
    X, loss, esi = load_backbone(path)
    acc = {g: [] for g in GROUPS}
    for sp in SPLIT_SEEDS:
        rng = np.random.default_rng(sp)
        idx = rng.permutation(len(loss))
        n = len(idx); ntr, nca = int(0.4 * n), int(0.3 * n)
        tr, ca, te = idx[:ntr], idx[ntr:ntr + nca], idx[ntr + nca:]
        s_ca = fit_scores(X[tr], loss[tr], X[ca], sp)
        s_te = fit_scores(X[tr], loss[tr], X[te], sp)
        for gname, (esis, alpha) in GROUPS.items():
            cam = np.isin(esi[ca], list(esis))
            tem = np.isin(esi[te], list(esis))
            if cam.sum() < 10 or tem.sum() < 5:
                acc[gname].append(dict(n=int(tem.sum()), alpha=alpha, base=float('nan'),
                                       commit=float('nan'), joint=float('nan'),
                                       cond=float('nan'), certified=0, degenerate=1))
                continue
            lam = select_lambda(s_ca[cam], loss[ca][cam], alpha, DELTA)
            if lam is None:
                acc[gname].append(dict(n=int(tem.sum()), alpha=alpha, base=float(loss[te][tem].mean()),
                                       commit=0.0, joint=0.0, cond=0.0, certified=0, degenerate=0))
                continue
            c = s_te[tem] <= lam
            lo = loss[te][tem]
            acc[gname].append(dict(n=int(tem.sum()), alpha=alpha, base=float(lo.mean()),
                                   commit=float(c.mean()), joint=float(np.mean(lo * c)),
                                   cond=float(np.mean(lo[c])) if c.any() else 0.0,
                                   certified=int(np.mean(lo * c) <= alpha), degenerate=0))
    rows = []
    for gname, recs in acc.items():
        df = pd.DataFrame(recs)
        rows.append(dict(group=gname, alpha=GROUPS[gname][1], n=int(df.n.mean()),
                         base=df.base.mean(), commit=df.commit.mean(),
                         joint=df.joint.mean(), cond=df.cond.mean(),
                         cert_frac=df.certified.mean(), degen=int(df.degenerate.sum())))
    return pd.DataFrame(rows)


print(f"=== ESI-3 THREE-GROUP MONDRIAN (primary=qwen14b, {len(SPLIT_SEEDS)} split-seeds) ===")
print("cohort ESI: 1=390 2=2860 3=4515 4=58 5=0  => low(ESI4-5) is tiny (n~58 total)")
dfp = run_backbone("runs_main/probe_qwen14b_n2000_seed42.npz")
print(f"{'group':13s} {'alpha':>5s} {'n':>4s} {'base':>6s} {'commit':>7s} {'joint':>7s} "
      f"{'cond':>6s} {'cert_frac':>9s}")
for _, r in dfp.iterrows():
    print(f"{r.group:13s} {r.alpha:5.2f} {r.n:4d} {r.base:6.3f} {r.commit:7.3f} "
          f"{r.joint:7.3f} {r.cond:6.3f} {r.cert_frac:9.2f}")
dfp.insert(0, "backbone", "qwen14b")
dfp.to_csv("esi3_threegroup_qwen14b.csv", index=False)

print(f"\n=== 8-BACKBONE: per-group mean joint (<=alpha?) and commit ===")
print(f"{'backbone':16s} {'high joint':>10s} {'mid joint':>9s} {'low joint':>9s} "
      f"| {'high com':>8s} {'mid com':>7s} {'low com':>7s}")
allr = []
for path in sorted(glob.glob("runs_main/probe_*_n2000_seed42.npz")):
    bb = re.match(r"probe_(.+)_n2000_seed", os.path.basename(path)).group(1)
    df = run_backbone(path); df["backbone"] = bb; allr.append(df)
    gj = {r.group.split("(")[0]: r for _, r in df.iterrows()}
    print(f"{bb:16s} {gj['high'].joint:10.3f} {gj['mid'].joint:9.3f} {gj['low'].joint:9.3f} "
          f"| {gj['high'].commit:8.3f} {gj['mid'].commit:7.3f} {gj['low'].commit:7.3f}")
pd.concat(allr).to_csv("esi3_threegroup_8backbone.csv", index=False)
print("\nsaved -> esi3_threegroup_qwen14b.csv, esi3_threegroup_8backbone.csv")
print("NOTE: low(ESI4-5) n~17/test-split -> calibration-degenerate; interpret mid(ESI3) as the isolated group.")
