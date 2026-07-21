# -*- coding: utf-8 -*-
"""_random_score_crc_baseline.py — random-score CRC deployment baseline
(internal-review item DA-M1; Online Resource 16).

Challenge addressed: "commit > 0 is not evidence of non-vacuity — at base rate
b even a random score gets certified at commit ~ alpha/b minus a finite-sample
margin; the utility evidence is gap-closed alone." This script measures that
claim directly, testing whether the probe commits substantially more than the
random baseline at the same certified joint risk (i.e., whether the probe
selectively commits the easy cases).

Procedure (identical to e0_crc.py): MAIN random split (tr 40 / cal 30 / te 30
per split seed), ct_miss loss, alpha=delta=0.10. probe = last-token layer_idx=3
trained then scored on cal/te; random = U(0,1) iid score per case. Both go
through select_lambda(cal) and commit_rate is measured on te.
select_lambda / hb_pvalue / _kl copied verbatim from e0_crc.py.
Runs: 8 backbones x sub{42,43,44} x split{42..51} = 240.
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
ALPHA = DELTA = 0.10
SPLIT_SEEDS = list(range(42, 52))


# ---- CRC functions (copied verbatim from e0_crc.py) ----
def _kl(a, b):
    a = min(max(a, 1e-12), 1 - 1e-12)
    b = min(max(b, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hb_pvalue(rhat, n, alpha):
    if n <= 0:
        return 1.0
    if rhat >= alpha:
        return 1.0
    hoeff = math.exp(-n * _kl(rhat, alpha))
    try:
        from scipy.stats import binom
        bent = math.e * float(binom.cdf(math.ceil(n * rhat), n, alpha))
    except Exception:
        bent = hoeff
    return float(min(hoeff, bent))


def select_lambda(s_cal, loss_cal, alpha, delta):
    n = len(s_cal)
    lam_hat = None
    for lam in np.sort(np.unique(s_cal)):
        committed = s_cal <= lam
        rhat = float(np.mean(loss_cal * committed))
        if hb_pvalue(rhat, n, alpha) <= delta:
            lam_hat = float(lam)
        else:
            break
    return lam_hat


def fit_probe_scores(Xtr, ytr, Xev, seed):
    n_pca = int(min(N_PCA, Xtr.shape[1], Xtr.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xev)[:, 1]


coh = pd.read_csv("cohort_v2.csv",
                  usecols=lambda c: c == "stay_id" or c.startswith(f"y_{W}__"))
gmap = {int(r["stay_id"]): r for r in coh.to_dict("records")}

rows = []
for path in sorted(glob.glob("runs_main/probe_*_n2000_seed*.npz")):
    m = re.match(r"probe_(.+)_n2000_seed(\d+)\.npz", os.path.basename(path))
    bb, sub = m.group(1), int(m.group(2))
    z = np.load(path, allow_pickle=True)
    stay_ids, valid, pred_bits = z["stay_ids"], z["valid"], z["pred_bits"]
    LABELS = [str(x) for x in z["labels"]]
    CT = [L for L in LABELS if L.startswith("CT_")]
    last = z["last"]

    keep, l_ct = [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        g = gmap.get(int(sid))
        if g is None:
            continue
        gold = {L for L in LABELS if int(g.get(f"y_{W}__{L}", 0) or 0) == 1}
        pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j]) == 1}
        ctg = {L for L in CT if L in gold}
        keep.append(i)
        l_ct.append(int(len(ctg - pred) > 0) if ctg else 0)
    keep = np.array(keep)
    loss = np.array(l_ct)
    X = last[keep][:, LAYER_POS, :].astype(np.float32)
    base = float(loss.mean())

    for sp in SPLIT_SEEDS:
        rng = np.random.default_rng(sp)
        idx = rng.permutation(len(keep))
        n = len(idx)
        ntr, nca = int(0.4 * n), int(0.3 * n)
        tr, ca, te = idx[:ntr], idx[ntr:ntr + nca], idx[ntr + nca:]

        # probe
        s_ca = fit_probe_scores(X[tr], loss[tr], X[ca], sp)
        s_te = fit_probe_scores(X[tr], loss[tr], X[te], sp)
        lam_p = select_lambda(s_ca, loss[ca], ALPHA, DELTA)
        if lam_p is None:
            p_commit, p_joint = 0.0, 0.0
        else:
            c = s_te <= lam_p
            p_commit, p_joint = float(c.mean()), float(np.mean(loss[te] * c))

        # random score
        r_ca = rng.random(len(ca))
        r_te = rng.random(len(te))
        lam_r = select_lambda(r_ca, loss[ca], ALPHA, DELTA)
        if lam_r is None:
            r_commit, r_joint = 0.0, 0.0
        else:
            c = r_te <= lam_r
            r_commit, r_joint = float(c.mean()), float(np.mean(loss[te] * c))

        rows.append(dict(backbone=bb, sub=sub, seed=sp, base=base,
                         probe_commit=p_commit, probe_joint=p_joint,
                         rand_commit=r_commit, rand_joint=r_joint,
                         alpha_over_base=min(1.0, ALPHA / base) if base > 0 else float("nan")))
    r = [x for x in rows if x["backbone"] == bb and x["sub"] == sub]
    print(f"[done] {bb} sub{sub}: base={base:.3f} "
          f"probe_commit={np.mean([x['probe_commit'] for x in r]):.3f} "
          f"rand_commit={np.mean([x['rand_commit'] for x in r]):.3f} "
          f"a/base={r[0]['alpha_over_base']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("random_score_crc_baseline.csv", index=False)
print("\n==== per-backbone (mean over 30 runs) : probe vs random commit @ joint<=0.10 ====")
print(f"{'backbone':16s} {'base':>6s} {'probe_commit':>12s} {'rand_commit':>11s} "
      f"{'a/base(pred)':>12s} {'probe/rand':>10s}")
for bb, g in df.groupby("backbone"):
    pc, rc = g.probe_commit.mean(), g.rand_commit.mean()
    print(f"{bb:16s} {g.base.mean():6.3f} {pc:12.3f} {rc:11.3f} "
          f"{g.alpha_over_base.mean():12.3f} {pc/rc if rc>1e-6 else float('inf'):10.2f}")
print("\nsaved -> random_score_crc_baseline.csv")
