# -*- coding: utf-8 -*-
"""_delta_sweep.py -- delta-sensitivity sweep (no GPU, existing npz only).

Replicates the E-guar procedure of e0_crc.run_real EXACTLY:
  probe = last-token, layer idx3 (rel-depth 0.875), StandardScaler->PCA(50)->Logistic,
  fixed function trained on train(40%), lambda_hat selected on calib(30%) via
  Hoeffding-Bentkus + fixed-sequence testing, evaluated on test(30%).
Fixed alpha=0.10; sweep delta in {0.01,0.05,0.10,0.20}. 3 sub-seeds x 10 split-seeds
= 30 runs per (model, delta). Validation: the delta=0.10 column must reproduce
results_main (Qwen joint~0.079, commit~0.429).

Run: python _delta_sweep.py [model ...]
"""
import sys, math, csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_main"
COHORT = HERE / "cohort_v2.csv"
IDX3 = 3
SUBSEEDS = [42, 43, 44]
SPLITSEEDS = list(range(42, 52))
ALPHA = 0.10
DELTAS = [0.01, 0.05, 0.10, 0.20]
MODELS = ["qwen14b", "llama31-8b", "gemma2-9b", "mistral7b-v03",
          "med42-8b", "openbio-8b", "phi3mini-128k", "mediphi"]

GOLD = pd.read_csv(COHORT, low_memory=False)


def _kl(a, b):
    a = min(max(a, 1e-12), 1 - 1e-12); b = min(max(b, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hb_pvalue(rhat, n, alpha):
    if n <= 0 or rhat >= alpha:
        return 1.0
    hoeff = math.exp(-n * _kl(rhat, alpha))
    from scipy.stats import binom
    bent = math.e * float(binom.cdf(math.ceil(n * rhat), n, alpha))
    return float(min(hoeff, bent))


def select_lambda(s_cal, loss_cal, alpha, delta):
    n = len(s_cal); lam_hat, risk_at = None, 0.0
    for lam in np.sort(np.unique(s_cal)):
        rhat = float(np.mean(loss_cal * (s_cal <= lam)))
        if hb_pvalue(rhat, n, alpha) <= delta:
            lam_hat, risk_at = float(lam), rhat
        else:
            break
    return lam_hat, risk_at


def ctmiss(stay_ids, valid, pred_bits, labels):
    LAB = list(labels); CT = [L for L in LAB if L.startswith("CT_")]
    cols = ["stay_id"] + [f"y_full__{L}" for L in LAB if f"y_full__{L}" in GOLD.columns]
    gmap = {int(r["stay_id"]): r for r in GOLD[cols].to_dict("records")}
    keep, y = [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        g = gmap.get(int(sid))
        if g is None:
            continue
        gold = {L for L in LAB if int(g.get(f"y_full__{L}", 0) or 0) == 1}
        pred = {LAB[j] for j in range(len(LAB)) if int(pred_bits[i][j]) == 1}
        ctg = {L for L in CT if L in gold}
        keep.append(i); y.append(int(len(ctg - pred) > 0) if ctg else 0)
    return np.array(keep), np.array(y, int)


def fit_scores(Xtr, ytr, Xev, seed):
    npca = int(min(50, Xtr.shape[1], Xtr.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=npca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xev)[:, 1]


def run_model(m):
    out = {d: [] for d in DELTAS}
    for sd in SUBSEEDS:
        f = RUNS / f"probe_{m}_n2000_seed{sd}.npz"
        if not f.exists():
            print(f"  [skip] {f.name}"); continue
        d = np.load(f, allow_pickle=True)
        keep, y = ctmiss(d["stay_ids"], d["valid"], d["pred_bits"], d["labels"])
        X = d["last"][keep][:, IDX3, :].astype(np.float32)
        loss = y.astype(float)
        for ss in SPLITSEEDS:
            rng = np.random.default_rng(ss)
            idx = rng.permutation(len(keep))
            n = len(idx); ntr = int(0.4 * n); nca = int(0.3 * n)
            tr, ca, te = idx[:ntr], idx[ntr:ntr + nca], idx[ntr + nca:]
            s_ca = fit_scores(X[tr], y[tr], X[ca], ss)
            s_te = fit_scores(X[tr], y[tr], X[te], ss)
            for dl in DELTAS:
                lam, _ = select_lambda(s_ca, loss[ca], ALPHA, dl)
                if lam is None:
                    out[dl].append((0.0, 0.0)); continue
                commit = (s_te <= lam)
                jr = float(np.mean(loss[te] * commit))
                out[dl].append((jr, float(commit.mean())))
    return out


if __name__ == "__main__":
    targets = sys.argv[1:] or MODELS
    rows = []
    print(f"alpha fixed = {ALPHA};  30 runs per cell (3 sub x 10 split)\n")
    print(f"{'model':15s} {'delta':>6s} {'joint_risk':>18s} {'commit':>16s} {'run-OK%':>8s}")
    for m in targets:
        res = run_model(m)
        for dl in DELTAS:
            arr = np.array(res[dl])
            jr, cm = arr[:, 0], arr[:, 1]
            okp = 100 * np.mean(jr <= ALPHA)
            jrci = 1.96 * jr.std(ddof=1) / math.sqrt(len(jr))
            cmci = 1.96 * cm.std(ddof=1) / math.sqrt(len(cm))
            print(f"{m:15s} {dl:>6.2f} {jr.mean():>10.4f}+-{jrci:.4f} "
                  f"{cm.mean():>8.4f}+-{cmci:.4f} {okp:>7.1f}")
            rows.append([m, dl, f"{jr.mean():.4f}", f"{jrci:.4f}",
                         f"{cm.mean():.4f}", f"{cmci:.4f}", f"{okp:.1f}"])
        print()
    with open(HERE / "delta_sweep_results.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["model", "delta", "joint_risk", "joint_ci", "commit", "commit_ci", "run_ok_pct"])
        w.writerows(rows)
    print(f"[saved] {HERE / 'delta_sweep_results.csv'}")
