# -*- coding: utf-8 -*-
"""_matched_pair.py -- second matched-pair replication (no GPU, existing npz).

Contribution-3 rests on ONE identical-input matched pair (Phi-3-mini vs MediPhi:
probes 0.514 vs 0.743). This adds the Llama-3 pair (Llama-3.1-8B general vs
Med42-8B medical, SAME Llama-3.1-8B base) to test whether the medical-tuning
effect on last-token error-readability replicates.

For each model: OOF CT-miss probe AUC at deployed last-token idx3 AND at the
prompt position idx3 (relocation diagnostic), 3-seed mean +- 95% CI.
Per pair: paired bootstrap Delta-AUC (medical - general) on the SHARED stay_ids
(identical inputs) -> is medical's self-error-readability distinguishably higher?

Run: python _matched_pair.py
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_main"
COHORT = HERE / "cohort_v2.csv"
IDX3 = 3
SEEDS = [42, 43, 44]
PAIRS = [("Phi (reference)", "phi3mini-128k", "mediphi"),
         ("Llama-3 (new)", "llama31-8b", "med42-8b")]
GOLD = pd.read_csv(COHORT, low_memory=False)


def ctmiss(stay_ids, valid, pred_bits, labels):
    LAB = list(labels); CT = [L for L in LAB if L.startswith("CT_")]
    cols = ["stay_id"] + [f"y_full__{L}" for L in LAB if f"y_full__{L}" in GOLD.columns]
    gmap = {int(r["stay_id"]): r for r in GOLD[cols].to_dict("records")}
    keep, y, sids = [], [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        g = gmap.get(int(sid))
        if g is None:
            continue
        gold = {L for L in LAB if int(g.get(f"y_full__{L}", 0) or 0) == 1}
        pred = {LAB[j] for j in range(len(LAB)) if int(pred_bits[i][j]) == 1}
        ctg = {L for L in CT if L in gold}
        keep.append(i); y.append(int(len(ctg - pred) > 0) if ctg else 0); sids.append(int(sid))
    return np.array(keep), np.array(y, int), np.array(sids)


def oof(X, y, seed):
    if len(np.unique(y)) < 2:
        return None
    npca = int(min(50, X.shape[1], X.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=npca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def model_auc(m):
    """3-seed OOF AUC at last@idx3 and prompt@idx3; also return seed42 scores/y/sids."""
    last_a, prompt_a, base = [], [], []
    ref = None
    for sd in SEEDS:
        d = np.load(RUNS / f"probe_{m}_n2000_seed{sd}.npz", allow_pickle=True)
        keep, y, sids = ctmiss(d["stay_ids"], d["valid"], d["pred_bits"], d["labels"])
        Xl = d["last"][keep][:, IDX3, :].astype(np.float32)
        Xp = d["prompt"][keep][:, IDX3, :].astype(np.float32)
        # fold seed fixed at 42 for ALL sub-seeds, matching _recompute_lasttoken_auc.py
        # (the canonical Table 4 protocol) so numbers reconcile exactly.
        sl = oof(Xl, y, 42); sp = oof(Xp, y, 42)
        last_a.append(roc_auc_score(y, sl)); prompt_a.append(roc_auc_score(y, sp))
        base.append(y.mean())
        if sd == 42:
            ref = dict(sl=sl, y=y, sids=sids)
    def ms(v):
        v = np.array(v); h = 2.262 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
        return v.mean(), h
    return ms(last_a), ms(prompt_a), ms(base), ref


def paired_delta(refG, refM, nboot=2000):
    """Paired bootstrap Delta-AUC (medical - general) on shared stay_ids (seed42)."""
    gm = {s: (sc, yy) for s, sc, yy in zip(refG["sids"], refG["sl"], refG["y"])}
    mm = {s: (sc, yy) for s, sc, yy in zip(refM["sids"], refM["sl"], refM["y"])}
    shared = np.array(sorted(set(gm) & set(mm)))
    gsc = np.array([gm[s][0] for s in shared]); gy = np.array([gm[s][1] for s in shared])
    msc = np.array([mm[s][0] for s in shared]); my = np.array([mm[s][1] for s in shared])
    a_g = roc_auc_score(gy, gsc); a_m = roc_auc_score(my, msc)
    rng = np.random.default_rng(42); n = len(shared); deltas = []
    for _ in range(nboot):
        b = rng.integers(0, n, n)
        try:
            deltas.append(roc_auc_score(my[b], msc[b]) - roc_auc_score(gy[b], gsc[b]))
        except ValueError:
            continue
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return a_g, a_m, a_m - a_g, lo, hi, len(shared)


if __name__ == "__main__":
    print("CT-miss OOF probe AUC (3-seed mean +- 95% CI); idx3 = rel-depth 0.875\n")
    print(f"{'model':16s} {'role':8s} {'last@idx3':>16s} {'prompt@idx3':>16s} {'ctmiss base':>12s}")
    cache = {}
    for _, gen, med in PAIRS:
        for m, role in [(gen, "general"), (med, "medical")]:
            (lm, lh), (pm, ph), (bm, bh), ref = model_auc(m)
            cache[m] = ref
            print(f"{m:16s} {role:8s} {lm:.3f}+-{lh:.3f}   {pm:.3f}+-{ph:.3f}   {bm:.3f}")
    print("\nPaired Delta-AUC (medical - general) on SHARED stay_ids (identical inputs, seed42):")
    for name, gen, med in PAIRS:
        a_g, a_m, dd, lo, hi, n = paired_delta(cache[gen], cache[med])
        sig = "distinguishable (CI excl. 0)" if lo > 0 else "NOT distinguishable (CI incl. 0)"
        print(f"  {name:16s} n_shared={n}  general={a_g:.3f}  medical={a_m:.3f}  "
              f"dAUC=+{dd:.3f} [95% {lo:+.3f},{hi:+.3f}] -> {sig}")
