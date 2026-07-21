# -*- coding: utf-8 -*-
"""_single_label_dilution.py — direct control for the multi-label dilution
hypothesis (internal-review item R2-W1; Online Resource 17).

Challenge addressed: surface signals (logprob etc.) reach error-prediction
AUC > 0.7 on single-answer QA yet fail at 0.50-0.53 here — is the cause
(a) dilution by the multi-label output format, or (b) the domain/RAG setting?
Control: how much better does the same signal predict the single-label BINARY
decision error (imaging vs no-imaging) than the multi-label errors
(ct_miss / y_err)?
  - binary AUC >> multi-label AUC  -> supports the dilution hypothesis
    (the signal works when not diluted)
  - binary AUC also at chance      -> not dilution (the failure is deeper;
    the negative result is strengthened)

Data: runs_e0x/{backbone}_deployed_surface.csv (E0X deployed signals,
n=1652-2000). Signals: U_lp (logprob), U_sc (self-consistency), S_mean
(retrieval sufficiency), D_jac (cross-source conflict). Both AUC and
max(AUC, 1-AUC) are reported so |AUC-0.5| comparisons are direction-agnostic.
Binary target: bin_err = (dep__no_imaging != gold_no_imaging), with
gold_no_imaging = (gold_card == 0). AUCs are single-feature (max(auc,1-auc)
alongside, sign-corrected); a 5-fold OOF logistic pool is also reported.
"""
import glob
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold

SIGNALS = ["U_lp", "U_sc", "S_mean", "D_jac"]
TARGETS_ML = ["ct_miss", "y_err"]


def auc_abs(y, s):
    """single-feature AUC; NaN-safe; returns (auc, |auc-0.5|-oriented max(auc,1-auc))."""
    m = ~(pd.isna(s) | pd.isna(y))
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return float("nan"), float("nan")
    a = roc_auc_score(y[m], s[m])
    return a, max(a, 1 - a)


def auc_oof(y, S, seed=42):
    """5-fold OOF multivariate logistic AUC over all signals (deployable pool)."""
    m = ~(pd.isna(y))
    for c in range(S.shape[1]):
        m &= ~pd.isna(S[:, c])
    if m.sum() < 50 or len(np.unique(y[m])) < 2:
        return float("nan")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    try:
        p = cross_val_predict(pipe, S[m], y[m], cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(y[m], p)
    except Exception:
        return float("nan")


coh = pd.read_csv("cohort_v2.csv", usecols=["stay_id", "no_imaging_full", "card_full"])
coh_map = coh.set_index("stay_id")

rows = []
for path in sorted(glob.glob("runs_e0x/*_deployed_surface.csv")):
    bb = os.path.basename(path).replace("_deployed_surface.csv", "")
    df = pd.read_csv(path)
    df = df.join(coh_map, on="stay_id", rsuffix="_coh")
    df = df[df["dep__no_imaging"].notna()].copy()              # drop failed generations
    # gold no-imaging: prefer cohort no_imaging_full; fallback gold_card==0
    if "no_imaging_full" in df:
        gold_ni = df["no_imaging_full"].fillna((df.get("gold_card", 0) == 0).astype(int)).astype(int)
    else:
        gold_ni = (df["gold_card"] == 0).astype(int)
    pred_ni = df["dep__no_imaging"].astype(int)
    bin_err = (pred_ni != gold_ni).astype(int).values          # single-label decision error

    y_ct = df["ct_miss"].astype(float).values
    y_er = df["y_err"].astype(float).values
    Sarr = df[SIGNALS].astype(float).values

    rec = {"backbone": bb, "n": len(df),
           "base_ct_miss": float(np.nanmean(y_ct)), "base_y_err": float(np.nanmean(y_er)),
           "base_bin_err": float(bin_err.mean())}
    # per-signal: multi-label vs single-label |AUC|
    for sig in SIGNALS:
        s = df[sig].astype(float).values
        _, a_ct = auc_abs(y_ct, s)
        _, a_bin = auc_abs(bin_err.astype(float), s)
        rec[f"{sig}|ct_miss"] = round(a_ct, 3)
        rec[f"{sig}|bin"] = round(a_bin, 3)
    # multivariate pool: multi-label vs single-label
    rec["POOL|ct_miss"] = round(auc_oof(y_ct, Sarr), 3)
    rec["POOL|y_err"] = round(auc_oof(y_er, Sarr), 3)
    rec["POOL|bin"] = round(auc_oof(bin_err.astype(float), Sarr), 3)
    rows.append(rec)
    print(f"[done] {bb:16s} n={len(df)} base_bin={rec['base_bin_err']:.3f}  "
          f"POOL ct_miss={rec['POOL|ct_miss']}  y_err={rec['POOL|y_err']}  bin={rec['POOL|bin']}",
          flush=True)

out = pd.DataFrame(rows)
out.to_csv("single_label_dilution.csv", index=False)

print("\n==== multivariate pool |AUC| : multi-label vs single-label(imaging decision) ====")
print(f"{'backbone':16s} {'ct_miss':>8s} {'y_err':>8s} {'bin(single)':>12s}")
for _, r in out.iterrows():
    print(f"{r['backbone']:16s} {r['POOL|ct_miss']:>8} {r['POOL|y_err']:>8} {r['POOL|bin']:>12}")
mlc = out["POOL|ct_miss"].mean(); mle = out["POOL|y_err"].mean(); mb = out["POOL|bin"].mean()
print(f"{'MEAN':16s} {mlc:8.3f} {mle:8.3f} {mb:12.3f}")

print("\n==== per-signal |AUC| (mean over backbones): ct_miss vs single-label ====")
for sig in SIGNALS:
    c = out[f"{sig}|ct_miss"].mean(); b = out[f"{sig}|bin"].mean()
    print(f"  {sig:8s}  ct_miss={c:.3f}   bin(single)={b:.3f}")
print("\nsaved -> single_label_dilution.csv")
