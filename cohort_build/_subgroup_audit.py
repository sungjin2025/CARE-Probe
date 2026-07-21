# -*- coding: utf-8 -*-
"""_subgroup_audit.py — demographic subgroup equity audit of the global policy
(an audit, NOT per-group certification;
Online Resource 18).

Motivation: the practice-based gold label encodes documented demographic
imaging disparities, and the probe is trained on that label, so disparities
could be reproduced in the deferral policy. Demographic conditioning had been
future work only — but MIMIC-IV carries age/sex/race variables, so a
descriptive subgroup audit is possible on existing data.

Procedure: calibrate GLOBAL CRC on the full calibration set exactly as
deployed (single lambda_hat), apply that policy to the test set, then
decompose {n, base, commit, joint, cond} per subgroup — i.e., an equity audit
of "how the global policy lands on each subgroup" (not per-group
certification). Axes: age (<65 / 65+), sex (Female/Male),
race (White/Black/Hispanic/Other).
Protocol: MAIN random split (tr40/cal30/te30), ct_miss, alpha=delta=0.10,
layer_idx=3. Primary backbone = qwen14b averaged over 3 sub-seeds x 10
split-seeds = 30 runs; an 8-backbone summary is reported alongside.
select_lambda / hb_pvalue / _kl copied from e0_crc.py.
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


def race_grp(r):
    r = str(r)
    if r == "White":
        return "White"
    if r == "Black":
        return "Black"
    if r.startswith("Hispanic"):
        return "Hispanic"
    return "Other"


demo = pd.read_csv("cohort_v2.csv", usecols=["stay_id", "age", "gender", "race"])
demo["age_grp"] = np.where(demo.age >= 65, "65+", "<65")
demo["race_grp"] = demo.race.map(race_grp)
dmap = demo.set_index("stay_id")

gold = pd.read_csv("cohort_v2.csv", usecols=lambda c: c == "stay_id" or c.startswith(f"y_{W}__"))
gmap = {int(r["stay_id"]): r for r in gold.to_dict("records")}

AXES = {"age_grp": ["<65", "65+"], "gender": ["Female", "Male"],
        "race_grp": ["White", "Black", "Hispanic", "Other"]}


def load_backbone(path):
    z = np.load(path, allow_pickle=True)
    stay_ids, valid, pred_bits = z["stay_ids"], z["valid"], z["pred_bits"]
    LABELS = [str(x) for x in z["labels"]]
    CT = [L for L in LABELS if L.startswith("CT_")]
    last = z["last"]
    keep, l_ct, sids = [], [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        g = gmap.get(int(sid))
        if g is None or int(sid) not in dmap.index:
            continue
        gset = {L for L in LABELS if int(g.get(f"y_{W}__{L}", 0) or 0) == 1}
        pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j]) == 1}
        ctg = {L for L in CT if L in gset}
        keep.append(i); l_ct.append(int(len(ctg - pred) > 0) if ctg else 0); sids.append(int(sid))
    keep = np.array(keep)
    return last[keep][:, LAYER_POS, :].astype(np.float32), np.array(l_ct), np.array(sids)


def audit_backbone(path, detail=False):
    X, loss, sids = load_backbone(path)
    sub = pd.DataFrame({"stay_id": sids}).join(dmap, on="stay_id")
    acc = {}  # (axis,grp) -> list of dicts per run
    for ssub in [42, 43, 44]:
        pass
    runs = 0
    for sp in SPLIT_SEEDS:
        rng = np.random.default_rng(sp)
        idx = rng.permutation(len(loss))
        n = len(idx); ntr, nca = int(0.4 * n), int(0.3 * n)
        tr, ca, te = idx[:ntr], idx[ntr:ntr + nca], idx[ntr + nca:]
        s_ca = fit_scores(X[tr], loss[tr], X[ca], sp)
        s_te = fit_scores(X[tr], loss[tr], X[te], sp)
        lam = select_lambda(s_ca, loss[ca], ALPHA, DELTA)
        if lam is None:
            continue
        runs += 1
        commit = s_te <= lam
        for axis, groups in AXES.items():
            col = sub.iloc[te][axis].values
            for grp in groups:
                m = (col == grp)
                if m.sum() < 10:
                    continue
                cm = commit[m]
                lo = loss[te][m]
                rec = dict(n=int(m.sum()), base=float(lo.mean()),
                           commit=float(cm.mean()),
                           joint=float(np.mean(lo * cm)),
                           cond=float(np.mean(lo[cm])) if cm.any() else 0.0)
                acc.setdefault((axis, grp), []).append(rec)
    out = []
    for (axis, grp), recs in acc.items():
        df = pd.DataFrame(recs)
        out.append(dict(axis=axis, group=grp, n=int(df.n.mean()),
                        base=df.base.mean(), commit=df.commit.mean(),
                        joint=df.joint.mean(), cond=df.cond.mean(),
                        joint_max=df.joint.max()))
    return pd.DataFrame(out), runs


# ---- primary backbone detailed audit ----
prim = "runs_main/probe_qwen14b_n2000_seed42.npz"
print(f"=== SUBGROUP AUDIT (primary = qwen14b, global CRC alpha={ALPHA}, {len(SPLIT_SEEDS)} split-seeds) ===")
dfp, runs = audit_backbone(prim, detail=True)
order = {"age_grp": 0, "gender": 1, "race_grp": 2}
dfp = dfp.sort_values(by=["axis", "group"], key=lambda s: s.map(order) if s.name == "axis" else s)
print(f"runs={runs}   (commit=auto-commit rate, joint=committed-harm rate <=? {ALPHA}, cond=E[harm|commit])")
print(f"{'axis':9s} {'group':10s} {'n':>5s} {'base':>6s} {'commit':>7s} {'joint':>7s} {'joint_max':>9s} {'cond':>6s}")
for _, r in dfp.iterrows():
    flag = "" if r.joint <= ALPHA else "  <-- >alpha"
    print(f"{r.axis:9s} {r.group:10s} {r.n:5d} {r.base:6.3f} {r.commit:7.3f} "
          f"{r.joint:7.3f} {r.joint_max:9.3f} {r.cond:6.3f}{flag}")
dfp.insert(0, "backbone", "qwen14b")
dfp.to_csv("subgroup_audit_qwen14b.csv", index=False)

# ---- 8-backbone summary: commit-rate disparity + max subgroup joint risk ----
print(f"\n=== 8-BACKBONE SUMMARY: within-axis commit disparity & max subgroup joint risk ===")
print(f"{'backbone':16s} {'age dCommit':>11s} {'sex dCommit':>11s} {'race dCommit':>12s} {'max subgrp joint':>16s}")
allrows = []
for path in sorted(glob.glob("runs_main/probe_*_n2000_seed42.npz")):
    bb = re.match(r"probe_(.+)_n2000_seed", os.path.basename(path)).group(1)
    df, _ = audit_backbone(path)
    df["backbone"] = bb
    allrows.append(df)
    def dcom(ax):
        s = df[df.axis == ax]
        return (s.commit.max() - s.commit.min()) if len(s) else float("nan")
    print(f"{bb:16s} {dcom('age_grp'):11.3f} {dcom('gender'):11.3f} "
          f"{dcom('race_grp'):12.3f} {df.joint.max():16.3f}")
pd.concat(allrows).to_csv("subgroup_audit_8backbone.csv", index=False)
print("\nsaved -> subgroup_audit_qwen14b.csv, subgroup_audit_8backbone.csv")
