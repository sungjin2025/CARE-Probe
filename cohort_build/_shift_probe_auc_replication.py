# -*- coding: utf-8 -*-
"""_shift_probe_auc_replication.py — probe discrimination measured UNDER temporal
shift (internal-review item R1-M2; main-text Table 4 / Online Resource 21).

Uses the same split and probe construction as E-shift (run_c3 / e0_crc.py
--split temporal --shift_test_group "2017 - 2019"): the probe is trained on
earlier anchor_year_group bins and its discrimination (AUC, AURC gap_closed)
is evaluated on the 2017-2019 test bin only. These shift-measured values
replace the in-distribution OOF gap% previously re-printed in Table 4.

Pipeline reproduced from e0_crc.py:
  - split: te = ayg=="2017 - 2019"; rest sorted by ayg (mergesort, earlier
    first), tr = rest[:60%], ca = rest[60%:]  (L293-299)
  - probe: last-token layer_idx=3, StandardScaler->PCA(<=50, rs=seed)->Logistic,
    probe_target=loss (trained on the ct_miss label; run_c3 L44)  (L183-195, 270-286)
  - loss/gold: y_full__*, ct_miss = gold-CT-minus-pred set non-empty  (L225-249)
  - AURC: risk_coverage (commit low-score first), aurc_random (200 reps),
    oracle = gold cardinality (card_full) ascending, gap=(rand-probe)/(rand-card)
    (L97-117, 356-367; np.trapz -> np.trapezoid, numpy>=2.0)
Runs: 8 backbones x sub-seeds{42,43,44} x split-seeds{42..51} = 240 (same scale as Table 4).
"""
import glob
import os
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

W = "full"
LAYER_POS = 3          # LAYER_IDX=3 (relative depth 0.875)
N_PCA = 50
SHIFT_TEST_GROUP = "2017 - 2019"
SPLIT_SEEDS = list(range(42, 52))

coh = pd.read_csv("cohort_v2.csv",
                  usecols=lambda c: c in ("stay_id", f"card_{W}") or c.startswith(f"y_{W}__"))
gmap = {int(r["stay_id"]): r for r in coh.to_dict("records")}
ay = pd.read_csv("stay_anchor_year.csv")
amap = {int(r["stay_id"]): str(r.get("anchor_year_group", "") or "") for r in ay.to_dict("records")}


def risk_coverage_aurc(scores, loss):
    order = np.argsort(scores, kind="mergesort")
    ls = loss[order].astype(float)
    n = len(ls)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(ls) / np.arange(1, n + 1)
    return float(np.trapezoid(risk, cov))


def aurc_random(loss, seed, reps=200):
    rng = np.random.default_rng(seed)
    return float(np.mean([risk_coverage_aurc(rng.random(len(loss)), loss) for _ in range(reps)]))


def fit_probe_scores(X_train, y_train, X_eval, seed):
    n_pca = int(min(N_PCA, X_train.shape[1], X_train.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_eval)[:, 1]


rows = []
npzs = sorted(glob.glob("runs_main/probe_*_n2000_seed*.npz"))
for path in npzs:
    m = re.match(r"probe_(.+)_n2000_seed(\d+)\.npz", os.path.basename(path))
    bb, sub = m.group(1), int(m.group(2))
    z = np.load(path, allow_pickle=True)
    stay_ids, valid, pred_bits = z["stay_ids"], z["valid"], z["pred_bits"]
    LABELS = [str(x) for x in z["labels"]]
    CT = [L for L in LABELS if L.startswith("CT_")]
    last = z["last"]

    keep, l_ct, card, ayg = [], [], [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        g = gmap.get(int(sid))
        a = amap.get(int(sid), "")
        if g is None or a == "":
            continue
        gold = {L for L in LABELS if int(g.get(f"y_{W}__{L}", 0) or 0) == 1}
        pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j]) == 1}
        ctg = {L for L in CT if L in gold}
        keep.append(i)
        l_ct.append(int(len(ctg - pred) > 0) if ctg else 0)
        card.append(float(g.get(f"card_{W}", 0) or 0))
        ayg.append(a)
    keep = np.array(keep)
    loss = np.array(l_ct)
    card = np.array(card)
    ayg = np.array(ayg)
    X = last[keep][:, LAYER_POS, :].astype(np.float32)

    te = np.where(ayg == SHIFT_TEST_GROUP)[0]
    rest = np.where(ayg != SHIFT_TEST_GROUP)[0]
    rest = rest[np.argsort(ayg[rest], kind="mergesort")]
    tr = rest[:int(0.6 * len(rest))]
    if len(te) < 30 or loss[te].sum() < 5 or loss[tr].sum() < 5:
        print(f"[skip] {bb} sub{sub}: te={len(te)} pos_te={int(loss[te].sum())}")
        continue

    for sp in SPLIT_SEEDS:
        s_te = fit_probe_scores(X[tr], loss[tr], X[te], sp)
        auc = roc_auc_score(loss[te], s_te)
        a_probe = risk_coverage_aurc(s_te, loss[te])
        a_rand = aurc_random(loss[te], sp)
        a_card = risk_coverage_aurc(card[te], loss[te])
        span = a_rand - a_card
        gap = (a_rand - a_probe) / span if span > 1e-9 else float("nan")
        rows.append(dict(backbone=bb, sub=sub, seed=sp, n_tr=len(tr), n_te=len(te),
                         base_te=float(loss[te].mean()), auc_shift=float(auc),
                         gap_shift=float(100 * gap)))
    r = [x for x in rows if x["backbone"] == bb and x["sub"] == sub]
    print(f"[done] {bb} sub{sub}: n_te={r[0]['n_te']} base={r[0]['base_te']:.3f} "
          f"AUC={np.mean([x['auc_shift'] for x in r]):.3f} "
          f"gap={np.mean([x['gap_shift'] for x in r]):.1f}%", flush=True)

df = pd.DataFrame(rows)
df.to_csv("shift_probe_auc_replication.csv", index=False)
print("\n==== per-backbone summary (mean +- 95% CI over runs) ====")
for bb, g in df.groupby("backbone"):
    for col in ["auc_shift", "gap_shift"]:
        v = g[col].values
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        print(f"  {bb:16s} {col}: {v.mean():7.3f} +- {ci:.3f}  (runs={len(v)})")
print("\nsaved -> shift_probe_auc_replication.csv")
