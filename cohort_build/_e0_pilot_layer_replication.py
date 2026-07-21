# -*- coding: utf-8 -*-
"""_e0_pilot_layer_replication.py — pilot-basis replication for the frozen probe
layer (Online Resource 20).

Purpose: replicate the per-(position, layer) probe AUC grid on the E0 pilot
screen (runs_e0/e0_probe_n500_seed42.npz, Qwen2.5-14B) to confirm that the
LAYER_IDX=3 (relative depth 0.875) frozen in MAIN is supported by the pilot.
Reproduces the e0_probe_analyze.py probe (Standardize->PCA<=50->Logistic,
5-fold CV) but reads cohort_v2.csv directly, without the paths /
imaging_runner_v2 dependency chain.
gold = y_full__{LABEL} (same as e0_crc.py --window full); ct_miss / y_err
definitions identical. Seeds fixed: PCA/CV seed=42 (as in the original script).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

NPZ = sys.argv[1] if len(sys.argv) > 1 else "runs_e0/e0_probe_n500_seed42.npz"
W = "full"

z = np.load(NPZ, allow_pickle=True)
stay_ids = z["stay_ids"]
valid = z["valid"]
pred_bits = z["pred_bits"]
LABELS = [str(x) for x in z["labels"]]
layer_idx = list(z["layer_idx"])
CT = [L for L in LABELS if L.startswith("CT_")]
n_layers_total = max(layer_idx)  # last grid point = final layer
fracs = [li / n_layers_total for li in layer_idx]

coh = pd.read_csv("cohort_v2.csv", usecols=lambda c: c == "stay_id" or c.startswith(f"y_{W}__"))
gmap = {int(r["stay_id"]): r for r in coh.to_dict("records")}

keep, y_err, l_ct = [], [], []
for i, sid in enumerate(stay_ids):
    if valid[i] != 1:
        continue
    g = gmap.get(int(sid))
    if g is None:
        continue
    gold = {L for L in LABELS if int(g.get(f"y_{W}__{L}", 0) or 0) == 1}
    pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j]) == 1}
    keep.append(i)
    y_err.append(int(pred != gold))
    ctg = {L for L in CT if L in gold}
    l_ct.append(int(len(ctg - pred) > 0) if ctg else 0)
keep = np.array(keep)
y_err = np.array(y_err)
l_ct = np.array(l_ct)
print(f"NPZ={NPZ}  usable={len(keep)}  base y_err={y_err.mean():.3f}  ct_miss={l_ct.mean():.3f}")
print(f"layer_idx={layer_idx}  (relative depth={['%.3f' % f for f in fracs]})")


def probe_auc(X, y, seed=42):
    if len(np.unique(y)) < 2:
        return float("nan")
    n_pca = int(min(50, X.shape[1], X.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


for tgt_name, tgt in [("ct_miss", l_ct), ("y_err", y_err)]:
    print(f"\n--- probe AUC per (position, layer) --- target={tgt_name}")
    best = (-1, None)
    for pos in ["prompt", "last"]:
        arr = z[pos][keep].astype(np.float32)  # (n, 5, hidden)
        row = []
        for li in range(len(layer_idx)):
            a = probe_auc(arr[:, li, :], tgt)
            row.append(a)
            if not np.isnan(a) and a > best[0]:
                best = (a, (pos, layer_idx[li], fracs[li]))
        print(f"  {pos:6s}: " + "  ".join(
            f"L{layer_idx[li]}(d={fracs[li]:.3f})={row[li]:.3f}" for li in range(len(layer_idx))))
    print(f"  -> best {tgt_name}: AUC={best[0]:.3f} @ pos={best[1][0]} layer={best[1][1]} depth={best[1][2]:.3f}")
