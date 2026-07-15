# -*- coding: utf-8 -*-
"""정본 파이프라인으로 walkthrough 3케이스 s(x)/commit/ct_miss 재검증(결정적)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, imaging_runner_v2 as ir
from e0_crc import _fit_probe_scores, select_lambda
L = list(ir.LABEL_SPACE_V2); CT = [x for x in L if x.startswith("CT_")]
coh = pd.read_csv(paths.COHORT_V2_CSV); gm = {int(r.stay_id): r for _, r in coh.iterrows()}
TARGETS = {"qwen14b": [38030632, 39789245], "mediphi": [34583596]}
for bb, stays in TARGETS.items():
    d = np.load(f"runs_main/probe_{bb}_n2000_seed42.npz", allow_pickle=True)
    sids, valid, pb, last, lidx = d["stay_ids"], d["valid"], d["pred_bits"], d["last"], list(d["layer_idx"])
    keep, lct = [], []
    for i, s in enumerate(sids):
        if valid[i] != 1 or int(s) not in gm: continue
        g = gm[int(s)]; gold = {x for x in L if int(g.get(f"y_full__{x}", 0) or 0) == 1}
        pred = {L[j] for j in range(len(L)) if int(pb[i][j]) == 1}
        keep.append(i); ctg = {x for x in CT if x in gold}
        lct.append(int(len(ctg - pred) > 0) if ctg else 0)
    keep = np.array(keep); lct = np.array(lct, float); y = lct.astype(int)
    X = last[keep][:, 3, :].astype(np.float32)
    rng = np.random.default_rng(42); idx = rng.permutation(len(keep))
    n = len(idx); tr, ca, te = idx[:int(.4*n)], idx[int(.4*n):int(.7*n)], idx[int(.7*n):]
    s_ca = _fit_probe_scores(X[tr], y[tr], X[ca], 50, 42)
    s_te = _fit_probe_scores(X[tr], y[tr], X[te], 50, 42)
    lam, _ = select_lambda(s_ca, lct[ca], .10, .10)
    keep_sid = sids[keep]
    for stay in stays:
        pos = np.where(keep_sid[te] == stay)[0]
        if len(pos) == 0: print(f"{bb} {stay}: NOT in test"); continue
        k = pos[0]; sx = float(s_te[k]); g = gm[stay]
        gold = sorted(x for x in L if int(g.get(f"y_full__{x}", 0) or 0) == 1)
        ri = te[k]; pred = sorted(L[j] for j in range(len(L)) if int(pb[keep[ri]][j]) == 1)
        print(f"{bb} lam={lam:.4f} | stay={stay} s(x)={sx:.4f} "
              f"decision={'COMMIT' if sx<=lam else 'DEFER'} ct_miss={int(lct[te][k])} "
              f"gold={gold} pred={pred}")
