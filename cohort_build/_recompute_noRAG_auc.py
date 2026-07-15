# -*- coding: utf-8 -*-
"""noRAG(closed-book) deployed last-token probe AUC 재계산.
기존 _recompute_lasttoken_auc.py의 파이프라인을 그대로 재사용(동일 정의 보장):
 position=last, layer=idx3(0.875), target=ct_miss, StdScaler->PCA50->Logistic,
 5-fold StratifiedKFold(seed=42) cross_val_predict, 3 generation sub-seed 평균.
대상: runs_e0/noRAG_2k_s{42,43,44}.npz (manifest §11 미검증 잔여 0.67 검증).
대조군으로 RAG main qwen14b(0.621)도 재출력.
"""
import math, statistics as st
import numpy as np
from pathlib import Path
import _recompute_lasttoken_auc as R   # 동일 파이프라인 재사용

HERE = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
IDX3 = R.DEPLOY_POS_IDX  # = 3

def auc_for_npz(path):
    d = np.load(path, allow_pickle=True)
    keep, y = R.ctmiss_labels(d["stay_ids"], d["valid"], d["pred_bits"], d["labels"])
    X = d["last"][keep][:, IDX3, :].astype(np.float32)
    return R.probe_auc(X, y), y.mean(), len(y), int(y.sum())

def summarize(vals):
    vals = [v for v in vals if not math.isnan(v)]
    m = st.mean(vals)
    h = (2.262 * st.stdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, h

print("=== noRAG (closed-book) deployed last-token@idx3 ct_miss probe AUC ===")
aucs, bases = [], []
for sd in SEEDS:
    f = HERE / "runs_e0" / f"noRAG_2k_s{sd}.npz"
    if not f.exists():
        print(f"  [missing] {f.name}"); continue
    a, base, n, npos = auc_for_npz(f)
    aucs.append(a); bases.append(base)
    print(f"  seed{sd}: AUC={a:.4f}  base_ctmiss={base:.4f}  (n={n}, pos={npos})")
m, h = summarize(aucs)
bm, bh = summarize(bases)
print(f"  --> 3-seed mean AUC = {m:.4f} ± {h:.4f}   base_ctmiss mean = {bm:.4f}")
print(f"  manifest 주장: noRAG AUC ~0.67, base 0.178")

# 대조군: RAG main qwen14b (이미 검증된 0.621)
print("\n=== [대조] RAG main qwen14b last@idx3 (manifest 0.621) ===")
(qm, qh), _, _, qbase = R.run_model("qwen14b")
print(f"  qwen14b RAG: last@idx3 = {qm:.4f} ± {qh:.4f}  base_ctmiss = {qbase:.4f}  (manifest 0.621, base 0.247)")
print(f"\n=== 비교: noRAG {m:.3f} vs RAG {qm:.3f}  (manifest 'comparable 0.67 vs 0.62') ===")
