# -*- coding: utf-8 -*-
"""
last-token CT-miss probe AUC를 npz에서 단일 정의로 재계산해 정합성 확정.
단일 정의: position='last', layer=idx3(상대깊이 0.875, 배포 레이어), target=ct_miss,
          StandardScaler->PCA(50)->Logistic, 5-fold StratifiedKFold CV (seed=42), 3 sub-seed 평균.
부수: prompt@idx3, grid-best(pos×layer) 도 같이 내어 어느 정본 계열이 무엇인지 식별.
실행: python _recompute_lasttoken_auc.py [model1 model2 ...]
"""
import sys, csv, statistics as st, math
import numpy as np
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
DEPLOY_POS_IDX = 3   # idx3 = 상대깊이 0.875
SEEDS = [42, 43, 44]
GAP = {"gemma2-9b":67.7,"llama31-8b":61.1,"med42-8b":68.1,"mediphi":76.5,
       "mistral7b-v03":54.0,"openbio-8b":45.4,"phi3mini-128k":2.7,"qwen14b":40.2}
MODELS = list(GAP.keys())

def probe_auc(X, y, n_pca=50, seed=42):
    if len(np.unique(y)) < 2: return float("nan")
    n_pca = int(min(n_pca, X.shape[1], X.shape[0]-1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))

# cohort gold (full window) 로드
def load_gold():
    import pandas as pd
    df = pd.read_csv(COHORT, low_memory=False)
    return df
GOLD = load_gold()

def ctmiss_labels(stay_ids, valid, pred_bits, labels):
    LAB = list(labels); CT = [L for L in LAB if L.startswith("CT_")]
    gcols = {int(r["stay_id"]): r for r in
             GOLD[["stay_id"] + [f"y_full__{L}" for L in LAB if f"y_full__{L}" in GOLD.columns]]
             .to_dict("records")}
    keep, y = [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1: continue
        g = gcols.get(int(sid))
        if g is None: continue
        gold = {L for L in LAB if int(g.get(f"y_full__{L}", 0) or 0) == 1}
        pred = {LAB[j] for j in range(len(LAB)) if int(pred_bits[i][j]) == 1}
        ctg = {L for L in CT if L in gold}
        keep.append(i); y.append(int(len(ctg - pred) > 0) if ctg else 0)
    return np.array(keep), np.array(y, dtype=int)

def run_model(m):
    last_idx3, prompt_idx3, gridbest = [], [], []
    base = None
    for sd in SEEDS:
        f = RUNS / f"probe_{m}_n2000_seed{sd}.npz"
        if not f.exists():
            print(f"  [skip] {f.name} 없음"); continue
        d = np.load(f, allow_pickle=True)
        keep, y = ctmiss_labels(d["stay_ids"], d["valid"], d["pred_bits"], d["labels"])
        base = y.mean()
        nl = d["last"].shape[1]
        Xl = d["last"][keep][:, DEPLOY_POS_IDX, :].astype(np.float32)
        last_idx3.append(probe_auc(Xl, y))
        Xp = d["prompt"][keep][:, DEPLOY_POS_IDX, :].astype(np.float32)
        prompt_idx3.append(probe_auc(Xp, y))
        gb = -1
        for posname in ("last", "prompt"):
            for li in range(nl):
                a = probe_auc(d[posname][keep][:, li, :].astype(np.float32), y)
                if not math.isnan(a) and a > gb: gb = a
        gridbest.append(gb)
    def ms(v):
        v=[x for x in v if not math.isnan(x)]
        if not v: return (float("nan"), 0.0)
        m_=st.mean(v); h=(2.262*st.stdev(v)/math.sqrt(len(v))) if len(v)>1 else 0.0
        return (m_, h)
    return ms(last_idx3), ms(prompt_idx3), ms(gridbest), base

if __name__ == "__main__":
    targets = sys.argv[1:] or MODELS
    rows = []
    print(f"{'backbone':16s} {'last@idx3':>14s} {'prompt@idx3':>14s} {'grid-best':>12s} {'base':>6s} {'gap%':>6s}")
    for m in targets:
        (lm,lh),(pm,ph),(gm,gh),base = run_model(m)
        print(f"{m:16s} {lm:.3f}±{lh:.3f}   {pm:.3f}±{ph:.3f}   {gm:.3f}±{gh:.3f}   {base:.3f}  {GAP[m]:5.1f}")
        rows.append((m, lm, lh, pm, gm, base, GAP[m]))
    # 상관 (last@idx3 mean vs gap%)
    if len(rows) >= 3:
        import numpy as np
        L=np.array([r[1] for r in rows]); G=np.array([r[6] for r in rows])
        def spearman(a,b):
            ra=a.argsort().argsort().astype(float); rb=b.argsort().argsort().astype(float)
            return float(np.corrcoef(ra,rb)[0,1])
        print(f"\n[corr] last@idx3 vs gap%:  Pearson r={float(np.corrcoef(L,G)[0,1]):+.3f}  Spearman rho={spearman(L,G):+.3f}")
    with open(HERE/"lasttoken_auc_recomputed.csv","w",newline="",encoding="utf-8") as fo:
        w=csv.writer(fo); w.writerow(["backbone","last_idx3","last_ci","prompt_idx3","grid_best","base_ctmiss","gap_pct"])
        w.writerows([(m,f"{lm:.4f}",f"{lh:.4f}",f"{pm:.4f}",f"{gm:.4f}",f"{base:.4f}",g) for (m,lm,lh,pm,gm,base,g) in rows])
    print(f"\n[saved] {HERE/'lasttoken_auc_recomputed.csv'}")
