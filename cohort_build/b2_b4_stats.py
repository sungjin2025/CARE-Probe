#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B2 + B4 (mock-review defense, 2026-07-06).

B2: combined multivariate model over ALL surface signals (Table 1 defense):
    features = token-logprob uncertainty (1 - mean conf), self-consistency U_sc,
    retrieval sufficiency (S_max, S_mean), cross-source conflict (D_jac, D_lab,
    D_dec, D_high) [+ variant adding deployable |y_hat| cardinality].
    Data = E0 artifacts (e0_preds_n500_seed42.csv + e0_unc_n500_seed42.csv +
    cohort gold), error label = majority-vote ct_miss / exact_mismatch,
    exactly as e0_analyze.py. 5-fold OOF logistic AUC + 1000-rep bootstrap CI.
    Per-signal AUCs with bootstrap CIs for Table 1 rows (incl. numeric sufficiency).

B4: (a) Table 3 correlation: bootstrap CI (10k), permutation p, leave-one-out
    influence, n=8 and n=7 (Phi excluded).
    (b) Binomial consistency: observed guarantee violations vs Binom(n, delta).
"""
import os
import numpy as np, pandas as pd, math
from itertools import combinations

BASE = os.environ.get("WORK_DIR", "/path/to/cohort_build")
LAB15 = None

def jaccard(a, b):
    u = a | b
    return (len(a & b) / len(u)) if u else 1.0

def main():
    rng = np.random.default_rng(42)
    pred = pd.read_csv(f"{BASE}/runs_e0/e0_preds_n500_seed42.csv")
    unc  = pd.read_csv(f"{BASE}/runs_e0/e0_unc_n500_seed42.csv")
    coh  = pd.read_csv(f"{BASE}/cohort_v2.csv", low_memory=False)
    LABELS = [c[len("van__"):] for c in pred.columns if c.startswith("van__") and not c.endswith("no_imaging")]
    CT = [L for L in LABELS if L.startswith("CT_")]
    SRC = ["P","W","G"]
    gold_cols = [f"y_full__{L}" for L in LABELS] + ["no_imaging_full"]
    m = pred.merge(coh[["stay_id"]+gold_cols], on="stay_id", how="left")
    m = m.merge(unc[["stay_id","U_sc"]], on="stay_id", how="left")
    ok = m[["van_flag","P_flag","W_flag","G_flag"]].apply(lambda r: all(str(v)=="ok" for v in r), axis=1)
    m = m[ok].reset_index(drop=True)
    print(f"n valid = {len(m)}")

    feats, y_ct, y_ex, card_pred = [], [], [], []
    for rec in m.to_dict("records"):
        sets = {sk: {L for L in LABELS if int(rec.get(f"{sk}__{L}",0) or 0)==1} for sk in SRC}
        gold = {L for L in LABELS if int(rec.get(f"y_full__{L}",0) or 0)==1}
        noimg = {sk: int(rec.get(f"{sk}__no_imaging",0) or 0) for sk in SRC}
        maj = {L for L in LABELS if sum(1 for sk in SRC if L in sets[sk]) >= 2}
        # conflict metrics
        pairs = list(combinations([sets[s] for s in SRC], 2))
        d_jac = 1.0 - float(np.mean([jaccard(a,b) for a,b in pairs]))
        d_lab = float(np.mean([len({0,1} <= {int(L in sets[sk]) for sk in SRC}) if False else (len({int(L in sets[sk]) for sk in SRC})>1) for L in LABELS]))
        dec = [1 if (len(sets[sk])>0 or noimg[sk]==0) else 0 for sk in SRC]
        d_dec = float(len(set(dec))>1)
        cts = [1 if any(L in sets[sk] for L in CT) else 0 for sk in SRC]
        d_high = float(len(set(cts))>1)
        s_vals = [float(rec.get(f"{sk}_strength",0.0) or 0.0) for sk in SRC]
        c_vals = [float(rec.get(f"{sk}_conf",0.0) or 0.0) for sk in SRC]
        u_lp = 1.0 - float(np.mean(c_vals))
        u_sc = float(rec.get("U_sc") if rec.get("U_sc")==rec.get("U_sc") else 0.0)
        feats.append([u_lp, u_sc, max(s_vals), float(np.mean(s_vals)), d_jac, d_lab, d_dec, d_high])
        card_pred.append(len(maj))
        ctg = {L for L in CT if L in gold}
        y_ct.append(int(len(ctg - maj)>0) if ctg else 0)
        y_ex.append(int(maj != gold))
    X = np.array(feats); card_pred = np.array(card_pred, float)
    names = ["U_logprob","U_selfcons","S_max","S_mean","D_jac","D_lab","D_dec","D_high"]
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    def boot_ci_auc(yv, s, reps=1000):
        idx = np.arange(len(yv)); vals=[]
        for _ in range(reps):
            b = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(yv[b]))<2: continue
            vals.append(roc_auc_score(yv[b], s[b]))
        return np.percentile(vals,[2.5,97.5])

    for tgt, yv in [("ct_miss", np.array(y_ct)), ("exact_mismatch", np.array(y_ex))]:
        print(f"\n=== B2 [{tgt}] base rate={yv.mean():.3f} ===")
        # per-signal AUC + CI (single feature, oriented)
        for j,nm in enumerate(names):
            s = X[:,j]
            a = roc_auc_score(yv, s); a = max(a, 1-a)
            lo,hi = boot_ci_auc(yv, s if roc_auc_score(yv,s)>=0.5 else -s)
            print(f"  {nm:11s} AUC={a:.3f}  95%CI[{lo:.3f},{hi:.3f}]")
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        s_all = cross_val_predict(pipe, X, yv, cv=cv, method="predict_proba")[:,1]
        a = roc_auc_score(yv, s_all); lo,hi = boot_ci_auc(yv, s_all)
        print(f"  COMBINED(8 feats, 5-fold OOF) AUC={a:.3f}  95%CI[{lo:.3f},{hi:.3f}]")
        X9 = np.column_stack([X, card_pred])
        s9 = cross_val_predict(pipe, X9, yv, cv=cv, method="predict_proba")[:,1]
        a9 = roc_auc_score(yv, s9); lo9,hi9 = boot_ci_auc(yv, s9)
        print(f"  COMBINED+|y_hat| (9 feats)   AUC={a9:.3f}  95%CI[{lo9:.3f},{hi9:.3f}]")

    # ---------------- B4 ----------------
    print("\n=== B4(a) Table 3 correlation robustness ===")
    auc = np.array([0.743,0.709,0.728,0.693,0.654,0.645,0.621,0.514])
    gap = np.array([76.5,68.1,67.7,61.1,54.0,45.4,40.2,2.7])
    bbs = ["mediphi","med42","gemma","llama","mistral","openbio","qwen","phi"]
    def pear(x,y_):
        return float(np.corrcoef(x,y_)[0,1])
    for label, mask in [("n=8", np.ones(8,bool)), ("n=7 (excl Phi)", np.arange(8)!=7)]:
        x,yv = auc[mask], gap[mask]
        r = pear(x,yv)
        boots=[]
        n=len(x); idx=np.arange(n)
        for _ in range(10000):
            b = rng.choice(idx,n,replace=True)
            if np.std(x[b])<1e-9 or np.std(yv[b])<1e-9: continue
            boots.append(pear(x[b],yv[b]))
        lo,hi = np.percentile(boots,[2.5,97.5])
        perm=[]
        for _ in range(10000):
            perm.append(pear(x, rng.permutation(yv)))
        p = float(np.mean(np.abs(perm)>=abs(r)))
        print(f"  {label}: r={r:+.3f}  boot95%CI[{lo:+.3f},{hi:+.3f}]  perm p={p:.4f}")
    loo = [pear(np.delete(auc,i), np.delete(gap,i)) for i in range(8)]
    print("  leave-one-out r range:", f"[{min(loo):+.3f}, {max(loo):+.3f}]",
          " (dropped:", bbs[int(np.argmin(loo))], "→ min)")

    print("\n=== B4(b) binomial consistency of run-level violations vs delta=0.10 ===")
    from scipy.stats import binom
    checks = [("MAIN 240", 240, 240-222), ("MAIN+shift 270", 270, 270-252),
              ("B-4 RAG 30", 30, 2), ("B-4 CB 30", 30, 3), ("B-3 full-n 10", 10, 1),
              ("B-6 a=0.10 240", 240, 8)]
    for nm,n,k in checks:
        p_ge = 1 - binom.cdf(k-1, n, 0.10)
        print(f"  {nm}: violations {k}/{n} = {k/n:.3f};  P(X>={k}|Binom(n,0.10)) = {p_ge:.3f}"
              f"  -> {'consistent' if k/n <= 0.10 or p_ge>0.05 else 'CHECK'}")

if __name__ == "__main__":
    main()
