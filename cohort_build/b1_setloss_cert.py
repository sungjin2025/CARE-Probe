#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1: l_set certification table (mock-review defense, 2026-07-06).

Protocol = frozen MAIN (e0_crc.py run_real) with ONE change: the certified loss
is l_set (two-sided) instead of l_ct_miss. The probe itself is UNCHANGED —
trained on the ct_miss harm label (probe_target=loss, as in MAIN), last-token
layer idx3, StandardScaler->PCA50->LogisticRegression, train40/calib30/test30.
CRC (LTT / Hoeffding-Bentkus, fixed-sequence largest-lambda) certifies
R(lambda)=E[l_set * 1{s<=lambda}] <= alpha at delta=0.10.

Because l_set is graded with a high base mean (~0.6), alpha=0.10 is a very
tight budget; we therefore evaluate an alpha grid {0.10,...,0.50} at no extra
fit cost (lambda selection is cheap once scores exist).

Fidelity check built in: for each run we ALSO certify l_ct_miss at alpha=0.10
and compare to frozen results_main.csv (must reproduce commit/joint per run).

Seeds fixed; no fabrication — every number recomputed from runs_main npz + cohort_v2.csv.
"""
import math, os, sys, time
import numpy as np
import pandas as pd

BASE = os.environ.get("WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "runs_main")
COHORT = os.path.join(BASE, "cohort_v2.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b1_setloss_results.csv")

BBS = ["gemma2-9b","llama31-8b","med42-8b","mediphi","mistral7b-v03","openbio-8b","phi3mini-128k","qwen14b"]
SUB_SEEDS = [42,43,44]
SPLIT_SEEDS = list(range(42,52))
ALPHAS = [0.10, 0.20, 0.30, 0.40, 0.50]
DELTA = 0.10
LAYER_IDX = 3
N_PCA = 50
W = "full"

def _kl(a,b):
    a=min(max(a,1e-12),1-1e-12); b=min(max(b,1e-12),1-1e-12)
    return a*math.log(a/b)+(1-a)*math.log((1-a)/(1-b))

def hb_pvalue(rhat,n,alpha):
    if n<=0 or rhat>=alpha: return 1.0
    hoeff = math.exp(-n*_kl(rhat,alpha))
    try:
        from scipy.stats import binom
        bent = math.e*float(binom.cdf(math.ceil(n*rhat),n,alpha))
    except Exception:
        bent = hoeff
    return float(min(hoeff,bent))

def select_lambda(s_cal,loss_cal,alpha,delta):
    n=len(s_cal); lam_hat=None; risk_at=0.0
    for lam in np.sort(np.unique(s_cal)):
        rhat=float(np.mean(loss_cal*(s_cal<=lam)))
        if hb_pvalue(rhat,n,alpha)<=delta:
            lam_hat,risk_at=float(lam),rhat
        else:
            break
    return lam_hat,risk_at

def fit_probe_scores(X_train,y_train,X_eval,n_pca,seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    n_pca=int(min(n_pca,X_train.shape[1],X_train.shape[0]-1))
    pipe=make_pipeline(StandardScaler(),
                       PCA(n_components=n_pca,random_state=seed),
                       LogisticRegression(max_iter=2000,C=1.0))
    pipe.fit(X_train,y_train)
    return pipe.predict_proba(X_eval)[:,1]

def main():
    coh = pd.read_csv(COHORT, low_memory=False)
    rows=[]
    t0=time.time()
    for bb in BBS:
        for sub in SUB_SEEDS:
            npz_path=os.path.join(RUNS,f"probe_{bb}_n2000_seed{sub}.npz")
            d=np.load(npz_path,allow_pickle=True)
            LABELS=[str(x) for x in d["labels"]]
            CT=[L for L in LABELS if L.startswith("CT_")]
            stay_ids=d["stay_ids"]; valid=d["valid"]; pred_bits=d["pred_bits"]; last=d["last"]
            gcols=["stay_id","triage",f"card_{W}"]+[f"y_{W}__{L}" for L in LABELS]
            sub_coh=coh[[c for c in gcols if c in coh.columns]]
            gmap={int(r["stay_id"]):r for r in sub_coh.to_dict("records")}
            keep=[]; l_ct=[]; l_set=[]
            for i,sid in enumerate(stay_ids):
                if valid[i]!=1: continue
                g=gmap.get(int(sid))
                if g is None: continue
                gold={L for L in LABELS if int(g.get(f"y_{W}__{L}",0) or 0)==1}
                pred={LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j])==1}
                keep.append(i)
                ctg={L for L in CT if L in gold}
                l_ct.append(int(len(ctg-pred)>0) if ctg else 0)
                n_miss,n_over=len(gold-pred),len(pred-gold)
                denom=2*len(gold)+len(pred)
                l_set.append(min(1.0,(2*n_miss+n_over)/denom) if denom else 0.0)
            keep=np.array(keep)
            l_ct=np.array(l_ct,float); l_set=np.array(l_set,float)
            y_probe=l_ct.astype(int)                     # UNCHANGED deployed probe target (MAIN)
            X=last[keep][:,LAYER_IDX,:].astype(np.float32)
            n=len(keep)
            for seed in SPLIT_SEEDS:
                rng=np.random.default_rng(seed)
                idx=rng.permutation(n)
                n_tr=int(0.4*n); n_ca=int(0.3*n)
                tr,ca,te=idx[:n_tr],idx[n_tr:n_tr+n_ca],idx[n_tr+n_ca:]
                s_ca=fit_probe_scores(X[tr],y_probe[tr],X[ca],N_PCA,seed)
                s_te=fit_probe_scores(X[tr],y_probe[tr],X[te],N_PCA,seed)
                # fidelity: ct_miss at alpha=0.10 must reproduce frozen MAIN
                lam0,_=select_lambda(s_ca,l_ct[ca],0.10,DELTA)
                if lam0 is None:
                    fid_commit=fid_joint=0.0
                else:
                    c0=s_te<=lam0
                    fid_commit=float(c0.mean()); fid_joint=float(np.mean(l_ct[te]*c0))
                for alpha in ALPHAS:
                    lam,_=select_lambda(s_ca,l_set[ca],alpha,DELTA)
                    if lam is None:
                        commit=joint=cond=0.0; lam_o=float("nan")
                    else:
                        comm=s_te<=lam
                        joint=float(np.mean(l_set[te]*comm))
                        cond=float(np.mean(l_set[te][comm])) if comm.any() else 0.0
                        commit=float(comm.mean()); lam_o=lam
                    rows.append(dict(backbone=bb,sub_seed=sub,split_seed=seed,alpha=alpha,
                                     base_setloss=float(l_set.mean()),lam=lam_o,
                                     commit=commit,joint=joint,cond=cond,
                                     ok=int(joint<=alpha),
                                     fid_ct_commit=fid_commit,fid_ct_joint=fid_joint))
            print(f"[{time.time()-t0:7.1f}s] {bb} sub{sub} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT,index=False)   # checkpoint per backbone
    pd.DataFrame(rows).to_csv(OUT,index=False)
    print("WROTE",OUT,len(rows),"rows")

if __name__=="__main__":
    main()
