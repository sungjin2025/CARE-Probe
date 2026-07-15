#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""c2_baseline_headtohead.py — head-to-head: CARE-Probe vs prior deferral signals.

Mock-review R2-W3/Q5 + meta Tier1-8: compare CARE-Probe against existing
deferral approaches ON THE SAME COHORT, SAME CRC/AURC machinery. We hold the
guarantee pipeline fixed (e0_crc.select_lambda / risk_coverage) and swap only the
deferral score s(x):

  * CARE-Probe     : internal-state probe (last-token idx3), 5-fold OOF   [ours]
  * Conf-CRC       : 1 - token-logprob confidence  (Kwon & Kim [20] analogue:
                     "calibrated confidence + conformal deferral")
  * SelfCons-CRC   : self-consistency uncertainty U_sc
  * Suff-CRC       : 1 - retrieval sufficiency (defer when evidence weak)
  * Conflict-CRC   : cross-source conflict D_jac
  * Cardinality    : predicted set size |y_hat| (MedAbstain [21] set-size analogue)
  * Random         : chance floor

Each score runs the IDENTICAL guarantee (joint risk <= alpha, largest-lambda LTT)
and AURC gap-closed, aggregated over sub-seeds x split-seeds. The claim under test:
a confidence/consistency baseline + risk control closes ~0% of the random->oracle
gap (surface signals are at chance, Table 1), while CARE-Probe closes 40-76%.

Inputs (all in the SAME 2000-case subsample; stay_id aligned):
  --npz_dir runs_main   probe_<bb>_n2000_seed<sub>.npz   (CARE-Probe + |y_hat| + gold)
  --sig_dir runs_e0     e0_preds_n2000_<bb>_seed<sub>.csv (conflict_runner: van_conf,
                        {P,W,G}_strength, {P,W,G}__<label>)   [optional]
                        e0_unc_n2000_<bb>_seed<sub>.csv     (uncertainty_runner: U_sc)  [optional]
Signals absent -> that baseline is skipped (CARE-Probe/Cardinality/Random always run).

Output: c2_headtohead_results.csv  (backbone, signal, gap_closed, joint_risk,
        commit, guarantee_ok) + printed summary table.

Run AFTER collecting the n=2000 surface signals (run_c2_baselines.sh).
Placed in cohort_build_v2/ so `import e0_crc` works. Seeds fixed.
"""
import os, argparse, itertools
import numpy as np, pandas as pd
from e0_crc import select_lambda, risk_coverage, aurc_random   # frozen CRC machinery
import imaging_runner_v2 as ir
import paths

BBS = ["gemma2-9b","llama31-8b","med42-8b","mediphi","mistral7b-v03","openbio-8b","phi3mini-128k","qwen14b"]
SUB_SEEDS = [42,43,44]
SPLIT_SEEDS = list(range(42,52))
SRC = ["P","W","G"]

def jaccard(a,b):
    u=a|b; return len(a&b)/len(u) if u else 1.0

def probe_oof(X, y, seed, n_pca=50):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    n_pca=int(min(n_pca,X.shape[1],X.shape[0]-1))
    pipe=make_pipeline(StandardScaler(),PCA(n_components=n_pca,random_state=seed),
                       LogisticRegression(max_iter=2000,C=1.0))
    cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    return cross_val_predict(pipe,X,y,cv=cv,method="predict_proba")[:,1]

def build(bb, sub, npz_dir, sig_dir):
    d=np.load(f"{npz_dir}/probe_{bb}_n2000_seed{sub}.npz",allow_pickle=True)
    LAB=[str(x) for x in d["labels"]]; CT=[L for L in LAB if L.startswith("CT_")]
    sid=d["stay_ids"]; val=d["valid"]; pb=d["pred_bits"]; last=d["last"]
    coh=ir.read_csv_robust(paths.COHORT_V2_CSV)
    gm={int(r["stay_id"]):r for r in coh.to_dict("records")}
    keep=[]; y=[]; card=[]; cardg=[]; stays=[]
    for i,s in enumerate(sid):
        if val[i]!=1: continue
        g=gm.get(int(s))
        if g is None: continue
        gold={L for L in LAB if int(g.get(f"y_full__{L}",0) or 0)==1}
        pred={LAB[j] for j in range(len(LAB)) if int(pb[i][j])==1}
        ctg={L for L in CT if L in gold}
        keep.append(i); y.append(int(len(ctg-pred)>0) if ctg else 0)
        card.append(len(pred))                          # predicted set size |y_hat| (MedAbstain baseline signal)
        cardg.append(float(g.get("card_full",0) or 0))  # gold cardinality card_full (ORACLE; matches e0_crc L248)
        stays.append(int(s))
    keep=np.array(keep); y=np.array(y); card=np.array(card,float)
    cardg=np.array(cardg,float); stays=np.array(stays)
    if not np.any(cardg):                               # safety: card_full missing -> oracle degenerate
        print(f"[warn] {bb} sub{sub}: card_full all-zero (column missing in cohort?) -> oracle degenerate")
    X=last[keep][:,3,:].astype(np.float32)
    sig={}   # stay_id-aligned surface signals (optional)
    pc=f"{sig_dir}/e0_preds_n2000_{bb}_seed{sub}.csv"
    if os.path.exists(pc):
        p=pd.read_csv(pc)
        if "van_conf" in p: sig["conf"]=p.set_index("stay_id")["van_conf"].to_dict()
        st={sk:p.set_index("stay_id").get(f"{sk}_strength") for sk in SRC}
        if all(v is not None for v in st.values()):
            sig["suff"]={s:max(st["P"].get(s,0),st["W"].get(s,0),st["G"].get(s,0)) for s in p["stay_id"]}
        # conflict D_jac from per-source label bits
        try:
            djac={}
            for r in p.to_dict("records"):
                sets=[{L for L in LAB if int(r.get(f"{sk}__{L}",0) or 0)==1} for sk in SRC]
                pr=list(itertools.combinations(sets,2))
                djac[int(r["stay_id"])]=1.0-float(np.mean([jaccard(a,b) for a,b in pr]))
            sig["conflict"]=djac
        except Exception: pass
    uc=f"{sig_dir}/e0_unc_n2000_{bb}_seed{sub}.csv"
    if os.path.exists(uc):
        u=pd.read_csv(uc)
        if "U_sc" in u: sig["selfcons"]=u.set_index("stay_id")["U_sc"].to_dict()
    return X, y, card, cardg, stays, sig

def score_vec(name, X, y, card, stays, sig, seed):
    """Return deferral score s(x) in [0,1]-ish (higher = more likely to defer)."""
    if name=="CARE-Probe":   return probe_oof(X,y,seed)
    if name=="Cardinality":  return card.astype(float)                 # more labels -> defer (MedAbstain analogue)
    if name=="Random":       return np.random.default_rng(seed).random(len(y))
    m=sig.get({"Conf-CRC":"conf","SelfCons-CRC":"selfcons","Suff-CRC":"suff","Conflict-CRC":"conflict"}[name])
    if m is None: return None
    v=np.array([m.get(int(s),np.nan) for s in stays],float)
    if np.isnan(v).all(): return None
    v=np.where(np.isnan(v),np.nanmedian(v),v)
    if name=="Conf-CRC":  v=1.0-v      # low confidence -> defer
    if name=="Suff-CRC":  v=1.0-v      # weak evidence  -> defer
    return v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz_dir",default="runs_main")
    ap.add_argument("--sig_dir",default="runs_e0")
    ap.add_argument("--alpha",type=float,default=0.10)
    ap.add_argument("--delta",type=float,default=0.10)
    ap.add_argument("--out",default="c2_headtohead_results.csv")
    ap.add_argument("--backbones",default=" ".join(BBS))
    args=ap.parse_args()
    SIGNALS=["CARE-Probe","Conf-CRC","SelfCons-CRC","Suff-CRC","Conflict-CRC","Cardinality","Random"]
    rows=[]
    for bb in args.backbones.split():
        for sub in SUB_SEEDS:
            try: X,y,card,cardg,stays,sig=build(bb,sub,args.npz_dir,args.sig_dir)
            except FileNotFoundError as e: print(f"[skip] {bb} sub{sub}: {e}"); continue
            for nm in SIGNALS:
                s_all=score_vec(nm,X,y,card,stays,sig,42)
                if s_all is None:
                    continue
                # AURC gap-closed (OOF-style, whole set) — oracle = gold cardinality card_full (matches e0_crc)
                _,_,aurc_sig=risk_coverage(s_all, y.astype(float))
                _,_,aurc_or =risk_coverage(cardg, y.astype(float))
                aurc_rd=aurc_random(y.astype(float),42)
                span=aurc_rd-aurc_or
                gap=(aurc_rd-aurc_sig)/span if span>1e-9 else float("nan")
                # guarantee via CRC over split-seeds
                for sp in SPLIT_SEEDS:
                    rng=np.random.default_rng(sp); idx=rng.permutation(len(y))
                    ntr=int(0.4*len(y)); nca=int(0.3*len(y))
                    ca,te=idx[ntr:ntr+nca],idx[ntr+nca:]
                    lam,_=select_lambda(s_all[ca], y[ca].astype(float), args.alpha, args.delta)
                    if lam is None: jr=cm=0.0
                    else:
                        c=s_all[te]<=lam; jr=float(np.mean(y[te]*c)); cm=float(c.mean())
                    rows.append(dict(backbone=bb,sub=sub,split=sp,signal=nm,
                                     gap_closed=gap,joint_risk=jr,commit=cm,ok=int(jr<=args.alpha)))
            print(f"[done] {bb} sub{sub}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(args.out,index=False)
    print("\n=== HEAD-TO-HEAD (mean over sub x split) ===")
    g=df.groupby(["signal"]).agg(gap=("gap_closed","mean"),joint=("joint_risk","mean"),
                                 commit=("commit","mean"),ok=("ok","mean")).reindex(SIGNALS).dropna()
    print(g.round(3).to_string())
    print(f"\nwrote {args.out} ({len(df)} rows)")

if __name__=="__main__":
    main()
