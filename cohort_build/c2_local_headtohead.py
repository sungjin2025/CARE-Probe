# -*- coding: utf-8 -*-
"""
Standalone local head-to-head: CARE-Probe vs deployable Cardinality vs Random,
under the frozen CRC guarantee, on the CLINICAL harm target (ct_miss = CT label
missed). Mirrors c2_baseline_headtohead.py build()/scoring exactly, but imports
only e0_crc (numpy-only) and reimplements read_csv_robust to avoid the heavy
imaging_runner_v2 -> torch/milvus import chain.

Reproducibility: probe OOF seed fixed (42); gap_closed oracle = gold card_full;
CRC calib/test split over seeds 42..51.
"""
import os, sys
import numpy as np
import pandas as pd

# numpy>=2.0 renamed trapz -> trapezoid; e0_crc (frozen) still calls np.trapz.
# Alias is numerically identical; do not edit the frozen CRC module.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e0_crc import select_lambda, risk_coverage, aurc_random  # numpy-only, safe

NPZ_DIR = os.path.join(HERE, "runs_main")
COHORT = os.path.join(HERE, "cohort_v2.csv")
SUB_SEEDS = [42, 43, 44]
SPLIT_SEEDS = list(range(42, 52))
ALPHA, DELTA = 0.10, 0.10
BACKBONES = ["qwen14b", "mediphi", "gemma2-9b", "llama31-8b", "med42-8b",
             "mistral7b-v03", "openbio-8b", "phi3mini-128k"]  # qwen = smoke anchor


def read_cohort(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding_errors="replace")


def probe_oof(X, y, seed, n_pca=50):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    n_pca = int(min(n_pca, X.shape[1], X.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def build(bb, sub, cohort_df):
    d = np.load(f"{NPZ_DIR}/probe_{bb}_n2000_seed{sub}.npz", allow_pickle=True)
    LAB = [str(x) for x in d["labels"]]
    CT = [L for L in LAB if L.startswith("CT_")]
    sid, val, pb, last = d["stay_ids"], d["valid"], d["pred_bits"], d["last"]
    gm = {int(r["stay_id"]): r for r in cohort_df.to_dict("records")}
    keep, y, card, cardg = [], [], [], []
    for i, s in enumerate(sid):
        if val[i] != 1:
            continue
        g = gm.get(int(s))
        if g is None:
            continue
        gold = {L for L in LAB if int(g.get(f"y_full__{L}", 0) or 0) == 1}
        pred = {LAB[j] for j in range(len(LAB)) if int(pb[i][j]) == 1}
        ctg = {L for L in CT if L in gold}
        keep.append(i)
        y.append(int(len(ctg - pred) > 0) if ctg else 0)   # ct_miss
        card.append(len(pred))                              # deployable predicted cardinality
        cardg.append(float(g.get("card_full", 0) or 0))     # gold oracle
    keep = np.array(keep)
    y = np.array(y)
    card = np.array(card, float)
    cardg = np.array(cardg, float)
    X = last[keep][:, 3, :].astype(np.float32)
    return X, y, card, cardg


def score_vec(name, X, y, card, seed):
    if name == "CARE-Probe":
        return probe_oof(X, y, seed)
    if name == "Cardinality":
        return card.astype(float)
    if name == "Random":
        return np.random.default_rng(seed).random(len(y))
    raise ValueError(name)


def main():
    coh = read_cohort(COHORT)
    SIGNALS = ["CARE-Probe", "Cardinality", "Random"]
    rows = []
    for bb in BACKBONES:
        for sub in SUB_SEEDS:
            try:
                X, y, card, cardg = build(bb, sub, coh)
            except FileNotFoundError as e:
                print(f"[skip] {bb} sub{sub}: {e}")
                continue
            prev = y.mean()
            for nm in SIGNALS:
                s_all = score_vec(nm, X, y, card, 42)
                _, _, aurc_sig = risk_coverage(s_all, y.astype(float))
                _, _, aurc_or = risk_coverage(cardg, y.astype(float))
                aurc_rd = aurc_random(y.astype(float), 42)
                span = aurc_rd - aurc_or
                gap = (aurc_rd - aurc_sig) / span if abs(span) > 1e-9 else float("nan")
                jrs, cms, oks = [], [], []
                for sp in SPLIT_SEEDS:
                    rng = np.random.default_rng(sp)
                    idx = rng.permutation(len(y))
                    ntr = int(0.4 * len(y))
                    nca = int(0.3 * len(y))
                    ca, te = idx[ntr:ntr + nca], idx[ntr + nca:]
                    lam, _ = select_lambda(s_all[ca], y[ca].astype(float), ALPHA, DELTA)
                    if lam is None:
                        jr = cm = 0.0
                    else:
                        c = s_all[te] <= lam
                        jr = float(np.mean(y[te] * c))
                        cm = float(c.mean())
                    jrs.append(jr); cms.append(cm); oks.append(int(jr <= ALPHA))
                rows.append(dict(backbone=bb, sub=sub, signal=nm, n=len(y),
                                 prev=round(prev, 3), gap_closed=round(gap, 3),
                                 joint_risk=round(np.mean(jrs), 4),
                                 commit=round(np.mean(cms), 3),
                                 ok_rate=round(np.mean(oks), 2)))
            print(f"[done] {bb} sub{sub} (n={len(y)}, ct_miss prev={prev:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "c2_local_headtohead_results.csv"), index=False)
    print("\n=== PER (backbone x signal): mean over sub-seeds x split-seeds ===")
    g = df.groupby(["backbone", "signal"]).agg(
        gap_closed=("gap_closed", "mean"), commit=("commit", "mean"),
        joint_risk=("joint_risk", "mean"), ok_rate=("ok_rate", "mean")
    ).reindex(pd.MultiIndex.from_product([BACKBONES, SIGNALS]))
    print(g.round(3).to_string())
    print(f"\n[saved] c2_local_headtohead_results.csv")


if __name__ == "__main__":
    main()
