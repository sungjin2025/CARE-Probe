# -*- coding: utf-8 -*-
"""
B-2 incremental-value experiment (LOCAL): does the internal-state probe add
information NOT recoverable from presentation-time input?

  probe      = logistic on last-token hidden state (deployed layer idx3, depth 0.875)
  input_only = logistic on structured triage + HPI TF-IDF (presentation-time)
  combined   = logistic on [PCA(H,50) ++ PCA(TFIDF,100) ++ scaled struct]

Nested comparison via DeLong paired test + paired bootstrap ΔAUC.
All targets: ct_miss (practice-performed CT omitted from greedy prediction).

Data (all local, native stay_id join — no row-alignment trick):
  runs_main/probe_{bb}_n2000_seed{42,43,44}.npz : stay_ids, last(2000,5,3584), pred_bits, labels, valid
  cohort_v2.csv : stay_id, hpi_clean, structured triage, y_full__* gold
"""
import numpy as np, pandas as pd, glob, os, json, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
COHORT = os.path.join(HERE, "cohort_v2.csv")
RUNS = os.path.join(HERE, "runs_main")
DEPLOY_IDX = 3            # relative depth 0.875 (deployed protocol)
SEEDS = [42]
BACKBONES = ["qwen14b", "mediphi", "med42-8b", "gemma2-9b",
             "llama31-8b", "mistral7b-v03", "openbio-8b", "phi3mini-128k"]
CT_COLS = ["CT_ABD_PELVIS", "CT_HEAD", "CT_CSPINE", "CT_CHEST", "CT_EXTREMITY", "CT_PELVIS"]
STRUCT_NUM = ["pain", "age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
STRUCT_CAT = ["gender", "race", "arrival_transport", "triage"]

# ---------- fast DeLong (paired) ----------
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_paired(y, s1, s2):
    """two-sided p for AUC(s1)==AUC(s2), paired (same y). returns (auc1,auc2,p)."""
    y = np.asarray(y); order = (-np.stack([s1, s2])).argsort()  # not used; keep simple
    pos = y == 1; neg = ~pos
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([s1, s2])
    k = 2
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(preds[r, pos]); ty[r] = _compute_midrank(preds[r, neg])
        tz[r] = _compute_midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    l = np.array([[1, -1]])
    var = l @ delongcov @ l.T
    from scipy import stats
    if var[0, 0] <= 0:
        p = 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var[0, 0])
        p = 2 * stats.norm.sf(abs(z))
    return aucs[0], aucs[1], float(p)

def paired_bootstrap_delta(y, s_hi, s_lo, nboot=2000, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); n = len(y); ds = []
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        ds.append(roc_auc_score(yy, s_hi[idx]) - roc_auc_score(yy, s_lo[idx]))
    return float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

# ---------- feature builders (fit on train only) ----------
def oof_scores(H, Xstruct, hpi, y, seed):
    """5-fold OOF probability for probe / input_only / combined."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    s_pr = np.zeros(len(y)); s_in = np.zeros(len(y)); s_cb = np.zeros(len(y))
    for tr, te in skf.split(H, y):
        # probe
        sc = StandardScaler().fit(H[tr]); Htr, Hte = sc.transform(H[tr]), sc.transform(H[te])
        pca = PCA(n_components=min(50, Htr.shape[1]), random_state=seed).fit(Htr)
        Zh_tr, Zh_te = pca.transform(Htr), pca.transform(Hte)
        s_pr[te] = LogisticRegression(max_iter=2000).fit(Zh_tr, y[tr]).predict_proba(Zh_te)[:, 1]
        # input: struct + tfidf
        ssc = StandardScaler().fit(Xstruct[tr]); Xs_tr, Xs_te = ssc.transform(Xstruct[tr]), ssc.transform(Xstruct[te])
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000).fit(hpi[tr])
        Tt_tr, Tt_te = tf.transform(hpi[tr]), tf.transform(hpi[te])
        ncomp = min(100, Tt_tr.shape[1] - 1, Tt_tr.shape[0] - 1)
        tp = PCA(n_components=max(2, ncomp), random_state=seed).fit(Tt_tr.toarray())
        Zt_tr, Zt_te = tp.transform(Tt_tr.toarray()), tp.transform(Tt_te.toarray())
        In_tr = np.hstack([Xs_tr, Zt_tr]); In_te = np.hstack([Xs_te, Zt_te])
        s_in[te] = LogisticRegression(max_iter=2000).fit(In_tr, y[tr]).predict_proba(In_te)[:, 1]
        # combined
        Cb_tr = np.hstack([Zh_tr, In_tr]); Cb_te = np.hstack([Zh_te, In_te])
        s_cb[te] = LogisticRegression(max_iter=2000).fit(Cb_tr, y[tr]).predict_proba(Cb_te)[:, 1]
    return s_pr, s_in, s_cb

# ---------- main ----------
def main():
    coh = pd.read_csv(COHORT)
    coh = coh.drop_duplicates("stay_id").set_index("stay_id")
    for c in STRUCT_NUM:
        coh[c] = pd.to_numeric(coh[c], errors="coerce")
    cat = pd.get_dummies(coh[STRUCT_CAT].astype(str), dummy_na=True)
    struct_all = pd.concat([coh[STRUCT_NUM], cat], axis=1)
    hpi_all = coh["hpi_clean"].fillna("").astype(str)

    rows = []
    for bb in BACKBONES:
        per_seed = []
        deland = None
        for seed in SEEDS:
            fp = os.path.join(RUNS, f"probe_{bb}-deleak_n2000_seed{seed}.npz")
            if not os.path.exists(fp):
                continue
            d = np.load(fp, allow_pickle=True)
            sid = d["stay_ids"]; H = d["last"][:, DEPLOY_IDX, :].astype(np.float32)
            pred = d["pred_bits"]; labels = list(d["labels"]); valid = d["valid"].astype(bool)
            # ct_miss from gold vs pred
            gold = coh.reindex(sid)
            ct_idx = [labels.index(c) for c in CT_COLS if c in labels]
            gold_ct = np.stack([gold[f"y_full__{labels[i]}"].fillna(0).values for i in ct_idx], axis=1)
            pred_ct = pred[:, ct_idx]
            ct_miss = ((gold_ct == 1) & (pred_ct == 0)).any(axis=1).astype(int)
            # keep valid rows with cohort match
            keep = valid & np.isin(sid, struct_all.index.values)
            sid, H, y = sid[keep], H[keep], ct_miss[keep]
            Xs = struct_all.reindex(sid).astype(float).fillna(struct_all.median(numeric_only=True)).values
            hp = hpi_all.reindex(sid).fillna("").values
            s_pr, s_in, s_cb = oof_scores(H, Xs, hp, y, seed)
            a_pr, a_in, a_cb = roc_auc_score(y, s_pr), roc_auc_score(y, s_in), roc_auc_score(y, s_cb)
            per_seed.append((a_pr, a_in, a_cb))
            if seed == 42:
                _, _, p_cb_in = delong_paired(y, s_cb, s_in)
                _, _, p_pr_in = delong_paired(y, s_pr, s_in)
                dmu, dlo, dhi = paired_bootstrap_delta(y, s_cb, s_in, seed=seed)
                deland = dict(p_comb_vs_input=p_cb_in, p_probe_vs_input=p_pr_in,
                              dAUC_comb_minus_input=dmu, ci=[dlo, dhi], n=len(y), pos=int(y.sum()))
        if not per_seed:
            continue
        arr = np.array(per_seed)
        r = dict(backbone=bb, n_seed=len(per_seed),
                 auc_probe=round(arr[:, 0].mean(), 3), auc_input=round(arr[:, 1].mean(), 3),
                 auc_combined=round(arr[:, 2].mean(), 3),
                 d_comb_input=round((arr[:, 2] - arr[:, 1]).mean(), 3),
                 d_probe_input=round((arr[:, 0] - arr[:, 1]).mean(), 3))
        if deland:
            r.update(deland)
        rows.append(r)
        print(f"{bb:16s} probe={r['auc_probe']:.3f} input={r['auc_input']:.3f} "
              f"comb={r['auc_combined']:.3f} Δ(comb-input)={r['d_comb_input']:+.3f} "
              f"{'DeLong p(comb>in)=%.3f' % deland['p_comb_vs_input'] if deland else ''}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "b2_incremental_value-deleak.csv"), index=False)
    print("\nwrote b2_incremental_value.csv")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
