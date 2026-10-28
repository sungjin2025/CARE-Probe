# -*- coding: utf-8 -*-
"""
E0X confound analysis: is pred_card a self-awareness signal or a gold-cardinality
(difficulty) proxy?

For each backbone x target (y_err, ct_miss):
  1. corr(pred_card, gold_card)  [Spearman + Pearson]
  2. AUC(pred_card -> target)          : raw predictive power
  3. AUC(gold_card -> target)          : the oracle difficulty proxy
  4. Partial AUC of pred_card | gold_card:
       - residualize pred_card on gold_card (OLS), AUC(residual -> target)
       - stratified/within-stratum AUC pooled across gold_card bins
  5. Logistic regression: target ~ gold_card (+ pred_card), likelihood-ratio test
     for the incremental contribution of pred_card beyond gold_card.
Reproducibility: fixed seed, deterministic.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

SEED = 42
np.random.seed(SEED)
HERE = Path(__file__).parent
BACKBONES = ["qwen14b", "mediphi", "gemma2-9b",
             "llama31-8b", "med42-8b", "mistral7b-v03", "openbio-8b", "phi3mini-128k"]
TARGETS = ["y_err", "ct_miss"]


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return np.nan
    a = roc_auc_score(y, s)
    return max(a, 1 - a)  # direction-agnostic, matches e0x_analyze convention


def residualize(x, z):
    """residual of x after OLS on z (with intercept)."""
    Z = np.column_stack([np.ones_like(z, dtype=float), z.astype(float)])
    beta, *_ = np.linalg.lstsq(Z, x.astype(float), rcond=None)
    return x.astype(float) - Z @ beta


def stratified_auc(y, s, strata):
    """pool within-stratum concordance: mean AUC weighted by discordant-pair count."""
    tot_w, acc = 0.0, 0.0
    for g in np.unique(strata):
        m = strata == g
        yy, ss = y[m], s[m]
        if len(np.unique(yy)) < 2:
            continue
        npos, nneg = (yy == 1).sum(), (yy == 0).sum()
        w = npos * nneg
        a = roc_auc_score(yy, ss)
        acc += w * a
        tot_w += w
    if tot_w == 0:
        return np.nan, 0
    return acc / tot_w, int(tot_w)


def lr_test(y, gold, pred):
    """LR test: does pred_card add beyond gold_card? returns (chi2, p, auc_full, auc_gold)."""
    y = np.asarray(y)
    g = np.asarray(gold, dtype=float).reshape(-1, 1)
    p = np.asarray(pred, dtype=float).reshape(-1, 1)
    m = ~(np.isnan(g[:, 0]) | np.isnan(p[:, 0]))
    y, g, p = y[m], g[m], p[m]

    def fit_ll(X):
        clf = LogisticRegression(max_iter=2000, C=1e6)
        clf.fit(X, y)
        prob = clf.predict_proba(X)[:, 1]
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        ll = np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        return ll, clf, prob

    ll_g, _, prob_g = fit_ll(g)
    Xf = np.column_stack([g[:, 0], p[:, 0]])
    ll_f, _, prob_f = fit_ll(Xf)
    chi2 = 2 * (ll_f - ll_g)
    pval = stats.chi2.sf(chi2, df=1)
    return chi2, pval, safe_auc(y, prob_f), safe_auc(y, prob_g)


rows = []
for bb in BACKBONES:
    df = pd.read_csv(HERE / f"{bb}-deleak_deployed_surface.csv")
    for tgt in TARGETS:
        y = df[tgt].astype(int).values
        pred = df["pred_card"].astype(float).values
        gold = df["gold_card"].astype(float).values

        r_p, _ = stats.pearsonr(pred, gold)
        r_s, _ = stats.spearmanr(pred, gold)
        auc_pred = safe_auc(y, pred)
        auc_gold = safe_auc(y, gold)
        resid = residualize(pred, gold)
        auc_resid = safe_auc(y, resid)
        auc_strat, wpairs = stratified_auc(y, pred, gold.astype(int))
        chi2, pval, auc_full, auc_g2 = lr_test(y, gold, pred)

        rows.append(dict(
            backbone=bb, target=tgt, n=len(y), prev=round(y.mean(), 3),
            corr_pearson=round(r_p, 3), corr_spearman=round(r_s, 3),
            auc_pred=round(auc_pred, 3), auc_gold=round(auc_gold, 3),
            auc_pred_resid_on_gold=round(auc_resid, 3),
            auc_pred_within_goldstrata=round(auc_strat, 3),
            lr_chi2=round(chi2, 2), lr_p=f"{pval:.1e}",
        ))

out = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(out.to_string(index=False))
out.to_csv(HERE / "confound_summary-deleak.csv", index=False)
print(f"\n[saved] {HERE / 'confound_summary.csv'}")
