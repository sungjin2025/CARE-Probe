# -*- coding: utf-8 -*-
"""_conditional_sgr_card.py -- CONDITIONAL risk certification (rebuttal card, no GPU).

Answers reviewer condition (b): "actually bound the conditional committed-error
rate E[l | commit]" -- implemented as SGR-style selection with guaranteed risk
(Geifman & El-Yaniv, NIPS 2017 = manuscript ref35): binary search over the probe
threshold with a Clopper-Pearson upper bound on the committed-case conditional
risk, Bonferroni-corrected over the <= ceil(log2 n_cal)+1 tested thresholds, so
    P( E[l | commit] <= alpha_cond ) >= 1 - delta   for the certified lambda.
Deferred cases exit; if no threshold certifies -> defer-all (no operating point).

Protocol is IDENTICAL to the frozen MAIN pipeline (_delta_sweep.py / e0_crc.py):
probe = last-token idx3 (rel-depth 0.875), Scaler->PCA(50)->Logistic trained on
train(40%), certified on calib(30%), evaluated on test(30%); loss = ct_miss;
3 sub-seeds x 10 split-seeds = 30 runs per (model, alpha_cond).
Sanity: the joint-CRC column at alpha=0.10 must reproduce Table 3 (Qwen
joint ~0.079 / commit ~0.429).

Run: python _conditional_sgr_card.py [model ...]
"""
import sys, math, csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_main"
COHORT = HERE / "cohort_v2.csv"
IDX3 = 3
SUBSEEDS = [42, 43, 44]
SPLITSEEDS = list(range(42, 52))
ALPHA_JOINT = 0.10          # sanity reproduction of the frozen MAIN guarantee
ALPHAS_COND = [0.10, 0.15, 0.20, 0.25]
DELTA = 0.10
MODELS = ["qwen14b", "llama31-8b", "gemma2-9b", "mistral7b-v03",
          "med42-8b", "openbio-8b", "phi3mini-128k", "mediphi"]

GOLD = pd.read_csv(COHORT, low_memory=False)


# ---- joint-risk CRC (frozen MAIN machinery, verbatim from _delta_sweep.py) ----
def _kl(a, b):
    a = min(max(a, 1e-12), 1 - 1e-12); b = min(max(b, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hb_pvalue(rhat, n, alpha):
    if n <= 0 or rhat >= alpha:
        return 1.0
    hoeff = math.exp(-n * _kl(rhat, alpha))
    from scipy.stats import binom
    bent = math.e * float(binom.cdf(math.ceil(n * rhat), n, alpha))
    return float(min(hoeff, bent))


def select_lambda_joint(s_cal, loss_cal, alpha, delta):
    n = len(s_cal); lam_hat = None
    for lam in np.sort(np.unique(s_cal)):
        rhat = float(np.mean(loss_cal * (s_cal <= lam)))
        if hb_pvalue(rhat, n, alpha) <= delta:
            lam_hat = float(lam)
        else:
            break
    return lam_hat


# ---- conditional-risk SGR (ref35): binary search + Clopper-Pearson + Bonferroni ----
def select_lambda_sgr(s_cal, loss_cal, alpha_cond, delta):
    """Largest-coverage lambda whose committed conditional risk is certified
    <= alpha_cond w.p. >= 1-delta. Binary loss required (ct_miss is 0/1)."""
    cands = np.sort(np.unique(s_cal))
    n = len(s_cal)
    k_steps = int(math.ceil(math.log2(max(n, 2)))) + 1
    dprime = delta / k_steps                      # Bonferroni over tested thresholds

    def ucb(lam):
        commit = s_cal <= lam
        m = int(commit.sum())
        if m == 0:
            return 1.0                            # vacuous: no operating point
        k = int(loss_cal[commit].sum())
        if k >= m:
            return 1.0
        return float(beta_dist.ppf(1.0 - dprime, k + 1, m - k))  # Clopper-Pearson upper

    lo, hi, best = 0, len(cands) - 1, None
    tests = 0
    while lo <= hi and tests < k_steps:
        mid = (lo + hi) // 2
        tests += 1
        if ucb(cands[mid]) <= alpha_cond:
            best = float(cands[mid]); lo = mid + 1
        else:
            hi = mid - 1
    return best


def ctmiss(stay_ids, valid, pred_bits, labels):
    LAB = list(labels); CT = [L for L in LAB if L.startswith("CT_")]
    cols = ["stay_id"] + [f"y_full__{L}" for L in LAB if f"y_full__{L}" in GOLD.columns]
    gmap = {int(r["stay_id"]): r for r in GOLD[cols].to_dict("records")}
    keep, y = [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        g = gmap.get(int(sid))
        if g is None:
            continue
        gold = {L for L in LAB if int(g.get(f"y_full__{L}", 0) or 0) == 1}
        pred = {LAB[j] for j in range(len(LAB)) if int(pred_bits[i][j]) == 1}
        ctg = {L for L in CT if L in gold}
        keep.append(i); y.append(int(len(ctg - pred) > 0) if ctg else 0)
    return np.array(keep), np.array(y, int)


def fit_scores(Xtr, ytr, Xev, seed):
    npca = int(min(50, Xtr.shape[1], Xtr.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=npca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xev)[:, 1]


def run_model(m):
    joint_rows = []                                # (joint_risk, commit) sanity
    cond_rows = {a: [] for a in ALPHAS_COND}       # (cond_te, commit, joint_te, deferall)
    for sd in SUBSEEDS:
        f = RUNS / f"probe_{m}_n2000_seed{sd}.npz"
        if not f.exists():
            print(f"  [skip] {f.name}"); continue
        d = np.load(f, allow_pickle=True)
        keep, y = ctmiss(d["stay_ids"], d["valid"], d["pred_bits"], d["labels"])
        X = d["last"][keep][:, IDX3, :].astype(np.float32)
        loss = y.astype(float)
        for ss in SPLITSEEDS:
            rng = np.random.default_rng(ss)
            idx = rng.permutation(len(keep))
            n = len(idx); ntr = int(0.4 * n); nca = int(0.3 * n)
            tr, ca, te = idx[:ntr], idx[ntr:ntr + nca], idx[ntr + nca:]
            s_ca = fit_scores(X[tr], y[tr], X[ca], ss)
            s_te = fit_scores(X[tr], y[tr], X[te], ss)
            # sanity: frozen joint CRC
            lamj = select_lambda_joint(s_ca, loss[ca], ALPHA_JOINT, DELTA)
            if lamj is None:
                joint_rows.append((0.0, 0.0))
            else:
                cj = s_te <= lamj
                joint_rows.append((float(np.mean(loss[te] * cj)), float(cj.mean())))
            # conditional SGR at each target
            for a in ALPHAS_COND:
                lam = select_lambda_sgr(s_ca, loss[ca], a, DELTA)
                if lam is None:
                    cond_rows[a].append((np.nan, 0.0, 0.0, 1))
                    continue
                cm = s_te <= lam
                if not cm.any():
                    cond_rows[a].append((np.nan, 0.0, 0.0, 1))
                    continue
                cond_te = float(np.mean(loss[te][cm]))
                cond_rows[a].append((cond_te, float(cm.mean()),
                                     float(np.mean(loss[te] * cm)), 0))
    return joint_rows, cond_rows


if __name__ == "__main__":
    targets = sys.argv[1:] or MODELS
    out_rows = []
    print(f"delta={DELTA}; 30 runs/cell; sanity joint alpha={ALPHA_JOINT}\n")
    for m in targets:
        jr, cr = run_model(m)
        j = np.array(jr)
        print(f"{m}: SANITY joint={j[:,0].mean():.4f} commit={j[:,1].mean():.4f} "
              f"(expect Table3)")
        for a in ALPHAS_COND:
            arr = cond_rows_np = np.array(cr[a], dtype=float)
            deferall = float(np.mean(arr[:, 3]))
            live = arr[arr[:, 3] == 0]
            if len(live) == 0:
                print(f"  a_cond={a:.2f}: DEFER-ALL in 30/30 runs (no operating point)")
                out_rows.append([m, a, "", "", "", "", f"{deferall:.3f}", "0"])
                continue
            cond_m = live[:, 0].mean(); com_m = live[:, 1].mean()
            jt_m = live[:, 2].mean()
            viol = float(np.mean(live[:, 0] > a))
            print(f"  a_cond={a:.2f}: cond={cond_m:.4f} commit={com_m:.4f} "
                  f"joint={jt_m:.4f} viol%={100*viol:.1f} deferall%={100*deferall:.1f} "
                  f"(live n={len(live)})")
            out_rows.append([m, a, f"{cond_m:.4f}", f"{com_m:.4f}", f"{jt_m:.4f}",
                             f"{viol:.3f}", f"{deferall:.3f}", str(len(live))])
        print()
    with open(HERE / "conditional_sgr_results.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["model", "alpha_cond", "cond_risk_test", "commit", "joint_test",
                    "run_viol_frac", "deferall_frac", "n_live_runs"])
        w.writerows(out_rows)
    print(f"[saved] {HERE / 'conditional_sgr_results.csv'}")
