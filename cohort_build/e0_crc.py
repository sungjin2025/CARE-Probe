#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0_crc.py — CARE-MSR S5a: Conformal RISK Control on the internal-state probe.

Ladder: E0 (surface signals at chance, AUC~0.50) -> S3 (last-token internal-state
probe predicts error MODESTLY, AUC~0.66) -> S5a (HERE): put a distribution-free
RISK guarantee on the probe score so the *harmful-error rate among auto-committed
cases* is controlled at a user-set alpha.

Deferral rule (threshold lambda on the probe error-score s(x) in [0,1]):
    commit  (model answers)        if  s(x) <= lambda
    defer   (physician HITL)       if  s(x) >  lambda

Guarantee (the defensible one we implement):
    JOINT risk  R(lambda) = E[ l(x) * 1{s(x) <= lambda} ]  is MONOTONE non-decreasing
    in lambda and bounded in [0,1]. We pick the LARGEST lambda whose finite-sample
    high-probability UPPER bound (Learn-Then-Test / RCPS, Hoeffding-Bentkus p-value
    inversion, fixed-sequence testing) stays <= alpha. => maximal auto-answer rate
    s.t.  P( fraction of ALL patients that are auto-committed-AND-harmfully-wrong
    <= alpha )  >= 1 - delta.
    We ALSO report the *conditional* selective risk E[l | commit] empirically
    (the [needs verification] item: rigorous bound is on the joint risk; conditional is
    descriptive). [needs verification]: selective-CRC conditional-risk estimator vs joint.

Loss l(x) in [0,1] (E0 lesson: tie deferral to HARM, not confidence):
    primary  ct_miss : 1 if a high-risk CT study is in gold but missed by the prediction.
    second.  setloss : (2*#missed + 1*#over) / (2*#gold + #pred), capped to [0,1].

Two modes:
  (1) real     --npz runs_e0/e0_probe_n500_seed42.npz   (server; needs probe states)
  (2) selftest --selftest   (LOCAL; synthetic data of KNOWN probe-AUC ~ validates the
               guarantee machinery: empirical test commit-risk <= alpha across an
               alpha grid, violation rate <= delta. No GPU / no npz needed.)

Usage:
  python e0_crc.py --selftest                                   # local guarantee sanity
  CRMSR_ENV=server python e0_crc.py --npz runs_e0/e0_probe_n500_seed42.npz --window full

Reproducibility: all randomness seeded (--seed). Bounded losses verified in [0,1].
"""
from __future__ import annotations
import argparse
import math
import numpy as np

# ----------------------------------------------------------------------------
# Finite-sample upper-confidence machinery (LTT / RCPS, Hoeffding-Bentkus)
# ----------------------------------------------------------------------------
def _kl(a: float, b: float) -> float:
    """Binary KL divergence (nats), clamped for numerical safety."""
    a = min(max(a, 1e-12), 1 - 1e-12)
    b = min(max(b, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hb_pvalue(rhat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: R >= alpha, given empirical mean rhat of a
    [0,1] loss over n iid draws (Bates et al. 2021; Angelopoulos et al. LTT 2021).
    Small p => evidence that the true risk R < alpha => threshold is certifiable.
    Returns 1.0 when rhat >= alpha (no evidence)."""
    if n <= 0:
        return 1.0
    if rhat >= alpha:
        return 1.0
    hoeff = math.exp(-n * _kl(rhat, alpha))            # Hoeffding (KL form)
    try:
        from scipy.stats import binom
        bent = math.e * float(binom.cdf(math.ceil(n * rhat), n, alpha))
    except Exception:
        bent = hoeff
    return float(min(hoeff, bent))


def select_lambda(s_cal: np.ndarray, loss_cal: np.ndarray,
                  alpha: float, delta: float):
    """Pick the LARGEST threshold lambda s.t. the joint-risk UCB stays <= alpha.

    Joint risk R(lambda) = mean_i( loss_i * 1{s_i <= lambda} ) is monotone in lambda.
    Fixed-sequence testing from the safest (smallest) candidate upward; stop at the
    first non-certifiable threshold => FWER-controlled, P(R(lambda_hat) <= alpha) >= 1-delta.

    Returns (lambda_hat or None, calib_joint_risk_at_lambda). None => defer everything."""
    n = len(s_cal)
    cands = np.unique(s_cal)                  # every achievable commit boundary
    lam_hat, risk_at = None, 0.0
    for lam in np.sort(cands):               # ascending = safest first
        committed = s_cal <= lam
        rhat = float(np.mean(loss_cal * committed))   # JOINT risk over all calib
        if hb_pvalue(rhat, n, alpha) <= delta:
            lam_hat, risk_at = float(lam), rhat
        else:
            break                            # monotone -> everything above also fails
    return lam_hat, risk_at


# ----------------------------------------------------------------------------
# Risk-coverage / AURC (E-win: lower = better selective signal)
# ----------------------------------------------------------------------------
def risk_coverage(scores: np.ndarray, loss: np.ndarray):
    """Commit the lowest-score cases first; return (coverage, conditional_risk, AURC).
    AURC = area under the risk-coverage curve (conditional risk vs commit fraction)."""
    order = np.argsort(scores, kind="mergesort")     # ascending: low score = commit first
    ls = loss[order].astype(float)
    n = len(ls)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(ls) / np.arange(1, n + 1)
    return cov, risk, float(np.trapz(risk, cov))


def aurc_random(loss: np.ndarray, seed: int, reps: int = 200) -> float:
    """Mean AURC under random deferral (== naive E0-chance signal)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        _, _, a = risk_coverage(rng.random(len(loss)), loss)
        vals.append(a)
    return float(np.mean(vals))


# ----------------------------------------------------------------------------
# (1) LOCAL self-test: validate the guarantee on synthetic data of known probe-AUC
# ----------------------------------------------------------------------------
def synth_dataset(n: int, auc: float, p_err: float, harm_frac: float, rng):
    """error y~Bern(p_err); probe error-score s with target AUC for predicting y;
    loss l = y * Bern(harm_frac)  (CT-miss = a fraction of errors are high-harm).
    Returns (s in [0,1], loss in {0,1})."""
    y = (rng.random(n) < p_err).astype(int)
    # two-Gaussian scores: AUC = Phi(d/sqrt2) -> d = sqrt2 * Phi^{-1}(auc)
    from scipy.stats import norm
    d = math.sqrt(2.0) * float(norm.ppf(auc))
    raw = rng.normal(loc=d * y, scale=1.0)            # higher raw => more likely error
    s = 1.0 / (1.0 + np.exp(-raw))                    # squash to [0,1]
    loss = (y * (rng.random(n) < harm_frac)).astype(float)
    return s, loss


def run_selftest(args):
    rng = np.random.default_rng(args.seed)
    alphas = [0.02, 0.05, 0.10, 0.15, 0.20]
    print("=== S5a CRC self-test (synthetic, validates guarantee machinery) ===")
    print(f"  n={args.n}  probe_AUC≈{args.auc}  P(err)={args.p_err}  harm_frac={args.harm_frac}"
          f"  delta={args.delta}  trials={args.trials}")
    print(f"  guarantee target: test JOINT commit-risk <= alpha, violation rate should be <= delta\n")
    print(f"  {'alpha':>6} {'cover':>7} {'test_risk':>10} {'cond_risk':>10} "
          f"{'viol_rate':>10} {'defer_all%':>10}")
    overall_ok = True
    for alpha in alphas:
        covs, risks, conds, viol, defer_all = [], [], [], 0, 0
        for t in range(args.trials):
            s, loss = synth_dataset(args.n, args.auc, args.p_err, args.harm_frac, rng)
            idx = rng.permutation(args.n)
            half = args.n // 2
            cal, te = idx[:half], idx[half:]
            lam, _ = select_lambda(s[cal], loss[cal], alpha, args.delta)
            if lam is None:
                defer_all += 1
                covs.append(0.0); risks.append(0.0); conds.append(0.0)
                continue
            committed = s[te] <= lam
            joint = float(np.mean(loss[te] * committed))      # guaranteed quantity
            cov = float(np.mean(committed))
            cond = float(np.mean(loss[te][committed])) if committed.any() else 0.0
            covs.append(cov); risks.append(joint); conds.append(cond)
            if joint > alpha:
                viol += 1
        vr = viol / args.trials
        overall_ok &= (vr <= args.delta + 1e-9)
        print(f"  {alpha:>6.2f} {np.mean(covs):>7.3f} {np.mean(risks):>10.4f} "
              f"{np.mean(conds):>10.4f} {vr:>10.4f} {100*defer_all/args.trials:>9.1f}%")
    print()
    if overall_ok:
        print("  ✅ GUARANTEE HOLDS — empirical test joint-risk violation rate <= delta at every alpha.")
        print("     CRC machinery is correct; coverage rises with alpha (more auto-answer when looser).")
    else:
        print("  ❌ VIOLATION — some alpha exceeded delta. Inspect bound / fixed-sequence logic.")
    print("  (Note: bound is conservative (Hoeffding-Bentkus) => realized risk typically << alpha.)")
    return overall_ok


# ----------------------------------------------------------------------------
# (2) REAL data: probe scores from npz + clinical loss from cohort gold
# ----------------------------------------------------------------------------
def _fit_probe_scores(X_train, y_train, X_eval, n_pca, seed):
    """StandardScaler -> PCA -> Logistic; return P(error) on X_eval (S3 convention)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    n_pca = int(min(n_pca, X_train.shape[1], X_train.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_eval)[:, 1]


def _cv_probe_scores(X, y, n_pca, seed):
    """Out-of-fold P(error) over all rows (unbiased scores for the RC/AURC curves)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    n_pca = int(min(n_pca, X.shape[1], X.shape[0] - 1))
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_pca, random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def run_real(args):
    import paths
    import imaging_runner_v2 as ir
    LABELS = list(ir.LABEL_SPACE_V2)
    CT = [L for L in LABELS if L.startswith("CT_")]

    d = np.load(args.npz, allow_pickle=True)
    stay_ids = d["stay_ids"]; valid = d["valid"]
    pred_bits = d["pred_bits"]
    layer_idx = list(d["layer_idx"])
    last = d["last"]                                   # (N, n_layers, hidden) last-token states
    w = args.window

    coh = ir.read_csv_robust(args.cohort_csv)
    gcols = ["stay_id", "triage", f"card_{w}"] + [f"y_{w}__{L}" for L in LABELS]
    coh = coh[[c for c in gcols if c in coh.columns]].copy()
    gmap = {int(r["stay_id"]): r for r in coh.to_dict("records")}

    # align valid + gold-matched rows; build error label + clinical losses
    keep, y_err, l_ct, l_set, card, acu = [], [], [], [], [], []
    for i, sid in enumerate(stay_ids):
        if valid[i] != 1:
            continue
        try:
            g = gmap[int(sid)]
        except (KeyError, ValueError):
            continue
        gold = {L for L in LABELS if int(g.get(f"y_{w}__{L}", 0) or 0) == 1}
        pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j]) == 1}
        keep.append(i)
        y_err.append(int(pred != gold))
        ctg = {L for L in CT if L in gold}
        l_ct.append(int(len(ctg - pred) > 0) if ctg else 0)          # ct_miss
        n_miss, n_over = len(gold - pred), len(pred - gold)
        denom = 2 * len(gold) + len(pred)
        l_set.append(min(1.0, (2 * n_miss + n_over) / denom) if denom else 0.0)  # setloss
        card.append(float(g.get(f"card_{w}", 0) or 0))
        acu.append(int(g.get("triage", 0) or 0))
    keep = np.array(keep)
    y_err = np.array(y_err); l_ct = np.array(l_ct, float)
    l_set = np.array(l_set, float); card = np.array(card); acu = np.array(acu)

    # E-shift: map kept stays -> anchor_year_group (REAL temporal; intime is shifted/invalid in MIMIC)
    ayg = None
    if args.split == "temporal":
        if not args.shift_csv:
            raise SystemExit("--split temporal requires --shift_csv (run add_anchor_year.py first)")
        sc = ir.read_csv_robust(args.shift_csv)
        amap = {int(r["stay_id"]): str(r.get("anchor_year_group", "") or "")
                for r in sc.to_dict("records")}
        ayg = np.array([amap.get(int(stay_ids[i]), "") for i in keep])
        print(f"  [E-shift] anchor_year_group matched {100*np.mean(ayg!=''):.1f}% of kept stays")
    print(f"=== S5a CRC (real) === N_usable={len(keep)}  window={w}")
    print(f"  base rates: err={y_err.mean():.3f}  ct_miss={l_ct.mean():.3f}  "
          f"setloss(mean)={l_set.mean():.3f}")

    loss = l_ct if args.loss == "ct_miss" else l_set
    print(f"  loss = {args.loss}  (mean={loss.mean():.3f})  alpha={args.alpha}  delta={args.delta}")

    # probe training label: generic error (pred!=gold) OR the loss-specific label
    if args.probe_target == "loss":
        y_probe = (loss > 0).astype(int) if args.loss == "setloss" else loss.astype(int)
    else:
        y_probe = y_err
    print(f"  probe target = {args.probe_target} "
          f"({'loss-label:'+args.loss if args.probe_target=='loss' else 'generic err'}; "
          f"positives={int(y_probe.sum())}/{len(y_probe)})")

    # choose probe layer (last position). PRIORITY: --layer_idx (position, uniform a-priori) > --layer (value) > highest
    if 0 <= args.layer_idx < len(layer_idx):
        li = args.layer_idx
    elif args.layer in layer_idx:
        li = layer_idx.index(args.layer)
    else:
        li = len(layer_idx) - 1
    X = last[keep][:, li, :].astype(np.float32)
    print(f"  probe: last-token, layer={layer_idx[li]} (idx {li}/{len(layer_idx)-1}), PCA<= {args.n_pca}")

    # ---- E-guar: proper train/calib/test split, probe is a FIXED function ----
    rng = np.random.default_rng(args.seed)
    have_ayg = ayg is not None and (ayg != "").all()
    if args.split == "temporal" and have_ayg and args.shift_test_group:
        te = np.where(ayg == args.shift_test_group)[0]
        rest = np.where(ayg != args.shift_test_group)[0]
        rest = rest[np.argsort(ayg[rest], kind="mergesort")]      # earlier first
        n_tr = int(0.6 * len(rest))
        tr, ca = rest[:n_tr], rest[n_tr:]
        print(f"  split = TEMPORAL max-shift (test = '{args.shift_test_group}' only; calib/train = earlier)")
    elif args.split == "temporal" and have_ayg:
        idx = np.argsort(ayg, kind="mergesort")                   # year-group sorts chronologically
        n = len(idx); n_tr = int(0.4 * n); n_ca = int(0.3 * n)
        tr, ca, te = idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]
        print("  split = TEMPORAL (calib=earlier year-groups, test=latest 30%)")
    else:
        if args.split == "temporal":
            print("  [warn] anchor_year_group incomplete -> random split fallback")
        idx = rng.permutation(len(keep))
        n = len(idx); n_tr = int(0.4 * n); n_ca = int(0.3 * n)
        tr, ca, te = idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]
        print("  split = random")
    if have_ayg:
        import collections
        cc = lambda s: dict(sorted(collections.Counter(ayg[s]).items()))
        print(f"    calib year-groups: {cc(ca)}")
        print(f"    test  year-groups: {cc(te)}")
    s_ca = _fit_probe_scores(X[tr], y_probe[tr], X[ca], args.n_pca, args.seed)
    s_te = _fit_probe_scores(X[tr], y_probe[tr], X[te], args.n_pca, args.seed)
    lam, cal_risk = select_lambda(s_ca, loss[ca], args.alpha, args.delta)
    g_commit = g_joint = g_cond = float("nan")
    print("\n--- E-guar (guarantee) ---")
    if lam is None:
        g_commit = g_joint = g_cond = 0.0
        print(f"  lambda_hat=None -> defer ALL (no threshold certifiable at alpha={args.alpha}).")
    else:
        comm = s_te <= lam
        g_joint = float(np.mean(loss[te] * comm))
        g_cond = float(np.mean(loss[te][comm])) if comm.any() else 0.0
        g_commit = float(comm.mean())
        print(f"  lambda_hat={lam:.4f}  calib_joint_risk={cal_risk:.4f}")
        print(f"  TEST: commit_rate={g_commit:.3f}  joint_risk={g_joint:.4f} (<= {args.alpha}? "
              f"{'OK' if g_joint <= args.alpha else 'VIOLATION'})  cond_risk={g_cond:.4f}")

    # ---- E-sev: Mondrian group-conditional CRC, severity-adaptive alpha (high-acuity tighter) ----
    if args.mondrian:
        print("\n--- E-sev (severity-adaptive risk control; group-conditional CRC) ---")
        grp = np.where(np.isin(acu, [1, 2]), "high(ESI1-2)", "lower(ESI3-5)")
        ag = {"high(ESI1-2)": args.alpha_high, "lower(ESI3-5)": args.alpha_low}
        for gname in ["high(ESI1-2)", "lower(ESI3-5)"]:
            a_g = ag[gname]
            cm = grp[ca] == gname
            tm = grp[te] == gname
            lam_g, _ = select_lambda(s_ca[cm], loss[ca][cm], a_g, args.delta)
            if lam_g is None:
                print(f"  {gname}: n_cal={int(cm.sum())} n_te={int(tm.sum())} alpha={a_g} "
                      f"-> defer ALL (uncertifiable)")
                continue
            comm_g = s_te[tm] <= lam_g
            joint_g = float(np.mean(loss[te][tm] * comm_g))
            cond_g = float(np.mean(loss[te][tm][comm_g])) if comm_g.any() else 0.0
            print(f"  {gname}: n_te={int(tm.sum())} alpha={a_g} lambda={lam_g:.3f} "
                  f"commit={comm_g.mean():.3f} joint_risk={joint_g:.4f}"
                  f"({'OK' if joint_g <= a_g else 'VIOLATION'}) committed_harm={cond_g:.4f}")
        print("  -> high-acuity gets tighter alpha => more deferral but lower committed harm for the sickest.")

    # ---- E-win: AURC vs baselines (unbiased OOF probe scores over all kept rows) ----
    print("\n--- E-win (selective signal, lower AURC better) ---")
    s_all = _cv_probe_scores(X, y_probe, args.n_pca, args.seed)
    _, _, aurc_probe = risk_coverage(s_all, loss)
    _, _, aurc_card = risk_coverage(card, loss)         # oracle: commit LOW card first (defer high-card)
    aurc_rand = aurc_random(loss, args.seed)
    print(f"  probe (ours) : AURC={aurc_probe:.4f}")
    print(f"  naive/E0     : AURC={aurc_rand:.4f}  (random deferral = chance signal)")
    print(f"  oracle-card  : AURC={aurc_card:.4f}  (un-deployable upper bound)")
    span = aurc_rand - aurc_card
    pos = (aurc_rand - aurc_probe) / span if span > 1e-9 else float("nan")
    print(f"  -> probe closes {100*pos:.1f}% of the random->oracle gap "
          f"({'positive' if aurc_probe < aurc_rand else 'NOT better than chance'}).")

    if args.out_csv:
        cov, risk, _ = risk_coverage(s_all, loss)
        np.savetxt(args.out_csv, np.column_stack([cov, risk]),
                   delimiter=",", header="coverage,conditional_risk", comments="")
        print(f"\n  risk-coverage curve -> {args.out_csv}")

    # ---- append one structured result row (for multi-seed aggregation) ----
    if args.results_csv:
        import os
        newf = not os.path.exists(args.results_csv)
        with open(args.results_csv, "a") as f:
            if newf:
                f.write("tag,split,loss,probe_target,layer,seed,alpha,delta,n_usable,base_rate,"
                        "commit,joint_risk,cond_risk,aurc_probe,aurc_rand,aurc_card,gap_closed\n")
            f.write(f"{args.tag},{args.split},{args.loss},{args.probe_target},{layer_idx[li]},{args.seed},"
                    f"{args.alpha},{args.delta},{len(keep)},{loss.mean():.4f},"
                    f"{g_commit:.4f},{g_joint:.4f},{g_cond:.4f},"
                    f"{aurc_probe:.4f},{aurc_rand:.4f},{aurc_card:.4f},{pos:.4f}\n")
        print(f"  [results] appended -> {args.results_csv}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="S5a Conformal RISK Control on the probe")
    ap.add_argument("--selftest", action="store_true", help="local synthetic guarantee sanity")
    ap.add_argument("--npz", default=None, help="probe runner output (real mode)")
    ap.add_argument("--cohort_csv", default=None)
    ap.add_argument("--window", default="full", choices=["full", "early"])
    ap.add_argument("--loss", default="ct_miss", choices=["ct_miss", "setloss"])
    ap.add_argument("--probe_target", default="generic", choices=["generic", "loss"],
                    help="train probe on generic error (pred!=gold) or on the loss-specific label")
    ap.add_argument("--split", default="random", choices=["random", "temporal"],
                    help="train/calib/test split: random, or temporal(anchor_year_group) for E-shift")
    ap.add_argument("--shift_csv", default=None,
                    help="stay_anchor_year.csv sidecar (required for --split temporal)")
    ap.add_argument("--shift_test_group", default=None,
                    help="temporal: use ONLY this anchor_year_group as test (clean max-shift), "
                         "e.g. '2017 - 2019'; calib/train = all earlier groups")
    ap.add_argument("--mondrian", action="store_true",
                    help="E-sev: group-conditional CRC by ESI acuity (cohort 'triage') w/ adaptive alpha")
    ap.add_argument("--alpha_high", type=float, default=0.05, help="E-sev risk target, high-acuity ESI 1-2")
    ap.add_argument("--alpha_low", type=float, default=0.15, help="E-sev risk target, lower-acuity ESI 3-5")
    ap.add_argument("--alpha", type=float, default=0.10, help="risk target")
    ap.add_argument("--delta", type=float, default=0.10, help="guarantee failure prob")
    ap.add_argument("--layer", type=int, default=-1, help="probe layer value (real); -1=highest")
    ap.add_argument("--layer_idx", type=int, default=-1,
                    help="pick probe layer by POSITION 0..n_layers-1 (uniform a-priori across models); overrides --layer")
    ap.add_argument("--tag", default="", help="run label for results_csv (e.g. model name)")
    ap.add_argument("--results_csv", default=None, help="append one structured result row (for aggregation)")
    ap.add_argument("--n_pca", type=int, default=50)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
    # selftest knobs
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--auc", type=float, default=0.66, help="synthetic probe AUC (S3 ~0.66)")
    ap.add_argument("--p_err", type=float, default=0.5)
    ap.add_argument("--harm_frac", type=float, default=0.4)
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()

    if args.selftest:
        ok = run_selftest(args)
        raise SystemExit(0 if ok else 1)
    if not args.npz:
        ap.error("provide --selftest (local) or --npz <probe.npz> (real, server)")
    if args.cohort_csv is None:
        import paths
        args.cohort_csv = paths.COHORT_V2_CSV
    run_real(args)


if __name__ == "__main__":
    main()
