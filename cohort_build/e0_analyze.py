#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0_analyze.py — CARE-MSR E0 analysis: cross-source conflict ↔ error.

Reads e0_conflict_runner.py output + cohort_v2 gold, computes:
  A1 conflict prevalence | A2 structure | A3 conflict↔error correlation
  A4 incremental info (does conflict predict error beyond sufficiency+uncertainty?)
  A5 qualitative high-conflict cases | + GO/NO-GO verdict.
(See proposals/E0_conflict_analysis_design.md)

Usage:
  python e0_analyze.py --pred_csv runs_e0/e0_preds_n500_seed42.csv --window full
"""
from __future__ import annotations
import argparse, json, itertools
import numpy as np
import pandas as pd

import paths
import imaging_runner_v2 as ir

LABELS = list(ir.LABEL_SPACE_V2)
NO_IMG = ir.NO_IMAGING_LABEL
CT_LABELS = [L for L in LABELS if L.startswith("CT_")]
SRC = ["P", "W", "G"]

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False
try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False


def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def src_set(rec, sk) -> set:
    return {L for L in LABELS if int(rec.get(f"{sk}__{L}", 0)) == 1}


# ---------- conflict metrics D(x) ----------
def d_jac(sets):  # mean pairwise Jaccard DISTANCE among 3 source sets
    return float(np.mean([1.0 - jaccard(a, b) for a, b in itertools.combinations(sets, 2)]))

def d_lab(recs_bits):  # fraction of labels where the 3 sources disagree
    dis = 0
    for L in LABELS:
        votes = {recs_bits[sk][L] for sk in SRC}
        if len(votes) > 1: dis += 1
    return dis / len(LABELS)

def d_dec(sets, noimg):  # decision (img / no) disagreement among sources
    decs = set()
    for sk in SRC:
        decs.add("img" if len(sets[sk]) > 0 else ("no" if noimg[sk] == 1 else "empty"))
    return 1.0 if len(decs) > 1 else 0.0

def d_high(sets):  # CT presence disagreement
    has = {any(L in sets[sk] for L in CT_LABELS) for sk in SRC}
    return 1.0 if len(has) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--cohort_csv", default=paths.COHORT_V2_CSV)
    ap.add_argument("--window", default="full", choices=["full", "early"])
    ap.add_argument("--error", default="exact_mismatch",
                    choices=["exact_mismatch", "under_imaging", "ct_miss"])
    ap.add_argument("--u_csv", default="",
                    help="optional self-consistency uncertainty CSV (e0_uncertainty_runner) "
                         "to REPLACE the broken logprob U")
    ap.add_argument("--u_col", default="U_sc", help="uncertainty column in --u_csv")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)
    coh = ir.read_csv_robust(args.cohort_csv)
    w = args.window
    gold_cols = [f"y_{w}__{L}" for L in LABELS] + [f"no_imaging_{w}"]
    coh = coh[["stay_id"] + [c for c in gold_cols if c in coh.columns]].copy()
    m = pred.merge(coh, on="stay_id", how="left")

    # keep rows where all four predictions are valid
    flag_cols = [c for c in ["van_flag", "P_flag", "W_flag", "G_flag"] if c in m.columns]
    ok = m[flag_cols].apply(lambda r: all(str(v) == "ok" for v in r), axis=1) if flag_cols else pd.Series([True]*len(m))
    n_total = len(m); m = m[ok].reset_index(drop=True)
    print(f"=== E0 analysis (window={w}) ===")
    print(f"rows: {n_total} total, {len(m)} with all-valid predictions "
          f"({100*len(m)/max(n_total,1):.0f}%)")
    if len(m) == 0:
        print("no valid rows — abort"); return

    # per-row derived quantities
    D = {"D_jac": [], "D_lab": [], "D_dec": [], "D_high": []}
    err = {"exact_mismatch": [], "under_imaging": [], "over_imaging": [], "ct_miss": []}
    Smax, Smean, Umean = [], [], []
    gold_card, van_correct = [], []
    maj_sets, gold_sets, src_sets_all = [], [], []

    for rec in m.to_dict("records"):
        sets = {sk: src_set(rec, sk) for sk in SRC}
        bits = {sk: {L: int(rec.get(f"{sk}__{L}", 0)) for L in LABELS} for sk in SRC}
        noimg = {sk: int(rec.get(f"{sk}__{NO_IMG}", 0)) for sk in SRC}
        gold = {L for L in LABELS if int(rec.get(f"y_{w}__{L}", 0) or 0) == 1}
        gold_noimg = int(rec.get(f"no_imaging_{w}", 0) or 0)

        D["D_jac"].append(d_jac([sets["P"], sets["W"], sets["G"]]))
        D["D_lab"].append(d_lab(bits))
        D["D_dec"].append(d_dec(sets, noimg))
        D["D_high"].append(d_high(sets))

        # majority vote aggregate over the 3 sources
        maj = {L for L in LABELS if sum(bits[sk][L] for sk in SRC) >= 2}
        maj_sets.append(maj); gold_sets.append(gold); src_sets_all.append(sets)

        err["exact_mismatch"].append(int(maj != gold))
        err["under_imaging"].append(int(len(gold - maj) > 0))
        err["over_imaging"].append(int(len(maj - gold) > 0))
        ct_gold = {L for L in CT_LABELS if L in gold}
        err["ct_miss"].append(int(len(ct_gold - maj) > 0) if ct_gold else 0)

        s_vals = [float(rec.get(f"{sk}_strength", 0.0) or 0.0) for sk in SRC]
        c_vals = [float(rec.get(f"{sk}_conf", 0.0) or 0.0) for sk in SRC]
        Smax.append(max(s_vals)); Smean.append(float(np.mean(s_vals)))
        Umean.append(1.0 - float(np.mean(c_vals)))  # uncertainty = 1 - mean confidence
        gold_card.append(len(gold))
        van_correct.append(int(src_set(rec, "van") == gold))

    for k in D: m[k] = D[k]
    for k in err: m[k] = err[k]
    m["S_max"] = Smax; m["S_mean"] = Smean; m["U_mean"] = Umean
    m["gold_card"] = gold_card; m["van_correct"] = van_correct

    # optional: replace broken logprob U with self-consistency U
    if args.u_csv:
        u = pd.read_csv(args.u_csv)
        m = m.merge(u[["stay_id", args.u_col]], on="stay_id", how="left")
        miss = int(m[args.u_col].isna().sum())
        m["U_mean"] = m[args.u_col].fillna(m[args.u_col].median())
        print(f"  [U override] U = '{args.u_col}' from {args.u_csv} "
              f"(missing={miss} → median-filled)  mean={m['U_mean'].mean():.3f} "
              f"std={m['U_mean'].std():.3f}")

    y = np.array(err[args.error], dtype=float)

    # ---------- A0 single-signal sanity (does ANY lone signal predict error?) ----------
    print("\n--- A0 single-signal AUC (lone-feature discrimination of error) ---")
    u_name = f"U({args.u_col})" if args.u_csv else "U(logprob)"
    sigs = {"S_max": m["S_max"].values, u_name: m["U_mean"].values,
            "card(oracle)": m["gold_card"].values,
            "D_dec": np.array(D["D_dec"]), "D_lab": np.array(D["D_lab"])}
    def auc1(x, yv):
        if not HAVE_SK or len(np.unique(yv)) < 2:
            return float("nan")
        try:
            xr = np.asarray(x, dtype=float).reshape(-1, 1)
            clf = LogisticRegression(max_iter=1000).fit(xr, yv)
            return roc_auc_score(yv, clf.predict_proba(xr)[:, 1])
        except Exception:
            return float("nan")
    for e in ["exact_mismatch", "under_imaging", "ct_miss"]:
        ev = np.array(err[e], dtype=float)
        print(f"  [{e:14s}] " + "  ".join(f"{nm}={auc1(x, ev):.3f}" for nm, x in sigs.items()))
    if HAVE_SCIPY:
        print(f"  Spearman(U, card)     = {spearmanr(m['U_mean'].values, m['gold_card'].values)[0]:+.3f}  "
              f"(if ~0, U doesn't even track difficulty)")
        print(f"  Spearman(S_max, card) = {spearmanr(m['S_max'].values, m['gold_card'].values)[0]:+.3f}")

    # ---------- A1 prevalence ----------
    print("\n--- A1 conflict prevalence ---")
    print(f"  D_dec (img/no disagreement among sources): {100*np.mean(D['D_dec']):.1f}% of stays")
    print(f"  D_high (CT presence disagreement):         {100*np.mean(D['D_high']):.1f}%")
    print(f"  D_jac>0 (any set divergence):              {100*np.mean(np.array(D['D_jac'])>0):.1f}%")
    print(f"  D_jac  mean={np.mean(D['D_jac']):.3f}  median={np.median(D['D_jac']):.3f}")
    print(f"  D_lab  mean={np.mean(D['D_lab']):.3f}  (avg fraction of labels in dispute)")

    # ---------- A2 structure ----------
    print("\n--- A2 structure ---")
    for a, b in itertools.combinations(SRC, 2):
        ag = np.mean([jaccard(s[a], s[b]) for s in src_sets_all])
        print(f"  agreement(Jaccard) {a}-{b}: {ag:.3f}")
    # most contested labels
    contest = []
    for L in LABELS:
        dis = np.mean([1.0 if len({int(rec.get(f'{sk}__{L}',0)) for sk in SRC})>1 else 0.0
                       for rec in m.to_dict("records")])
        contest.append((L, dis))
    contest.sort(key=lambda x: -x[1])
    print("  most contested labels:", ", ".join(f"{L}={d:.2f}" for L, d in contest[:6]))
    for sk in SRC:
        acc = np.mean([1.0 if src_set(rec, sk) == {L for L in LABELS if int(rec.get(f'y_{w}__{L}',0) or 0)==1}
                       else 0.0 for rec in m.to_dict("records")])
        print(f"  single-source exact acc {sk}: {acc:.3f}")

    # ---------- A3 conflict ↔ error (all error types) ----------
    ERROR_SET = ["exact_mismatch", "under_imaging", "ct_miss"]
    def a3_line(dk, ev):
        d = np.array(D[dk], dtype=float)
        if set(np.unique(d)) <= {0.0, 1.0}:
            lo = ev[d == 0].mean() if (d == 0).any() else float("nan")
            hi = ev[d == 1].mean() if (d == 1).any() else float("nan")
            s = f"  {dk:6s}: err|0={lo:.3f} err|1={hi:.3f} Δ={hi-lo:+.3f}"
        else:
            med = np.median(d); hi = ev[d > med].mean(); lo = ev[d <= med].mean()
            s = f"  {dk:6s}: err|lo={lo:.3f} err|hi={hi:.3f} Δ={hi-lo:+.3f}"
        if HAVE_SCIPY and len(np.unique(d)) > 1 and len(np.unique(ev)) > 1:
            rho, p = spearmanr(d, ev); s += f"  | ρ={rho:+.3f} p={p:.2e}"
        return s
    for e in ERROR_SET:
        ev = np.array(err[e], dtype=float)
        print(f"\n--- A3 conflict ↔ error  [{e}]  (base rate={ev.mean():.3f}) ---")
        for dk in ["D_jac", "D_lab", "D_dec", "D_high"]:
            print(a3_line(dk, ev))

    # ---------- A4 incremental information ----------
    print(f"\n--- A4 incremental: ΔAUC adding best-conflict over baselines ---")
    print("   baseline S+U = deployable signals only; S+U+card adds gold-difficulty (oracle).")
    def auc_cv(X, yv):
        if not HAVE_SK or len(np.unique(yv)) < 2: return float("nan")
        from sklearn.model_selection import cross_val_predict
        try:
            pr = cross_val_predict(LogisticRegression(max_iter=1000), X, yv,
                                   cv=5, method="predict_proba")[:, 1]
            return roc_auc_score(yv, pr)
        except Exception:
            clf = LogisticRegression(max_iter=1000).fit(X, yv)
            return roc_auc_score(yv, clf.predict_proba(X)[:, 1])
    SU  = np.column_stack([m["S_max"].values, m["U_mean"].values])
    SUc = np.column_stack([SU, m["gold_card"].values])
    a4 = {}
    for e in ERROR_SET:
        ev = np.array(err[e], dtype=float)
        if len(np.unique(ev)) < 2:
            print(f"  [{e}] degenerate (base rate {ev.mean():.3f}) — skip"); continue
        best = max(["D_jac", "D_lab", "D_dec", "D_high"],
                   key=lambda k: abs(spearmanr(np.array(D[k]), ev)[0]) if HAVE_SCIPY else 0.0)
        dvec = np.array(D[best])
        print(f"  [{e}]  best D = {best}")
        d_su = float("nan")
        for lab, base in [("S+U", SU), ("S+U+card", SUc)]:
            ab = auc_cv(base, ev); af = auc_cv(np.column_stack([base, dvec]), ev)
            if lab == "S+U": d_su = af - ab
            print(f"      base {lab:9s}: AUC={ab:.3f}  +{best}={af:.3f}  ΔAUC={af-ab:+.3f}")
        dp = float("nan")
        if HAVE_SM:
            try:
                r = sm.Logit(ev, sm.add_constant(np.column_stack([SU, dvec]))).fit(disp=0)
                dp = float(r.pvalues[-1])
                print(f"      logit {best} coef={r.params[-1]:+.3f} p={dp:.3g} (over S+U)")
            except Exception as ex:
                print(f"      [logit fail: {ex}]")
        a4[e] = {"best": best, "dAUC_SU": d_su, "p": dp}

    # ---------- A5 qualitative ----------
    print("\n--- A5 high-conflict cases (top by D_jac) ---")
    idx = np.argsort(-np.array(D["D_jac"]))[:8]
    recs = m.to_dict("records")
    for j in idx:
        r = recs[j]
        s = {sk: sorted(src_set(r, sk)) for sk in SRC}
        gold = sorted({L for L in LABELS if int(r.get(f"y_{w}__{L}",0) or 0)==1})
        print(f"  stay={r['stay_id']} D_jac={D['D_jac'][j]:.2f} | P={s['P']} W={s['W']} G={s['G']} | gold={gold}")

    # ---------- GO / NO-GO ----------
    print("\n=== GO / NO-GO VERDICT ===")
    clin = [a4[e]["dAUC_SU"] for e in ("ct_miss", "under_imaging") if e in a4]
    clin_ok = any((not np.isnan(v)) and v > 0.010 for v in clin)
    sig_ok = False
    if HAVE_SCIPY:
        for e in ("ct_miss", "under_imaging", "exact_mismatch"):
            ev = np.array(err[e], dtype=float)
            if len(np.unique(ev)) < 2:
                continue
            if any(spearmanr(np.array(D[k]), ev)[0] > 0 and spearmanr(np.array(D[k]), ev)[1] < 0.05
                   for k in D):
                sig_ok = True
    if clin_ok and sig_ok:
        print("  ✅ GO — conflict adds CLINICAL-error signal beyond deployable S+U.")
        print("     → ② headline stands (clinical framing); proceed to Stage A.")
    elif sig_ok:
        print("  ⚠ PARTIAL — conflict relates to error but adds little beyond S+U (ΔAUC small).")
        print("     → Headline to uncertainty (option a), conflict=secondary; OR scale to full n / refine D.")
    else:
        print("  ❌ WEAK — conflict not a reliable independent error signal at n=500.")
        print("     → Lean ③ benchmark / ④ analysis; revisit D definition, top_k, or scale n.")
    print("  (heuristic; read A3/A4 + A5. n=500 underpowered for weak effects.)")


if __name__ == "__main__":
    main()
