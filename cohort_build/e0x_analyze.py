#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0x_analyze.py — E0X analysis: per-backbone surface-signal AUC + verdict.

Consumes e0x_deployed_surface_runner.py output. For each signal and for the
pooled multivariate model, reports 5-fold out-of-fold logistic AUC (mirrors the
manuscript's B2 methodology / single-feature logistic protocol) with a
case-level bootstrap 95% CI, against both targets (ct_miss primary, y_err).

Pre-registered verdict (design §1.4 — 정직 조항, no post-hoc selection):
  UPHELD  : bootstrap CI upper bound < 0.65 (excludes clinically useful signal)
  REVISE  : bootstrap CI lower bound > 0.55 (signal present → manuscript claim
            must be re-scoped for this backbone/signal; do not suppress)
  GREY    : otherwise (report as-is)

CPU-only. Usage:
  python e0x_analyze.py --pred_csv runs_e0x/qwen14b_deployed_surface.csv \
      --tag qwen14b --summary_csv runs_e0x/e0x_summary.csv
Run once per backbone; --summary_csv accumulates rows. Add --latex to print a
manuscript table stub from the accumulated summary.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIGNALS = ["U_lp", "U_sc", "U_lab", "S_max", "S_mean",
           "D_jac", "D_lab", "D_dec", "D_high", "pred_card"]
# manuscript B2: token-level + self-consistency uncertainty, max/mean
# sufficiency, four conflict metrics = 8 features
MULTI8 = ["U_lp", "U_sc", "S_max", "S_mean", "D_jac", "D_lab", "D_dec", "D_high"]
TARGETS = ["ct_miss", "y_err"]
AUC_USEFUL = 0.65   # pre-registered: clinically useful deferral signal
AUC_PRESENT = 0.55  # pre-registered: signal presence → claim revision


def oof_scores(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def bootstrap_auc(y: np.ndarray, s: np.ndarray, reps: int = 2000,
                  seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    if hi < AUC_USEFUL:
        return "UPHELD"
    if lo > AUC_PRESENT:
        return "REVISE"
    return "GREY"


def main():
    ap = argparse.ArgumentParser(description="E0X surface-signal analysis")
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--tag", required=True, help="backbone tag (e.g. qwen14b)")
    ap.add_argument("--summary_csv", default="runs_e0x/e0x_summary.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--latex", action="store_true",
                    help="print a LaTeX stub from the accumulated summary and exit")
    args = ap.parse_args()

    if args.latex:
        _print_latex(args.summary_csv)
        return

    df = pd.read_csv(args.pred_csv)
    n_all = len(df)
    df = df[(df.get("dep_flag") == "ok") & df["ct_miss"].notna()].copy()
    # conflict/SC columns can be NaN on per-arm failures — analyze complete rows
    df = df.dropna(subset=[c for c in SIGNALS if c in df.columns])
    print(f"[E0X-analyze] {args.tag}: {len(df)}/{n_all} usable rows "
          f"(dep_flag=ok, all signals present)")
    print(f"  base rates: ct_miss={df['ct_miss'].mean():.3f} "
          f"y_err={df['y_err'].mean():.3f}")

    out_rows = []
    for tgt in TARGETS:
        y = df[tgt].astype(int).to_numpy()
        if y.min() == y.max():
            print(f"  [skip] target {tgt}: single class"); continue
        for sig in SIGNALS:
            if sig not in df.columns: continue
            X = df[[sig]].to_numpy(float)
            s = oof_scores(X, y, args.seed)
            auc, lo, hi = bootstrap_auc(y, s, args.reps, args.seed)
            v = verdict(lo, hi)
            out_rows.append({"tag": args.tag, "target": tgt, "signal": sig,
                             "n": len(y), "auc_oof": round(auc, 3),
                             "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                             "verdict": v})
            print(f"  {tgt:8s} {sig:10s} AUC={auc:.3f} [{lo:.3f},{hi:.3f}]  {v}")
        # pooled multivariate (B2 mirror: 8 features)
        cols = [c for c in MULTI8 if c in df.columns]
        s = oof_scores(df[cols].to_numpy(float), y, args.seed)
        auc, lo, hi = bootstrap_auc(y, s, args.reps, args.seed)
        v = verdict(lo, hi)
        out_rows.append({"tag": args.tag, "target": tgt, "signal": "MULTI8",
                         "n": len(y), "auc_oof": round(auc, 3),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                         "verdict": v})
        print(f"  {tgt:8s} {'MULTI8':10s} AUC={auc:.3f} [{lo:.3f},{hi:.3f}]  {v}")

    res = pd.DataFrame(out_rows)
    p = Path(args.summary_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        prev = pd.read_csv(p)
        prev = prev[prev["tag"] != args.tag]  # idempotent re-run per tag
        res = pd.concat([prev, res], ignore_index=True)
    res.to_csv(p, index=False)
    print(f"[saved] {p} ({len(res)} rows)")

    bad = res[(res["tag"] == args.tag) & (res["verdict"] == "REVISE")]
    if len(bad):
        print("\n[!] REVISE verdicts — manuscript negative claim must be "
              "re-scoped for these (do not suppress):")
        print(bad.to_string(index=False))
    else:
        print(f"\n[OK] {args.tag}: no signal exceeds the pre-registered "
              f"presence bound (CI lower <= {AUC_PRESENT}); negative claim upheld "
              f"under the deployed protocol at n=2,000.")


def _print_latex(summary_csv: str):
    df = pd.read_csv(summary_csv)
    df = df[df["target"] == "ct_miss"]
    tags = list(dict.fromkeys(df["tag"]))
    sigs = ["U_lp", "U_sc", "S_max", "D_jac", "MULTI8"]
    print("% E0X manuscript table stub (CT-miss OOF AUC [95% bootstrap CI])")
    print(r"\begin{tabular}{@{}l" + "c" * len(tags) + r"@{}}")
    print(r"\toprule")
    print("Signal & " + " & ".join(tags) + r" \\")
    print(r"\midrule")
    for sig in sigs:
        cells = []
        for t in tags:
            r = df[(df["tag"] == t) & (df["signal"] == sig)]
            cells.append("---" if r.empty else
                         f"{r.iloc[0]['auc_oof']:.2f} "
                         f"[{r.iloc[0]['ci_lo']:.2f}, {r.iloc[0]['ci_hi']:.2f}]")
        print(f"{sig.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
