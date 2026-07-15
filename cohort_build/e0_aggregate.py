#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0_aggregate.py — aggregate run_main_experiment.sh results into mean +/- 95% CI tables.

Reads results_main.csv (one row per model x subsample_seed x split_seed x mode),
groups by (model, mode), and reports mean +/- 95% CI over all seeds for each metric.
=> the multi-seed table that answers "no seed repetition" reviewer concern.

Usage:
  python e0_aggregate.py --results_csv runs_main/results_main.csv
"""
import argparse
import numpy as np
import pandas as pd


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = x.mean()
    h = 1.96 * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, h, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--order", default="qwen14b llama31-8b gemma2-9b mistral7b-v03 "
                    "med42-8b openbio-8b phi3mini-128k mediphi")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    # tag = "model|subSEED|mode"
    parts = df["tag"].astype(str).str.split("|", expand=True)
    df["model"] = parts[0]
    df["sub"] = parts[1] if parts.shape[1] > 1 else "sub42"
    df["mode"] = (parts[2] if parts.shape[1] > 2 else "main").fillna("main")

    order = [m for m in args.order.split() if m in set(df["model"])]
    order += [m for m in df["model"].unique() if m not in order]

    metrics = [("gap_closed", "gap%(R→oracle)", 100.0),
               ("joint_risk", "joint_risk", 1.0),
               ("commit", "commit", 1.0),
               ("cond_risk", "cond_risk", 1.0)]

    for mode in ["main", "shift"]:
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue
        alpha = sub["alpha"].iloc[0]
        print(f"\n=== mode={mode}  (loss={sub['loss'].iloc[0]}, alpha={alpha}, "
              f"n_seeds/model up to {sub.groupby('model').size().max()}) ===")
        print(f"  {'model':16s} {'gap%(mean±CI)':>18s} {'joint_risk':>16s} "
              f"{'commit':>14s} {'cond_risk':>14s}  {'≤α?':>5s}")
        for model in order:
            g = sub[sub["model"] == model]
            if len(g) == 0:
                continue
            cells = []
            jr_mean = None
            for col, _, scale in metrics:
                m, h, n = ci95(g[col])
                cells.append(f"{scale*m:6.3f}±{scale*h:5.3f}")
                if col == "joint_risk":
                    jr_mean = m
            ok = "OK" if (jr_mean is not None and jr_mean <= alpha + 1e-9) else "VIOL"
            print(f"  {model:16s} {cells[0]:>18s} {cells[1]:>16s} {cells[2]:>14s} "
                  f"{cells[3]:>14s}  {ok:>5s}")

    # overall guarantee validity (fraction of all runs with joint_risk <= alpha)
    ok_frac = float((df["joint_risk"] <= df["alpha"] + 1e-9).mean())
    print(f"\n  guarantee held in {100*ok_frac:.1f}% of all {len(df)} runs "
          f"(target: >= {100*(1-df['delta'].iloc[0]):.0f}%).")
    print("  → gap%: higher=better (probe closes random→oracle gap). joint_risk ≤ α = guarantee.")


if __name__ == "__main__":
    main()
