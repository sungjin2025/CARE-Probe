#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_v2.py — Evaluation for imaging_runner_v2 predictions vs cohort_v2 gold.

Metrics:
  - Decision distribution + abstain reasons
  - Per-label P/R/F1 (TP/FP/FN), macro/micro F1
  - Set-level: mean Jaccard, exact set match accuracy
  - no_imaging: P/R/F1
  - Clinical: over-imaging, under-imaging, CT over-imaging, CT recall
  - Cardinality distributions (gold vs pred)
  - Per-cohort breakdown

Usage:
  python eval_v2.py --pred_csv pred_msrc_qwen14b_seed42.csv \\
                    --cohort_csv cohort_v2.csv \\
                    --window full \\
                    --tag msrc_qwen14b_seed42
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
import numpy as np

import paths
from policyA_decode_v2 import LABEL_SPACE_V2, NO_IMAGING_LABEL


# ====================================================================
# Helpers
# ====================================================================
def parse_gold_set(s):
    """Parse JSON-serialized gold set from cohort_v2.csv (e.g. '[\"CT_HEAD\",\"CXR_CHEST\"]')."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


def parse_pred_set(s):
    """Parse semicolon-separated pred_label_set from runner output."""
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]


def prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    r = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / max(p + r, 1e-12) if (p + r) > 0 else 0.0
    return p, r, f1


def jaccard(s1, s2) -> float:
    S1, S2 = set(s1), set(s2)
    if not S1 and not S2: return 1.0
    if not S1 or not S2: return 0.0
    return len(S1 & S2) / len(S1 | S2)


# ====================================================================
# Core eval
# ====================================================================
def evaluate(pred_df: pd.DataFrame, cohort_df: pd.DataFrame, window: str = "full") -> dict:
    """Compute metrics. Returns dict (JSON-serializable)."""
    gold_col      = f"gold_set_{window}"
    no_img_col    = f"no_imaging_{window}"
    legacy_col    = f"imaging_choice_v2_{window}"
    needed = ["stay_id", "cohort", gold_col, no_img_col, legacy_col, "card_full", "card_early"]
    needed = [c for c in needed if c in cohort_df.columns]
    cohort_min = cohort_df[needed].copy()

    m = pred_df.merge(cohort_min, on="stay_id", how="inner")
    n_total = len(m)
    if n_total == 0:
        raise ValueError("Empty join — pred_csv and cohort_csv have no matching stay_ids.")

    # Parse sets
    m["gold_set"]        = m[gold_col].apply(parse_gold_set)
    m["pred_set"]        = m["pred_label_set"].apply(parse_pred_set)
    m["gold_no_imaging"] = m[no_img_col].fillna(0).astype(int)
    m["gold_card"]       = m["gold_set"].apply(len)
    m["pred_card"]       = m["pred_set"].apply(len)

    # Decision distribution
    n_commit_img = int((m["decision"] == "commit_imaging").sum())
    n_commit_no  = int((m["decision"] == "commit_no_imaging").sum())
    n_abstain    = int((m["decision"] == "abstain").sum())
    commit_rate  = (n_commit_img + n_commit_no) / max(n_total, 1)
    abstain_rate = n_abstain / max(n_total, 1)

    # Per-label metrics
    label_metrics = {}
    for L in LABEL_SPACE_V2:
        gold_has = m["gold_set"].apply(lambda s: L in s)
        pred_has = m["pred_set"].apply(lambda s: L in s)
        tp = int((gold_has & pred_has).sum())
        fp = int((~gold_has & pred_has).sum())
        fn = int((gold_has & ~pred_has).sum())
        p, r, f1 = prf(tp, fp, fn)
        label_metrics[L] = {"tp": tp, "fp": fp, "fn": fn,
                            "p": float(p), "r": float(r), "f1": float(f1)}

    macro_p  = float(np.mean([v["p"] for v in label_metrics.values()]))
    macro_r  = float(np.mean([v["r"] for v in label_metrics.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in label_metrics.values()]))
    tot_tp   = sum(v["tp"] for v in label_metrics.values())
    tot_fp   = sum(v["fp"] for v in label_metrics.values())
    tot_fn   = sum(v["fn"] for v in label_metrics.values())
    micro_p, micro_r, micro_f1 = prf(tot_tp, tot_fp, tot_fn)

    # Set-level
    m["jaccard"]     = m.apply(lambda r: jaccard(r["gold_set"], r["pred_set"]), axis=1)
    m["exact_match"] = m.apply(lambda r: set(r["gold_set"]) == set(r["pred_set"]), axis=1)
    mean_jaccard     = float(m["jaccard"].mean())
    exact_match_rate = float(m["exact_match"].mean())

    # no_imaging metrics (predicted via commit_no_imaging)
    pred_no = (m["decision"] == "commit_no_imaging").astype(int)
    tp_no = int(((m["gold_no_imaging"] == 1) & (pred_no == 1)).sum())
    fp_no = int(((m["gold_no_imaging"] == 0) & (pred_no == 1)).sum())
    fn_no = int(((m["gold_no_imaging"] == 1) & (pred_no == 0)).sum())
    p_no, r_no, f1_no = prf(tp_no, fp_no, fn_no)

    # Clinical
    over_imaging  = int(((m["gold_no_imaging"] == 1) & (m["decision"] == "commit_imaging")).sum())
    under_imaging = int(((m["gold_no_imaging"] == 0) &
                         ((m["decision"] == "commit_no_imaging") | (m["decision"] == "abstain"))).sum())
    ct_labels = [L for L in LABEL_SPACE_V2 if L.startswith("CT_")]
    has_ct_pred = m["pred_set"].apply(lambda s: any(L in s for L in ct_labels))
    has_ct_gold = m["gold_set"].apply(lambda s: any(L in s for L in ct_labels))
    ct_over = int(((m["gold_no_imaging"] == 1) & has_ct_pred).sum())
    ct_recall = float(((has_ct_gold) & (has_ct_pred)).sum() / max(int(has_ct_gold.sum()), 1))
    ct_over_imaging_rate = ct_over / max(n_total, 1)

    # Per-cohort
    by_cohort = {}
    if "cohort" in m.columns:
        for c in sorted(m["cohort"].dropna().unique()):
            sub = m[m["cohort"] == c]
            by_cohort[c] = {
                "n":           int(len(sub)),
                "jaccard":     float(sub["jaccard"].mean()),
                "exact_match": float(sub["exact_match"].mean()),
                "abstain_rate": float((sub["decision"] == "abstain").mean()),
                "commit_img_rate": float((sub["decision"] == "commit_imaging").mean()),
            }

    return {
        "n_total": n_total,
        "decision_dist": {
            "commit_imaging": n_commit_img,
            "commit_no_imaging": n_commit_no,
            "abstain": n_abstain,
        },
        "commit_rate":   float(commit_rate),
        "abstain_rate":  float(abstain_rate),
        "abstain_reasons": m.loc[m["decision"] == "abstain", "abstain_reason"]
                            .value_counts().to_dict() if n_abstain > 0 else {},
        "per_label":     label_metrics,
        "macro":         {"p": macro_p, "r": macro_r, "f1": macro_f1},
        "micro":         {"p": float(micro_p), "r": float(micro_r), "f1": float(micro_f1)},
        "mean_jaccard":      mean_jaccard,
        "exact_set_match":   exact_match_rate,
        "no_imaging":    {"tp": tp_no, "fp": fp_no, "fn": fn_no,
                          "p": float(p_no), "r": float(r_no), "f1": float(f1_no)},
        "clinical": {
            "over_imaging":      over_imaging,
            "under_imaging":     under_imaging,
            "ct_over_imaging":   ct_over,
            "ct_recall":         ct_recall,
            "ct_over_imaging_rate": ct_over_imaging_rate,
        },
        "gold_card_dist":   m["gold_card"].value_counts().sort_index().to_dict(),
        "pred_card_dist":   m["pred_card"].value_counts().sort_index().to_dict(),
        "by_cohort":        by_cohort,
        "window":           window,
    }


# ====================================================================
# Report formatter
# ====================================================================
def format_report(metrics: dict, tag: str = "") -> str:
    L = []
    L.append("=" * 78)
    L.append(f"EVAL REPORT  tag={tag}  window={metrics.get('window','?')}  "
             f"n={metrics['n_total']:,}")
    L.append("=" * 78)

    L.append("\n[Decision distribution]")
    for d, c in metrics["decision_dist"].items():
        pct = 100 * c / max(metrics["n_total"], 1)
        L.append(f"  {d:<22s} {c:>8,}  ({pct:5.2f}%)")
    L.append(f"  commit rate: {metrics['commit_rate']*100:.2f}%   "
             f"abstain rate: {metrics['abstain_rate']*100:.2f}%")

    if metrics.get("abstain_reasons"):
        L.append("\n[Abstain reasons]")
        for r, c in sorted(metrics["abstain_reasons"].items(), key=lambda x: -x[1])[:12]:
            L.append(f"  {r:<28s} {c:>6}")

    L.append("\n[Per-label P/R/F1]")
    L.append(f"  {'label':<18s}  {'TP':>5s} {'FP':>5s} {'FN':>5s}    {'P':>6s} {'R':>6s} {'F1':>6s}")
    label_sorted = sorted(metrics["per_label"].items(), key=lambda x: -x[1]["tp"])
    for lab, v in label_sorted:
        L.append(f"  {lab:<18s}  {v['tp']:>5} {v['fp']:>5} {v['fn']:>5}    "
                 f"{v['p']:.3f} {v['r']:.3f} {v['f1']:.3f}")

    L.append("")
    L.append(f"  Macro  P={metrics['macro']['p']:.3f}  R={metrics['macro']['r']:.3f}  "
             f"F1={metrics['macro']['f1']:.3f}")
    L.append(f"  Micro  P={metrics['micro']['p']:.3f}  R={metrics['micro']['r']:.3f}  "
             f"F1={metrics['micro']['f1']:.3f}")

    L.append("\n[Set-level]")
    L.append(f"  Mean Jaccard:      {metrics['mean_jaccard']:.4f}")
    L.append(f"  Exact set match:   {metrics['exact_set_match']:.4f}")

    L.append("\n[no_imaging]")
    ni = metrics["no_imaging"]
    L.append(f"  TP={ni['tp']:>5} FP={ni['fp']:>5} FN={ni['fn']:>5}    "
             f"P={ni['p']:.3f} R={ni['r']:.3f} F1={ni['f1']:.3f}")

    L.append("\n[Clinical]")
    cl = metrics["clinical"]
    L.append(f"  Over-imaging:     {cl['over_imaging']:,}  ({100*cl['over_imaging']/metrics['n_total']:.2f}%)")
    L.append(f"  Under-imaging:    {cl['under_imaging']:,}  ({100*cl['under_imaging']/metrics['n_total']:.2f}%)")
    L.append(f"  CT over-imaging:  {cl['ct_over_imaging']:,}  ({100*cl['ct_over_imaging_rate']:.2f}%)")
    L.append(f"  CT recall:        {cl['ct_recall']:.4f}")

    L.append("\n[Cardinality distribution]")
    L.append("  gold (cohort_v2):")
    for k, v in sorted(metrics["gold_card_dist"].items()):
        L.append(f"    card={k}: {v:,}")
    L.append("  pred (runner):")
    for k, v in sorted(metrics["pred_card_dist"].items()):
        L.append(f"    card={k}: {v:,}")

    if metrics.get("by_cohort"):
        L.append("\n[Per-cohort summary]")
        L.append(f"  {'cohort':<20s}  {'n':>6s}  {'Jaccard':>8s}  {'Exact':>8s}  "
                 f"{'Abstain':>8s}  {'CommitImg':>10s}")
        for c, v in sorted(metrics["by_cohort"].items(), key=lambda x: -x[1]["n"]):
            L.append(f"  {c:<20s}  {v['n']:>6}  {v['jaccard']:>8.3f}  "
                     f"{v['exact_match']:>8.3f}  {v['abstain_rate']:>8.3f}  "
                     f"{v['commit_img_rate']:>10.3f}")

    return "\n".join(L)


# ====================================================================
# CLI
# ====================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="M1.5 eval: predictions vs cohort_v2 gold")
    ap.add_argument("--pred_csv", required=True, help="runner output CSV")
    ap.add_argument("--cohort_csv", default=paths.COHORT_V2_CSV)
    ap.add_argument("--window", choices=["full", "early"], default="full",
                    help="gold window: full (ED stay) or early (2h)")
    ap.add_argument("--out_dir", default=str(paths.WORK_DIR))
    ap.add_argument("--tag", default="eval")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[ENV={paths.ENV}]")
    print(f"  pred:   {args.pred_csv}")
    print(f"  cohort: {args.cohort_csv}")
    print(f"  window: {args.window}")
    print(f"  tag:    {args.tag}\n")

    pred = pd.read_csv(args.pred_csv)
    cohort = pd.read_csv(args.cohort_csv)
    print(f"Loaded: pred={len(pred):,} rows, cohort={len(cohort):,} rows")

    metrics = evaluate(pred, cohort, window=args.window)
    report = format_report(metrics, tag=args.tag)
    print("\n" + report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.tag}_metrics.json"
    txt_path  = out_dir / f"{args.tag}_report.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


# ====================================================================
# Internal tests
# ====================================================================
def _test():
    """Smoke test: build a tiny synthetic pred + cohort, verify metrics math."""
    passed = failed = 0
    def check(name, actual, expected, tol=1e-6):
        nonlocal passed, failed
        if isinstance(expected, float):
            ok = abs(float(actual) - expected) < tol
        else:
            ok = actual == expected
        if ok: passed += 1
        else:
            failed += 1
            print(f"FAIL: {name}  expected={expected!r}  actual={actual!r}")

    # Synthetic cohort: 4 stays
    cohort = pd.DataFrame([
        {"stay_id": 1, "cohort": "abdominal_pain", "gold_set_full": json.dumps(["CT_HEAD"]),
         "no_imaging_full": 0, "imaging_choice_v2_full": "ct_head",
         "card_full": 1, "card_early": 0,
         "gold_set_early": json.dumps([]), "no_imaging_early": 1,
         "imaging_choice_v2_early": "none"},
        {"stay_id": 2, "cohort": "chest_pain", "gold_set_full": json.dumps(["CXR_CHEST", "CT_CHEST"]),
         "no_imaging_full": 0, "imaging_choice_v2_full": "multi",
         "card_full": 2, "card_early": 0,
         "gold_set_early": json.dumps([]), "no_imaging_early": 1,
         "imaging_choice_v2_early": "none"},
        {"stay_id": 3, "cohort": "other", "gold_set_full": json.dumps([]),
         "no_imaging_full": 1, "imaging_choice_v2_full": "none",
         "card_full": 0, "card_early": 0,
         "gold_set_early": json.dumps([]), "no_imaging_early": 1,
         "imaging_choice_v2_early": "none"},
        {"stay_id": 4, "cohort": "other", "gold_set_full": json.dumps(["US_ABDOMEN"]),
         "no_imaging_full": 0, "imaging_choice_v2_full": "us",
         "card_full": 1, "card_early": 0,
         "gold_set_early": json.dumps([]), "no_imaging_early": 1,
         "imaging_choice_v2_early": "none"},
    ])

    # Perfect-prediction case
    pred = pd.DataFrame([
        {"stay_id": 1, "decision": "commit_imaging", "pred_label_set": "CT_HEAD",
         "abstain_reason": "", "is_commit_imaging": 1, "is_commit_no_imaging": 0,
         "is_abstain": 0, "pred_no_imaging": 0, "is_multi_pred": 0},
        {"stay_id": 2, "decision": "commit_imaging", "pred_label_set": "CXR_CHEST;CT_CHEST",
         "abstain_reason": "", "is_commit_imaging": 1, "is_commit_no_imaging": 0,
         "is_abstain": 0, "pred_no_imaging": 0, "is_multi_pred": 1},
        {"stay_id": 3, "decision": "commit_no_imaging", "pred_label_set": "",
         "abstain_reason": "", "is_commit_imaging": 0, "is_commit_no_imaging": 1,
         "is_abstain": 0, "pred_no_imaging": 1, "is_multi_pred": 0},
        {"stay_id": 4, "decision": "commit_imaging", "pred_label_set": "US_ABDOMEN",
         "abstain_reason": "", "is_commit_imaging": 1, "is_commit_no_imaging": 0,
         "is_abstain": 0, "pred_no_imaging": 0, "is_multi_pred": 0},
    ])

    M = evaluate(pred, cohort, window="full")
    check("perfect: n_total", M["n_total"], 4)
    check("perfect: commit_imaging count", M["decision_dist"]["commit_imaging"], 3)
    check("perfect: commit_no_imaging count", M["decision_dist"]["commit_no_imaging"], 1)
    check("perfect: abstain count", M["decision_dist"]["abstain"], 0)
    check("perfect: mean Jaccard = 1.0", M["mean_jaccard"], 1.0)
    check("perfect: exact set match = 1.0", M["exact_set_match"], 1.0)
    check("perfect: CT_HEAD F1 = 1.0", M["per_label"]["CT_HEAD"]["f1"], 1.0)
    check("perfect: no_imaging F1 = 1.0", M["no_imaging"]["f1"], 1.0)
    check("perfect: macro F1 (only used labels)", M["macro"]["f1"] > 0.0, True)
    check("perfect: over-imaging = 0", M["clinical"]["over_imaging"], 0)
    check("perfect: under-imaging = 0", M["clinical"]["under_imaging"], 0)
    check("perfect: CT recall = 1.0", M["clinical"]["ct_recall"], 1.0)

    # Imperfect case: stay 4 predicts CT_HEAD instead of US_ABDOMEN
    pred_bad = pred.copy()
    pred_bad.loc[pred_bad["stay_id"] == 4, "pred_label_set"] = "CT_HEAD"
    M2 = evaluate(pred_bad, cohort, window="full")
    check("bad: stay 4 jaccard = 0",
          M2["per_label"]["US_ABDOMEN"]["fn"], 1)  # US_ABDOMEN missed once
    check("bad: CT_HEAD has extra FP",
          M2["per_label"]["CT_HEAD"]["fp"], 1)
    check("bad: exact match < 1.0", M2["exact_set_match"] < 1.0, True)

    # Abstain case
    pred_abs = pred.copy()
    pred_abs.loc[pred_abs["stay_id"] == 1, "decision"] = "abstain"
    pred_abs.loc[pred_abs["stay_id"] == 1, "abstain_reason"] = "ambiguous"
    pred_abs.loc[pred_abs["stay_id"] == 1, "pred_label_set"] = ""
    pred_abs.loc[pred_abs["stay_id"] == 1, "is_commit_imaging"] = 0
    pred_abs.loc[pred_abs["stay_id"] == 1, "is_abstain"] = 1
    M3 = evaluate(pred_abs, cohort, window="full")
    check("abstain: count = 1", M3["decision_dist"]["abstain"], 1)
    check("abstain: abstain_rate = 0.25", M3["abstain_rate"], 0.25)
    check("abstain: under_imaging counts stay 1",
          M3["clinical"]["under_imaging"], 1)
    check("abstain: reasons contain 'ambiguous'",
          "ambiguous" in M3["abstain_reasons"], True)

    # Over-imaging case
    pred_over = pred.copy()
    pred_over.loc[pred_over["stay_id"] == 3, "decision"] = "commit_imaging"
    pred_over.loc[pred_over["stay_id"] == 3, "pred_label_set"] = "CT_HEAD"
    pred_over.loc[pred_over["stay_id"] == 3, "is_commit_imaging"] = 1
    pred_over.loc[pred_over["stay_id"] == 3, "is_commit_no_imaging"] = 0
    pred_over.loc[pred_over["stay_id"] == 3, "pred_no_imaging"] = 0
    M4 = evaluate(pred_over, cohort, window="full")
    check("over: over-imaging = 1", M4["clinical"]["over_imaging"], 1)
    check("over: CT over-imaging = 1", M4["clinical"]["ct_over_imaging"], 1)

    print(f"\n=== Tests: {passed}/{passed+failed} passed ===")
    if failed > 0:
        print(f"[FAIL] {failed} tests failed")
        return False
    print("[OK] All tests passed")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        ok = _test()
        sys.exit(0 if ok else 1)
    main()
