# -*- coding: utf-8 -*-
"""M1.2e: cross-validate cohort_v2 vs legacy cohort_imaging_labels_4class.csv.
Reports per-class agreement, confusion matrix, kappa, multi-modality recovery, and disagreement examples."""
import sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import pandas as pd
import numpy as np
import paths


def cohen_kappa(y_true, y_pred):
    """Manual Cohen's kappa (categorical)."""
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    po = (y_true == y_pred).mean()
    # expected agreement
    pe = 0.0
    for L in labels:
        pe += (y_true == L).mean() * (y_pred == L).mean()
    if 1 - pe == 0:
        return 1.0
    return (po - pe) / (1 - pe)


def collapse_v2_to_legacy_4class(label7):
    """v2's 7-class → legacy 4-class (matches legacy builder's mapping)."""
    if label7 in ("ct_head", "ct_abd"):
        return "ct"
    if label7 in ("cxr", "us"):
        return label7
    return "none"  # 'none' / 'other' / 'multi' → 'none' (per legacy collapse)


def main():
    print(f"[ENV={paths.ENV}]")

    # Load v2
    print(f"\n[1] Loading cohort_v2 ({paths.COHORT_V2_CSV})...", flush=True)
    v2 = pd.read_csv(paths.COHORT_V2_CSV)
    v2_cols_needed = ["stay_id", "imaging_choice_v2_full", "imaging_choice_v2_early",
                      "card_full", "card_early", "no_imaging_full", "no_imaging_early",
                      "gold_set_full", "cohort"]
    v2 = v2[v2_cols_needed].copy()
    print(f"    {len(v2):,} rows")

    # Load legacy
    print(f"\n[2] Loading legacy cohort ({paths.LEGACY_COHORT_CSV})...", flush=True)
    legacy = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1", "cp949"]:
        try:
            legacy = pd.read_csv(paths.LEGACY_COHORT_CSV, encoding=enc, low_memory=False)
            print(f"    [encoding ok: {enc}]")
            break
        except UnicodeDecodeError:
            continue
    if legacy is None:
        raise UnicodeDecodeError("Could not decode legacy CSV")
    print(f"    {len(legacy):,} rows; cols(sample)={list(legacy.columns)[:25]}")

    # Derive missing columns from imaging_choice if needed
    VALID_4 = {"none", "cxr", "ct_head", "ct_abd", "us"}
    if "imaging_choice_4" not in legacy.columns:
        legacy["imaging_choice_4"] = legacy["imaging_choice"].apply(collapse_v2_to_legacy_4class)
        print(f"    derived imaging_choice_4 from imaging_choice")
    if "gold4_valid" not in legacy.columns:
        legacy["gold4_valid"] = legacy["imaging_choice"].isin(VALID_4).astype(int)
        print(f"    derived gold4_valid (=1 if imaging_choice in {VALID_4})")
    if "imaging_choice_early" in legacy.columns:
        if "imaging_choice_early_4" not in legacy.columns:
            legacy["imaging_choice_early_4"] = legacy["imaging_choice_early"].apply(collapse_v2_to_legacy_4class)
        if "gold4_early_valid" not in legacy.columns:
            legacy["gold4_early_valid"] = legacy["imaging_choice_early"].isin(VALID_4).astype(int)
    if "y_choice_multi" not in legacy.columns:
        legacy["y_choice_multi"] = (legacy["imaging_choice"] == "multi").astype(int)
        print(f"    derived y_choice_multi from imaging_choice=='multi'")

    legacy_cols = ["stay_id", "imaging_choice", "imaging_choice_early",
                   "imaging_choice_4", "imaging_choice_early_4",
                   "gold4_valid", "gold4_early_valid", "y_choice_multi"]
    legacy_cols = [c for c in legacy_cols if c in legacy.columns]
    legacy = legacy[legacy_cols].copy()

    # Join on stay_id
    print(f"\n[3] Joining v2 × legacy on stay_id...", flush=True)
    m = v2.merge(legacy, on="stay_id", how="inner")
    print(f"    joined: {len(m):,} stays")
    if len(m) == 0:
        raise RuntimeError("Empty join — stay_id mismatch")

    L = []
    L.append("=" * 70)
    L.append(f"CROSS-VALIDATE REPORT — cohort_v2 vs legacy")
    L.append("=" * 70)
    L.append(f"v2 rows: {len(v2):,}")
    L.append(f"legacy rows: {len(legacy):,}")
    L.append(f"joined: {len(m):,}")

    # 4a. 7-class agreement (full window)
    L.append("\n" + "-" * 70)
    L.append("[A] 7-class label agreement (FULL window)")
    L.append("-" * 70)
    agree_full7 = (m["imaging_choice_v2_full"] == m["imaging_choice"]).mean()
    kappa_full7 = cohen_kappa(m["imaging_choice"], m["imaging_choice_v2_full"])
    L.append(f"agreement: {agree_full7:.4f} ({100*agree_full7:.2f}%)")
    L.append(f"Cohen's κ:  {kappa_full7:.4f}")

    # Confusion matrix (legacy × v2)
    L.append("\nConfusion matrix (rows=legacy, cols=v2_full):")
    cm = pd.crosstab(m["imaging_choice"], m["imaging_choice_v2_full"], dropna=False, margins=True)
    L.append(cm.to_string())

    # Per-label breakdown
    L.append("\nPer-label sensitivity (legacy → v2 same label):")
    for lab in sorted(m["imaging_choice"].unique()):
        sub = m[m["imaging_choice"] == lab]
        match = (sub["imaging_choice_v2_full"] == lab).sum()
        L.append(f"  legacy={lab:<10s}  n={len(sub):>5}  v2 same: {match:>5} ({100*match/max(1,len(sub)):.1f}%)")
        # Show top v2 distributions for this legacy class
        top_v2 = sub["imaging_choice_v2_full"].value_counts().head(3)
        for v, c in top_v2.items():
            L.append(f"    └ v2={v:<10s}  {c:>5} ({100*c/len(sub):.1f}%)")

    # 4b. 7-class agreement (early window)
    L.append("\n" + "-" * 70)
    L.append("[B] 7-class label agreement (EARLY 2h window)")
    L.append("-" * 70)
    agree_early7 = (m["imaging_choice_v2_early"] == m["imaging_choice_early"]).mean()
    kappa_early7 = cohen_kappa(m["imaging_choice_early"], m["imaging_choice_v2_early"])
    L.append(f"agreement: {agree_early7:.4f} ({100*agree_early7:.2f}%)")
    L.append(f"Cohen's κ:  {kappa_early7:.4f}")
    L.append("\nConfusion matrix (rows=legacy, cols=v2_early):")
    cm_e = pd.crosstab(m["imaging_choice_early"], m["imaging_choice_v2_early"], dropna=False, margins=True)
    L.append(cm_e.to_string())

    # 5. 4-class agreement (within gold4_valid=1 scope)
    L.append("\n" + "-" * 70)
    L.append("[C] 4-class agreement (legacy gold4_valid=1 scope only)")
    L.append("-" * 70)
    m["v2_4_full"] = m["imaging_choice_v2_full"].apply(collapse_v2_to_legacy_4class)
    m["v2_4_early"] = m["imaging_choice_v2_early"].apply(collapse_v2_to_legacy_4class)
    valid_full = m[m["gold4_valid"] == 1]
    if len(valid_full) > 0:
        agree_full4 = (valid_full["v2_4_full"] == valid_full["imaging_choice_4"]).mean()
        kappa_full4 = cohen_kappa(valid_full["imaging_choice_4"], valid_full["v2_4_full"])
        L.append(f"scope: gold4_valid=1, n={len(valid_full):,}")
        L.append(f"FULL agreement: {agree_full4:.4f} ({100*agree_full4:.2f}%)")
        L.append(f"FULL Cohen's κ: {kappa_full4:.4f}")
        L.append("\nConfusion (rows=legacy_4, cols=v2_4_full):")
        cm4 = pd.crosstab(valid_full["imaging_choice_4"], valid_full["v2_4_full"], dropna=False, margins=True)
        L.append(cm4.to_string())
    if "gold4_early_valid" in m.columns:
        valid_early = m[m["gold4_early_valid"] == 1]
        if len(valid_early) > 0:
            agree_early4 = (valid_early["v2_4_early"] == valid_early["imaging_choice_early_4"]).mean()
            kappa_early4 = cohen_kappa(valid_early["imaging_choice_early_4"], valid_early["v2_4_early"])
            L.append(f"\nEARLY scope: gold4_early_valid=1, n={len(valid_early):,}")
            L.append(f"EARLY agreement: {agree_early4:.4f} ({100*agree_early4:.2f}%)")
            L.append(f"EARLY Cohen's κ: {kappa_early4:.4f}")

    # 6. Multi-modality recovery
    L.append("\n" + "-" * 70)
    L.append("[D] Multi-modality recovery")
    L.append("-" * 70)
    m["legacy_is_multi"] = (m.get("y_choice_multi", 0) == 1).astype(int)
    m["v2_is_multi"] = (m["card_full"] >= 2).astype(int)
    legacy_multi_n = m["legacy_is_multi"].sum()
    v2_multi_n = m["v2_is_multi"].sum()
    overlap = ((m["legacy_is_multi"] == 1) & (m["v2_is_multi"] == 1)).sum()
    legacy_only = ((m["legacy_is_multi"] == 1) & (m["v2_is_multi"] == 0)).sum()
    v2_only = ((m["legacy_is_multi"] == 0) & (m["v2_is_multi"] == 1)).sum()
    both_no = ((m["legacy_is_multi"] == 0) & (m["v2_is_multi"] == 0)).sum()
    L.append(f"legacy multi (y_choice_multi=1): {legacy_multi_n:,}")
    L.append(f"v2 multi (card_full>=2):         {v2_multi_n:,}")
    L.append(f"both multi:                      {overlap:,}")
    L.append(f"legacy-only multi:               {legacy_only:,}")
    L.append(f"v2-only multi:                   {v2_only:,}")
    L.append(f"both single/none:                {both_no:,}")
    if legacy_multi_n > 0:
        L.append(f"\nRecovery rate (legacy multi → v2 multi): {overlap}/{legacy_multi_n} = {100*overlap/legacy_multi_n:.1f}%")

    # When v2 says multi but legacy didn't:
    if v2_only > 0:
        v2_only_df = m[(m["legacy_is_multi"] == 0) & (m["v2_is_multi"] == 1)]
        L.append(f"\nv2-only multi: top legacy 7-class labels (these were collapsed to single by legacy):")
        L.append(v2_only_df["imaging_choice"].value_counts().head(7).to_string())

    # 7. Disagreement sample dump (for inspection)
    print(f"\n[4] Sampling disagreement examples...", flush=True)
    disagree = m[m["imaging_choice_v2_full"] != m["imaging_choice"]].copy()
    L.append("\n" + "-" * 70)
    L.append(f"[E] Disagreement summary")
    L.append("-" * 70)
    L.append(f"total disagreements: {len(disagree):,}")
    L.append(f"\nDisagreement patterns (legacy → v2_full, top 20):")
    patterns = disagree.groupby(["imaging_choice", "imaging_choice_v2_full"]).size().sort_values(ascending=False).head(20)
    L.append(patterns.to_string())

    # Save disagreement CSV (sample of cases for manual review)
    sample_dis = disagree[["stay_id", "imaging_choice", "imaging_choice_v2_full",
                           "card_full", "gold_set_full", "cohort"]].copy()
    # Take a stratified-ish sample: up to 200 per disagreement pattern
    sample_dis.to_csv(paths.CROSS_VALIDATE_DISAGREE_CSV, index=False)
    L.append(f"\nSaved disagreement details: {paths.CROSS_VALIDATE_DISAGREE_CSV}")

    # Save report
    report = "\n".join(L)
    with open(paths.CROSS_VALIDATE_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nSaved: {paths.CROSS_VALIDATE_REPORT_TXT}")


if __name__ == "__main__":
    main()
