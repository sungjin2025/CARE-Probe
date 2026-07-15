# -*- coding: utf-8 -*-
"""M1.2d: build cohort_v2.csv with multi-label gold from imaging_events_v2.
One row per stay; gold = SET of (modality, body_part) tuples; no_imaging mutex; legacy 7-class for cross-validate."""
import re, json, time
from collections import Counter
import numpy as np
import pandas as pd
import paths

EARLY_HOURS = 2.0
TOP_N_LABELS = 15

# --- Cohort rules (V3 latest, 5 chief complaints + other) ---
COHORT_RULES = {
    "chest_pain":         [r"\bchest pain\b", r"\bcp\b", r"\bchest discomfort\b",
                           r"\bpressure\b", r"\bpalpitations?\b"],
    "sob":                [r"\bshortness of breath\b", r"\bsob\b", r"\bdyspn(ea|oe)a\b",
                           r"\brespiratory distress\b", r"\bbreath(ing)? difficulty\b"],
    "abdominal_pain":     [r"\babdominal pain\b", r"\babd(ominal)? pain\b", r"\bstomach pain\b",
                           r"\bepigastric\b", r"\bruq\b", r"\brlq\b", r"\blu?q\b", r"\bllq\b",
                           r"\bnausea\b", r"\bvomit(ing)?\b"],
    "head_injury_dizzy":  [r"\bhead( )?injur(y|ies)\b", r"\bfall\b", r"\btrauma\b",
                           r"\bdizziness?\b", r"\bvertigo\b", r"\bsyncope\b",
                           r"\bfaint(ing)?\b", r"\bheadache\b"],
    "flank_pain":         [r"\bflank pain\b", r"\brenal colic\b", r"\bnephrolith(iasis|ic)\b",
                           r"\bkidney stone\b", r"\bhematuria\b"],
}
COMPILED_RULES = {k: [re.compile(p) for p in v] for k, v in COHORT_RULES.items()}

def assign_cohort(cc):
    if not isinstance(cc, str):
        return "other"
    t = re.sub(r"\s+", " ", cc.strip().lower())
    for cohort, patterns in COMPILED_RULES.items():
        for p in patterns:
            if p.search(t):
                return cohort
    return "other"


def vectorized_legacy_label(modality, body_part):
    """Map (modality, body_part) → legacy 7-class. Returns Series."""
    m = modality.astype(str).str.upper()
    b = body_part.astype(str).str.upper()
    ct_head = (m == "CT") & (b == "HEAD")
    ct_abd  = (m == "CT") & (b.isin(["ABDOMEN", "ABD_PELVIS", "PELVIS"]))
    return np.select(
        condlist=[m == "CXR", m == "US", ct_head, ct_abd],
        choicelist=["cxr",   "us",      "ct_head", "ct_abd"],
        default="other",
    )


def agg_set_7(labels):
    s = sorted({x for x in labels if x and x != "none"})
    if not s: return "none"
    if len(s) == 1: return s[0]
    return "multi"


def main():
    T0 = time.time()
    print(f"[ENV={paths.ENV}]")

    # 1. imaging_events_v2
    print(f"\n[1] Loading imaging_events_v2 ({paths.IMAGING_EVENTS_V2_CSV})...", flush=True)
    img = pd.read_csv(paths.IMAGING_EVENTS_V2_CSV)
    img["event_time"] = pd.to_datetime(img["event_time"], errors="coerce")
    img = img.dropna(subset=["event_time", "stay_id"]).copy()
    img["stay_id"] = img["stay_id"].astype(int)
    print(f"    {len(img):,} events, {img['stay_id'].nunique():,} stays")

    # 2. edstays
    print(f"\n[2] Loading edstays...", flush=True)
    ed = pd.read_csv(paths.EDSTAYS_CSV, usecols=["stay_id", "subject_id", "intime", "outtime"])
    ed["intime"]  = pd.to_datetime(ed["intime"], errors="coerce")
    ed["outtime"] = pd.to_datetime(ed["outtime"], errors="coerce")
    ed = ed.dropna(subset=["intime", "outtime"]).copy()
    ed["stay_id"] = ed["stay_id"].astype(int)
    print(f"    {len(ed):,} ED stays")

    # 3. master (mimic_ed_input_hpi) — robust encoding fallback
    print(f"\n[3] Loading master ({paths.MIMIC_ED_INPUT_HPI_CSV})...", flush=True)
    master = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1", "cp949"]:
        try:
            master = pd.read_csv(paths.MIMIC_ED_INPUT_HPI_CSV, encoding=enc, low_memory=False)
            print(f"    [encoding ok: {enc}]")
            break
        except UnicodeDecodeError:
            continue
    if master is None:
        raise UnicodeDecodeError("Could not decode master CSV with common encodings")
    print(f"    {len(master):,} rows; cols(sample)={list(master.columns)[:15]}")
    if "stay_id" not in master.columns:
        raise ValueError(f"master missing stay_id. cols: {list(master.columns)[:30]}")
    cc_col = "chiefcomplaint" if "chiefcomplaint" in master.columns else None
    if cc_col is None:
        # fallback
        for cand in ["chief_complaint", "complaint", "ChiefComplaint"]:
            if cand in master.columns:
                cc_col = cand; break
    if cc_col is None:
        raise ValueError(f"master missing chiefcomplaint column. cols: {list(master.columns)[:30]}")
    master["stay_id"] = pd.to_numeric(master["stay_id"], errors="coerce").astype("Int64")
    master = master.dropna(subset=["stay_id"]).copy()
    master["stay_id"] = master["stay_id"].astype(int)
    print(f"    chiefcomplaint column: {cc_col}")

    # 4. derive label_key + legacy_label (vectorized)
    print(f"\n[4] Deriving per-event labels...", flush=True)
    img["label_key"] = img["modality"].astype(str) + "_" + img["body_part"].astype(str)
    img["legacy_label"] = vectorized_legacy_label(img["modality"], img["body_part"])

    # Attach intime for early-window
    img = img.merge(ed[["stay_id", "intime"]], on="stay_id", how="left", suffixes=("", "_ed"))
    img = img.dropna(subset=["intime"]).copy()
    img["delta_hours"] = (img["event_time"] - img["intime"]).dt.total_seconds() / 3600.0
    print(f"    {len(img):,} events with intime")

    # 5. aggregate per stay
    print(f"\n[5] Aggregating per-stay sets...", flush=True)
    full_sets   = img.groupby("stay_id")["label_key"].apply(lambda s: sorted(set(s))).rename("gold_set_full").reset_index()
    full_count  = img.groupby("stay_id").size().rename("n_events_full").reset_index()
    legacy_full = img.groupby("stay_id")["legacy_label"].apply(lambda s: agg_set_7(list(s))).rename("imaging_choice_v2_full").reset_index()

    img_early = img[(img["delta_hours"] >= 0) & (img["delta_hours"] <= EARLY_HOURS)]
    early_sets   = img_early.groupby("stay_id")["label_key"].apply(lambda s: sorted(set(s))).rename("gold_set_early").reset_index()
    early_count  = img_early.groupby("stay_id").size().rename("n_events_early").reset_index()
    legacy_early = img_early.groupby("stay_id")["legacy_label"].apply(lambda s: agg_set_7(list(s))).rename("imaging_choice_v2_early").reset_index()
    print(f"    full sets for {len(full_sets):,} stays; early sets for {len(early_sets):,} stays")

    # 6. merge into cohort table
    print(f"\n[6] Building cohort table...", flush=True)
    out = master.merge(ed[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
    out["has_edstay"] = out["intime"].notna().astype(int)
    out["cohort"] = out[cc_col].apply(assign_cohort)
    for df_, key in [(full_sets, None), (early_sets, None), (full_count, None), (early_count, None),
                     (legacy_full, None), (legacy_early, None)]:
        out = out.merge(df_, on="stay_id", how="left")

    # fill defaults
    out["gold_set_full"]  = out["gold_set_full"].apply(lambda x: x if isinstance(x, list) else [])
    out["gold_set_early"] = out["gold_set_early"].apply(lambda x: x if isinstance(x, list) else [])
    out["n_events_full"]  = out["n_events_full"].fillna(0).astype(int)
    out["n_events_early"] = out["n_events_early"].fillna(0).astype(int)
    out["imaging_choice_v2_full"]  = out["imaging_choice_v2_full"].fillna("none")
    out["imaging_choice_v2_early"] = out["imaging_choice_v2_early"].fillna("none")

    # no_imaging mutex flags
    out["no_imaging_full"]  = (out["n_events_full"]  == 0).astype(int)
    out["no_imaging_early"] = (out["n_events_early"] == 0).astype(int)

    # cardinality
    out["card_full"]  = out["gold_set_full"].apply(len)
    out["card_early"] = out["gold_set_early"].apply(len)

    # 7. top-N one-hot labels
    print(f"\n[7] Top-{TOP_N_LABELS} labels for one-hot...", flush=True)
    label_counter = Counter()
    for s in out["gold_set_full"]:
        for lab in s:
            label_counter[lab] += 1
    top_labels = [lab for lab, _ in label_counter.most_common(TOP_N_LABELS)]
    print(f"    top labels (count): {[(l, label_counter[l]) for l in top_labels]}")
    for lab in top_labels:
        out[f"y_full__{lab}"]  = out["gold_set_full"].apply(lambda s, L=lab: int(L in s))
        out[f"y_early__{lab}"] = out["gold_set_early"].apply(lambda s, L=lab: int(L in s))

    # 8. serialize set columns + save
    print(f"\n[8] Saving cohort_v2.csv...", flush=True)
    out["gold_set_full"]  = out["gold_set_full"].apply(json.dumps)
    out["gold_set_early"] = out["gold_set_early"].apply(json.dumps)
    out.to_csv(paths.COHORT_V2_CSV, index=False)
    print(f"    Saved: {paths.COHORT_V2_CSV} ({len(out):,} rows, {len(out.columns)} cols)")

    # 9. stats
    print(f"\n=== cohort_v2 stats ===")
    print(f"Total master stays: {len(out):,}")
    print(f"has_edstay: {out['has_edstay'].sum():,} ({100*out['has_edstay'].mean():.1f}%)")
    print(f"\nCohort distribution:")
    print(out["cohort"].value_counts().to_string())
    print(f"\nNo imaging (full window): {out['no_imaging_full'].sum():,} ({100*out['no_imaging_full'].mean():.1f}%)")
    print(f"No imaging (early 2h):    {out['no_imaging_early'].sum():,} ({100*out['no_imaging_early'].mean():.1f}%)")
    print(f"\nGold set cardinality (full window) distribution:")
    print(out["card_full"].value_counts().sort_index().to_string())
    print(f"\nGold set cardinality (early 2h) distribution:")
    print(out["card_early"].value_counts().sort_index().to_string())
    print(f"\nLegacy imaging_choice_v2_full:")
    print(out["imaging_choice_v2_full"].value_counts().to_string())
    print(f"\nLegacy imaging_choice_v2_early:")
    print(out["imaging_choice_v2_early"].value_counts().to_string())

    # Multi-label by cohort (full)
    print(f"\nMulti-label (card_full>=2) by cohort:")
    mlc = out[out["card_full"] >= 2].groupby("cohort").size()
    print(mlc.to_string())

    # Stats JSON
    stats = {
        "total_master_stays": int(len(out)),
        "has_edstay":        int(out["has_edstay"].sum()),
        "cohort_counts":     out["cohort"].value_counts().to_dict(),
        "no_imaging_full":   int(out["no_imaging_full"].sum()),
        "no_imaging_early":  int(out["no_imaging_early"].sum()),
        "card_full_dist":    out["card_full"].value_counts().sort_index().to_dict(),
        "card_early_dist":   out["card_early"].value_counts().sort_index().to_dict(),
        "top_labels":        top_labels,
        "top_label_counts":  {lab: int(label_counter[lab]) for lab in top_labels},
        "all_label_counts":  {lab: int(c) for lab, c in label_counter.most_common()},
        "legacy_full_dist":  out["imaging_choice_v2_full"].value_counts().to_dict(),
        "legacy_early_dist": out["imaging_choice_v2_early"].value_counts().to_dict(),
        "EARLY_HOURS":       EARLY_HOURS,
    }
    with open(paths.COHORT_V2_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {paths.COHORT_V2_STATS_JSON}")

    print(f"\nTotal elapsed: {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
