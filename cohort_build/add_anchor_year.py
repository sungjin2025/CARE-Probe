#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""add_anchor_year.py — build a stay_id -> anchor_year_group sidecar for E-shift.

MIMIC-IV shifts each patient's dates into a random year, so `intime` is USELESS
for temporal split (cross-patient order is random). The only REAL temporal signal
is `anchor_year_group` (true 3-year bins: 2008-2010 ... 2017-2019) in the hosp
`patients` table. cohort_v2 has only stay_id, so we 2-hop join:
    cohort_v2.stay_id --(MIMIC-IV-ED edstays)--> subject_id --(hosp patients)--> anchor_year_group

Output sidecar CSV: stay_id, subject_id, anchor_year_group, anchor_year
→ used by e0_crc.py --split temporal --shift_csv <out>.

Usage (server; fill in your MIMIC paths):
  python add_anchor_year.py \
    --edstays_csv  /path/to/mimic-iv-ed/ed/edstays.csv.gz \
    --patients_csv /path/to/mimic-iv/hosp/patients.csv.gz \
    --out_csv      stay_anchor_year.csv
"""
import argparse
import pandas as pd

try:
    import paths
    DEFAULT_COHORT = paths.COHORT_V2_CSV
    DEFAULT_OUT = str(__import__("pathlib").Path(paths.WORK_DIR) / "stay_anchor_year.csv")
except Exception:
    DEFAULT_COHORT, DEFAULT_OUT = "cohort_v2.csv", "stay_anchor_year.csv"


def _read(path, usecols):
    # handles .csv and .csv.gz; only the needed columns
    return pd.read_csv(path, usecols=lambda c: c in usecols)


def main():
    ap = argparse.ArgumentParser(description="cohort_v2 stay_id -> anchor_year_group sidecar")
    ap.add_argument("--cohort_csv", default=DEFAULT_COHORT)
    ap.add_argument("--edstays_csv", required=True, help="MIMIC-IV-ED edstays.csv[.gz]")
    ap.add_argument("--patients_csv", required=True, help="MIMIC-IV hosp patients.csv[.gz]")
    ap.add_argument("--out_csv", default=DEFAULT_OUT)
    args = ap.parse_args()

    coh = _read(args.cohort_csv, {"stay_id"}).drop_duplicates("stay_id")
    ed = _read(args.edstays_csv, {"stay_id", "subject_id"}).drop_duplicates("stay_id")
    pa = _read(args.patients_csv, {"subject_id", "anchor_year_group", "anchor_year"}).drop_duplicates("subject_id")
    print(f"  cohort stays={len(coh):,}  edstays rows={len(ed):,}  patients rows={len(pa):,}")

    m = coh.merge(ed, on="stay_id", how="left").merge(pa, on="subject_id", how="left")
    n_sid = int(m["subject_id"].notna().sum())
    n_ayg = int(m["anchor_year_group"].notna().sum())
    print(f"  matched subject_id: {n_sid}/{len(m)} ({100*n_sid/len(m):.1f}%)")
    print(f"  matched anchor_year_group: {n_ayg}/{len(m)} ({100*n_ayg/len(m):.1f}%)")
    print("  anchor_year_group distribution:")
    print(m["anchor_year_group"].value_counts(dropna=False).sort_index().to_string())

    out = m[["stay_id", "subject_id", "anchor_year_group", "anchor_year"]]
    out.to_csv(args.out_csv, index=False)
    print(f"  -> saved {args.out_csv}")


if __name__ == "__main__":
    main()
