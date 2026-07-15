# -*- coding: utf-8 -*-
"""M1.2c: build imaging_events_v2.csv from raw MIMIC sources + our deterministic mapping."""
import gzip, io, csv, time
import pandas as pd
import paths

MAX_PER_STAY = 2000  # safety cap


def load_detail_primary():
    """Stream radiology_detail gzip → DataFrame (note_id, exam_name, exam_code) for ordinal=1 only."""
    print(f"[1] [ENV={paths.ENV}] Stream-loading radiology_detail primary entries...", flush=True)
    t0 = time.time()
    name, code = {}, {}
    n = 0
    with paths.open_radiology_detail_gz() as f:
        with gzip.GzipFile(fileobj=f) as gz:
            rdr = csv.DictReader(io.TextIOWrapper(gz, encoding="utf-8"))
            for row in rdr:
                n += 1
                if row.get("field_ordinal") != "1":
                    continue
                nid = row.get("note_id", "")
                fn = row.get("field_name", "")
                fv = row.get("field_value", "")
                if fn == "exam_name":
                    name[nid] = fv
                elif fn == "exam_code":
                    code[nid] = fv
                if n % 1000000 == 0:
                    print(f"    ... {n:,} rows ({time.time()-t0:.1f}s)", flush=True)
    df = pd.DataFrame({"note_id": list(name.keys()), "exam_name": list(name.values())})
    df["exam_code"] = df["note_id"].map(code).fillna("")
    print(f"    → {len(df):,} primary exam_name rows ({time.time()-t0:.1f}s)", flush=True)
    return df


def main():
    T0 = time.time()
    detail = load_detail_primary()

    print(f"\n[2] Loading mapping {paths.EXAM_NAME_MAPPING_CSV}...", flush=True)
    mapping = pd.read_csv(paths.EXAM_NAME_MAPPING_CSV)
    print(f"    {len(mapping):,} rules", flush=True)

    print(f"\n[3] Applying mapping to detail...", flush=True)
    detail = detail.merge(
        mapping[["exam_name", "modality", "body_part"]], on="exam_name", how="left"
    )
    nun = detail["modality"].isna().sum()
    if nun > 0:
        print(f"    WARN: {nun:,} unmapped → OTHER/OTHER", flush=True)
        detail["modality"] = detail["modality"].fillna("OTHER")
        detail["body_part"] = detail["body_part"].fillna("OTHER")
    print(f"    modality dist (pre-MAMMO filter):")
    print(detail["modality"].value_counts().to_string())

    n0 = len(detail)
    detail = detail[detail["modality"] != "MAMMO"].copy()
    print(f"\n[4] Filtered MAMMO: {n0:,} → {len(detail):,}", flush=True)

    print(f"\n[5] Loading radiology metadata (no text) from {paths.RADIOLOGY_CSV}...", flush=True)
    t1 = time.time()
    rad = pd.read_csv(
        paths.RADIOLOGY_CSV,
        usecols=["note_id", "subject_id", "hadm_id", "charttime"],
        low_memory=False,
    )
    rad["charttime"] = pd.to_datetime(rad["charttime"], errors="coerce")
    rad = rad.dropna(subset=["charttime", "subject_id"]).copy()
    rad["subject_id"] = rad["subject_id"].astype(int)
    print(f"    {len(rad):,} rows ({time.time()-t1:.1f}s)", flush=True)

    print(f"\n[6] Joining detail × radiology by note_id...", flush=True)
    events = detail.merge(rad, on="note_id", how="inner")
    print(f"    {len(events):,} events with subject_id+charttime", flush=True)

    print(f"\n[7] Loading edstays from {paths.EDSTAYS_CSV}...", flush=True)
    ed = pd.read_csv(paths.EDSTAYS_CSV, usecols=["stay_id", "subject_id", "intime", "outtime"])
    ed["intime"] = pd.to_datetime(ed["intime"], errors="coerce")
    ed["outtime"] = pd.to_datetime(ed["outtime"], errors="coerce")
    ed = ed.dropna(subset=["intime", "outtime", "subject_id"]).copy()
    ed["stay_id"] = ed["stay_id"].astype(int)
    ed["subject_id"] = ed["subject_id"].astype(int)
    print(f"    {len(ed):,} ED stays", flush=True)

    print(f"\n[8] Time-window join (subject_id + charttime ∈ [intime,outtime])...", flush=True)
    t2 = time.time()
    merged = events.merge(ed, on="subject_id", how="inner")
    print(f"    pre-filter merged rows: {len(merged):,} ({time.time()-t2:.1f}s)", flush=True)
    mask = (merged["charttime"] >= merged["intime"]) & (merged["charttime"] <= merged["outtime"])
    final = merged.loc[mask].copy()
    print(f"    in-window events: {len(final):,}", flush=True)

    if MAX_PER_STAY and MAX_PER_STAY > 0:
        final = final.sort_values(["stay_id", "charttime"]).groupby("stay_id").head(MAX_PER_STAY).copy()
        print(f"    after max_per_stay={MAX_PER_STAY} cap: {len(final):,}", flush=True)

    print(f"\n[9] Saving output to {paths.IMAGING_EVENTS_V2_CSV}...", flush=True)
    out = pd.DataFrame({
        "stay_id":    final["stay_id"].astype(int),
        "event_time": final["charttime"],
        "modality":   final["modality"],
        "body_part":  final["body_part"],
        "exam_name":  final["exam_name"],
        "exam_code":  final["exam_code"],
        "note_id":    final["note_id"],
        "subject_id": final["subject_id"].astype(int),
    }).sort_values(["stay_id", "event_time"]).reset_index(drop=True)
    out.to_csv(paths.IMAGING_EVENTS_V2_CSV, index=False)
    print(f"    Saved ({len(out):,} rows)", flush=True)

    print(f"\n=== imaging_events_v2 QC ===")
    print(f"Total events: {len(out):,}")
    print(f"Unique stays: {out['stay_id'].nunique():,}")
    print(f"Avg events/stay: {len(out)/max(1,out['stay_id'].nunique()):.2f}")
    print(f"\nmodality:")
    print(out["modality"].value_counts().to_string())
    print(f"\nbody_part:")
    print(out["body_part"].value_counts().to_string())
    print(f"\n(modality, body_part) top 25 combos:")
    print(out.groupby(["modality","body_part"]).size().sort_values(ascending=False).head(25).to_string())
    epps = out.groupby("stay_id").size()
    print(f"\nEvents per stay (quantiles):")
    print(epps.describe().to_string())
    print(f"Stays with >=2 events: {(epps>=2).sum():,} ({100*(epps>=2).sum()/len(epps):.1f}%)")
    multi_mod = out.groupby("stay_id")["modality"].nunique()
    print(f"Stays with multiple distinct modalities: {(multi_mod>=2).sum():,} ({100*(multi_mod>=2).sum()/len(multi_mod):.1f}%)")

    print(f"\nTotal elapsed: {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
