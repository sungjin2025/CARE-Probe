#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""extract_arms_REAL.py — CARE-Probe chart-review arm extraction (real data, JMS-aligned).

Reproduces e0_crc.run_real's keep-filter / 40-30-30 split / probe(layer_idx3,
target=ct_miss) / select_lambda **exactly** to build the deployment operating point,
then draws the 3 arms (A commit&ct_miss / B defer / C commit&correct) from the TEST
split to populate the adjudication sheet.

Reproducibility: fixed seed. Defaults to the full Qwen cohort (runs_b3).
Output: adjudication sheet + case_id mapping CSVs written to OUT_DIR.
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
import imaging_runner_v2 as ir
from e0_crc import _fit_probe_scores, select_lambda   # reuse the canonical implementation as-is

LABELS = list(ir.LABEL_SPACE_V2)
CT = [L for L in LABELS if L.startswith("CT_")]
OUT_DIR = os.environ.get("OUT_DIR", "/path/to/output")

# De-identified clinical text fields (excludes outcome/bias-inducing columns such as diagnosis/race)
TEXT_COLS = [("age", "Age"), ("gender", "Sex"), ("arrival_transport", "Arrival mode"),
             ("triage", "ESI"), ("chiefcomplaint", "Chief complaint"), ("pain", "Pain"),
             ("temperature", "Temp"), ("heartrate", "HR"), ("resprate", "RR"),
             ("o2sat", "SpO2"), ("sbp", "SBP"), ("dbp", "DBP"), ("hpi_clean", "HPI")]
FIELDS = ["case_id", "case_text", "exam_under_review", "q1_indication",
          "q2_harm_if_omitted", "q3_missed_by_practice", "q3_practice_harm", "note", "time_min"]


def build_text(row):
    parts = []
    for col, ko in TEXT_COLS:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() not in ("", "nan"):
            parts.append(f"{ko}={row[col]}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(HERE, "runs_b3", "probe_qwen14b_full_seed42.npz"))
    ap.add_argument("--tag", default="qwen_full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer_idx", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--nA", type=int, default=150)
    ap.add_argument("--nB", type=int, default=50)
    ap.add_argument("--nC", type=int, default=50)
    ap.add_argument("--n_pca", type=int, default=50)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    sids, valid, pred_bits = d["stay_ids"], d["valid"], d["pred_bits"]
    last, layer_idx = d["last"], list(d["layer_idx"])

    coh = pd.read_csv(paths.COHORT_V2_CSV)
    gcols = ["stay_id", "triage"] + [f"y_full__{L}" for L in LABELS] + [c for c, _ in TEXT_COLS]
    coh = coh[[c for c in dict.fromkeys(gcols) if c in coh.columns]].copy()
    gmap = {int(r["stay_id"]): r for _, r in coh.iterrows()}

    # ---- reproduce e0_crc.run_real: keep + labels ----
    keep, y_err, l_ct = [], [], []
    for i, sid in enumerate(sids):
        if valid[i] != 1:
            continue
        g = gmap.get(int(sid))
        if g is None:
            continue
        gold = {L for L in LABELS if int(g.get(f"y_full__{L}", 0) or 0) == 1}
        pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[i][j]) == 1}
        keep.append(i)
        y_err.append(int(pred != gold))
        ctg = {L for L in CT if L in gold}
        l_ct.append(int(len(ctg - pred) > 0) if ctg else 0)
    keep = np.array(keep); l_ct = np.array(l_ct, float)
    y_probe = l_ct.astype(int)                                  # probe target = loss(ct_miss)
    li = args.layer_idx if 0 <= args.layer_idx < len(layer_idx) else len(layer_idx) - 1
    X = last[keep][:, li, :].astype(np.float32)

    # ---- 40/30/30 split (same rng) ----
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(keep))
    n = len(idx); n_tr, n_ca = int(0.4 * n), int(0.3 * n)
    tr, ca, te = idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]
    s_ca = _fit_probe_scores(X[tr], y_probe[tr], X[ca], args.n_pca, args.seed)
    s_te = _fit_probe_scores(X[tr], y_probe[tr], X[te], args.n_pca, args.seed)
    lam, cal_risk = select_lambda(s_ca, l_ct[ca], args.alpha, args.delta)
    commit = s_te <= lam
    joint = float(np.mean(l_ct[te] * commit)); cond = float(np.mean(l_ct[te][commit]))
    print(f"[repro] N_usable={n} test={len(te)} lambda={lam:.4f} commit={commit.mean():.3f} "
          f"joint={joint:.4f}(<= {args.alpha}) cond={cond:.4f} ct_base={l_ct.mean():.3f}")

    # ---- arm classification (TEST) ----
    te_ct = l_ct[te].astype(int)
    armA = [te[k] for k in range(len(te)) if commit[k] and te_ct[k] == 1]
    armB = [te[k] for k in range(len(te)) if not commit[k]]
    armC = [te[k] for k in range(len(te)) if commit[k] and te_ct[k] == 0]
    print(f"[arms available] A={len(armA)} B={len(armB)} C={len(armC)}")
    rng2 = np.random.default_rng(args.seed)
    def pick(lst, k):
        lst = list(lst); rng2.shuffle(lst); return lst[:min(k, len(lst))]
    chosen = [(int(keep[j]), "A") for j in pick(armA, args.nA)] \
           + [(int(keep[j]), "B") for j in pick(armB, args.nB)] \
           + [(int(keep[j]), "C") for j in pick(armC, args.nC)]
    rng2.shuffle(chosen)                                        # arm blinding

    rows, mapping = [], []
    for n_i, (row_i, arm) in enumerate(chosen, 1):
        sid = int(sids[row_i]); cid = f"CR{n_i:04d}"
        g = gmap[sid]
        gold = [L for L in LABELS if int(g.get(f"y_full__{L}", 0) or 0) == 1]
        pred = {LABELS[j] for j in range(len(LABELS)) if int(pred_bits[row_i][j]) == 1}
        missed_ct = sorted({L for L in CT if L in set(gold)} - pred)
        mapping.append({"case_id": cid, "stay_id": sid, "arm": arm,
                        "missed_ct": ";".join(missed_ct) or "none",
                        "model_pred": ";".join(sorted(pred)) or "NONE"})
        text = build_text(g)
        exams = gold if gold else ["NONE"]
        for exam in exams:
            rows.append({"case_id": cid, "case_text": text, "exam_under_review": exam,
                         "q1_indication": "", "q2_harm_if_omitted": "",
                         "q3_missed_by_practice": "", "q3_practice_harm": "", "note": "", "time_min": ""})

    os.makedirs(OUT_DIR, exist_ok=True)
    sheet = os.path.join(OUT_DIR, f"adjudication_sheet_{args.tag}.csv")
    mapf = os.path.join(OUT_DIR, f"case_id_mapping_{args.tag}.csv")
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    with open(mapf, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "stay_id", "arm", "missed_ct", "model_pred"])
        w.writeheader(); w.writerows(mapping)
    print(f"[OK] cases={len(chosen)} rows={len(rows)} -> {sheet}")
    print(f"[OK] map -> {mapf}  (keep private within the research team)")


if __name__ == "__main__":
    main()
