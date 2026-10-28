# -*- coding: utf-8 -*-
"""work_b_signal_crc.py  (SERVER drop-in, CPU only) -- signal-utility protocol.

input-only / combined / probe / random  through the FROZEN CRC machinery,
all 8 backbones, for the signal-utility table and figure:
"does the internal-state probe certify more auto-commits than a supervised
classifier over the clinical input, under the same joint-risk budget?"

Reuses existing machinery with ZERO new logic:
  * careprobe_incremental_value.oof_scores  -> s_probe / s_input / s_combined
      probe    = logistic on last-token hidden state (deployed idx3, depth 0.875)
      input    = logistic on structured triage + HPI TF-IDF  (presentation-time)
      combined = logistic on [PCA(H,50) ++ PCA(TFIDF,100) ++ scaled struct]
  * e0_crc.select_lambda / risk_coverage / aurc_random  (same guarantee as paper)

PROTOCOL (env SUBS):
  SUBS="42"        -> single-subsample head-to-head (default); anchor-comparable to
                      the P(True) 8-backbone asset.  -> work_b_signal_crc_results.csv
  SUBS="42,43,44"  -> MAIN protocol (seed-averaged, 3 sub x 10 split = 30 runs);
                      matches main Table III.         -> ..._results_MAIN.csv
alpha=delta=0.10, split seeds 42-51.

=====================================================================
HOW TO RUN (server): place THIS FILE inside cohort_build_v2/ then:
  validation / P(True)-comparable:   python work_b_signal_crc.py
  paper numbers (match Table III):   SUBS=42,43,44 python work_b_signal_crc.py
CPU only -- no GPU, no LLM inference (reads stored probe_*.npz). Send back the CSV(s).
LOCAL run from a separate folder: set env CB_DIR to the cohort_build_v2 path
  (original folder stays read-only; outputs land next to this script).
De-leak variant: PROBE_TAG=-deleak (after deleak_probe_runner.py regenerates .npz).
=====================================================================
"""
import os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
# CB_DIR = where cohort_build_v2 lives (data + machinery). Default = HERE
# (server: place this file inside cohort_build_v2). For a LOCAL run from a separate
# folder, set env CB_DIR to the cohort_build_v2 path; outputs land in HERE.
CB_DIR = os.environ.get("CB_DIR", HERE)
sys.path.insert(0, CB_DIR)

from e0_crc import select_lambda, risk_coverage, aurc_random
from careprobe_incremental_value import oof_scores

if os.environ.get("CB_DIR"):
    COHORT = os.path.join(CB_DIR, "cohort_v2.csv")
else:
    try:
        import paths
        COHORT = paths.COHORT_V2_CSV
    except Exception:
        COHORT = os.path.join(CB_DIR, "cohort_v2.csv")

if not hasattr(np, "trapz"):           # numpy>=2 compat (matches _ptrue_crc_allbb)
    np.trapz = np.trapezoid

RUNS = os.path.join(CB_DIR, "runs_main")
ASSET = os.path.join(CB_DIR, "_ptrue_crc_allbb_results.csv")   # optional anchor
PROBE_TAG = os.environ.get("PROBE_TAG", "")

DEPLOY_IDX = 3                          # relative depth 0.875 (deployed protocol)
SUB_SEEDS = [int(x) for x in os.environ.get("SUBS", "42").split(",") if x.strip()]
SPLIT_SEEDS = list(range(42, 52))       # 42..51
ALPHA = DELTA = 0.10
BACKBONES = ["qwen14b", "llama31-8b", "gemma2-9b", "mistral7b-v03", "phi3mini-128k",
             "med42-8b", "openbio-8b", "mediphi"]
CT_COLS = ["CT_ABD_PELVIS", "CT_HEAD", "CT_CSPINE", "CT_CHEST", "CT_EXTREMITY", "CT_PELVIS"]
STRUCT_NUM = ["pain", "age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
STRUCT_CAT = ["gender", "race", "arrival_transport", "triage"]
SIGS = ["probe", "input-only", "combined", "random"]


def crc_runs(s_all, y, cardg):
    """One subsample: AURC gap-closed + per-split (jr, cm, cond, ok)."""
    _, _, aurc_sig = risk_coverage(s_all, y.astype(float))
    _, _, aurc_or = risk_coverage(cardg, y.astype(float))
    aurc_rd = aurc_random(y.astype(float), 42)
    span = aurc_rd - aurc_or
    gap = (aurc_rd - aurc_sig) / span if abs(span) > 1e-9 else float("nan")
    jr_l, cm_l, cond_l, ok_l = [], [], [], []
    for sp in SPLIT_SEEDS:
        rng = np.random.default_rng(sp)
        idx = rng.permutation(len(y))
        ntr, nca = int(0.4 * len(y)), int(0.3 * len(y))
        ca, te = idx[ntr:ntr + nca], idx[ntr + nca:]
        lam, _ = select_lambda(s_all[ca], y[ca].astype(float), ALPHA, DELTA)
        if lam is None:
            jr = cm = 0.0
        else:
            c = s_all[te] <= lam
            jr = float(np.mean(y[te] * c)); cm = float(c.mean())
        jr_l.append(jr); cm_l.append(cm)
        cond_l.append(jr / cm if cm > 1e-9 else 0.0)
        ok_l.append(int(jr <= ALPHA))
    return float(roc_auc_score(y, s_all)), gap, jr_l, cm_l, cond_l, ok_l


def load_bb(bb, sub, struct_all, hpi_all, coh, has_card):
    fp = os.path.join(RUNS, f"probe_{bb}{PROBE_TAG}_n2000_seed{sub}.npz")
    if not os.path.exists(fp):
        return None
    d = np.load(fp, allow_pickle=True)
    sid = d["stay_ids"]; H = d["last"][:, DEPLOY_IDX, :].astype(np.float32)
    pred = d["pred_bits"]; labels = list(d["labels"]); valid = d["valid"].astype(bool)
    gold = coh.reindex(sid)
    ct_idx = [labels.index(c) for c in CT_COLS if c in labels]
    gold_ct = np.stack([gold[f"y_full__{labels[i]}"].fillna(0).values for i in ct_idx], axis=1)
    ct_miss = ((gold_ct == 1) & (pred[:, ct_idx] == 0)).any(axis=1).astype(int)
    keep = valid & np.isin(sid, struct_all.index.values)
    sid, H, y = sid[keep], H[keep], ct_miss[keep]
    Xs = struct_all.reindex(sid).astype(float).fillna(struct_all.median(numeric_only=True)).values
    hp = hpi_all.reindex(sid).fillna("").values
    if has_card:
        cardg = pd.to_numeric(coh["card_full"].reindex(sid), errors="coerce").fillna(0).values.astype(float)
    else:
        cardg = np.zeros(len(y))
    return sid, H, y, Xs, hp, cardg


def load_b4_signals(bb, sub, sid):
    """Optional P(True)/semantic-entropy signals from the b4 CSV (same PROBE_TAG),
    stay_id-aligned to the kept stays. Higher = more uncertain = defer (CRC-compatible).
    Returns {} if the b4 CSV is absent (e.g. WITH-icd10 run with no local b4)."""
    f = os.path.join(CB_DIR, "runs_e0", f"b4_semantic_ptrue_{bb}{PROBE_TAG}_n2000_seed{sub}.csv")
    if not os.path.exists(f):
        return {}
    b = pd.read_csv(f)
    out = {}
    for nm, col in [("ptrue", "u_ptrue"), ("sement", "semantic_entropy")]:
        if col not in b.columns:
            continue
        m = b.set_index("stay_id")[col].to_dict()
        v = np.array([m.get(int(s), np.nan) for s in sid], float)
        if np.isnan(v).all():
            continue
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        out[nm] = v
    return out


def main():
    coh = pd.read_csv(COHORT).drop_duplicates("stay_id").set_index("stay_id")
    for c in STRUCT_NUM:
        coh[c] = pd.to_numeric(coh[c], errors="coerce")
    cat = pd.get_dummies(coh[STRUCT_CAT].astype(str), dummy_na=True)
    struct_all = pd.concat([coh[STRUCT_NUM], cat], axis=1)
    hpi_all = coh["hpi_clean"].fillna("").astype(str)
    has_card = "card_full" in coh.columns
    asset = pd.read_csv(ASSET) if os.path.exists(ASSET) else None
    anchor_on = (SUB_SEEDS == [42]) and (asset is not None)
    print(f"[config] SUBS={SUB_SEEDS} SPLITS={SPLIT_SEEDS[0]}-{SPLIT_SEEDS[-1]} "
          f"alpha={ALPHA} PROBE_TAG='{PROBE_TAG}' anchor={'on' if anchor_on else 'off'}", flush=True)

    rows = []
    persplit = []  # pass-1b 2026-10-25: per-split rows -> paired Delta(combined-input) CIs for main Table III
    for bb in BACKBONES:
        acc = {}
        nseen = 0
        for sub in SUB_SEEDS:
            loaded = load_bb(bb, sub, struct_all, hpi_all, coh, has_card)
            if loaded is None:
                print(f"[skip] {bb} sub{sub}: npz missing"); continue
            sid, H, y, Xs, hp, cardg = loaded
            nseen += 1
            s_pr, s_in, s_cb = oof_scores(H, Xs, hp, y, sub)
            s_rd = np.random.default_rng(42).random(len(y))
            siglist = [("probe", s_pr), ("input-only", s_in), ("combined", s_cb), ("random", s_rd)]
            siglist += list(load_b4_signals(bb, sub, sid).items())   # P(True)/SemEnt if present
            for nm, s in siglist:
                auc, gap, jr_l, cm_l, cond_l, ok_l = crc_runs(s, y, cardg)
                a = acc.setdefault(nm, dict(auc=[], gap=[], jr=[], cm=[], cond=[], ok=[]))
                a["auc"].append(auc); a["gap"].append(gap)
                a["jr"] += jr_l; a["cm"] += cm_l
                a["cond"] += cond_l; a["ok"] += ok_l
                for k, sp in enumerate(SPLIT_SEEDS):  # pass-1b per-split dump
                    persplit.append(dict(backbone=bb, sub=sub, signal=nm, split=sp,
                                         joint_risk=round(float(jr_l[k]), 4),
                                         commit=round(float(cm_l[k]), 4),
                                         cond_risk=round(float(cond_l[k]), 4),
                                         ok=int(ok_l[k])))
            print(f"[build] {bb} sub{sub}: n={len(y)} ct_miss_prev={y.mean():.3f} "
                  f"signals={len(siglist)}", flush=True)
        if nseen == 0:
            continue
        for nm in acc:
            a = acc[nm]
            rows.append(dict(backbone=bb, signal=nm, n_sub=nseen,
                             auc=round(float(np.mean(a["auc"])), 3),
                             gap_closed=round(float(np.mean(a["gap"])), 3),
                             joint_risk=round(float(np.mean(a["jr"])), 4),
                             commit=round(float(np.mean(a["cm"])), 3),
                             cond_risk=round(float(np.mean(a["cond"])), 3),
                             ok_rate=round(float(np.mean(a["ok"])), 2)))
        if anchor_on:
            aa = asset[(asset.backbone == bb) & (asset.signal == "CARE-Probe")]
            mine = [r for r in rows if r["backbone"] == bb and r["signal"] == "probe"][0]
            if len(aa):
                ar = aa.iloc[0]
                flag = "OK" if abs(mine["commit"] - float(ar["commit"])) < 0.02 else "CHECK"
                print(f"[anchor:{flag}] {bb}: probe commit {mine['commit']:.3f} vs asset "
                      f"{float(ar['commit']):.3f} | auc {mine['auc']:.3f} vs {float(ar['auc']):.3f}",
                      flush=True)

    if not rows:
        print("[error] no backbones processed -- check runs_main/probe_*.npz paths"); return
    proto = "" if SUB_SEEDS == [42] else "_MAIN"
    out = os.path.join(HERE, f"work_b_signal_crc_results{PROBE_TAG}{proto}.csv")
    df = pd.DataFrame(rows)[["backbone", "signal", "n_sub", "auc", "gap_closed",
                             "joint_risk", "commit", "cond_risk", "ok_rate"]]
    df.to_csv(out, index=False)
    print("\n[saved]", out)
    if persplit:  # pass-1b: enables paired per-split Delta(combined-input) bootstrap CIs
        outp = os.path.join(HERE, f"work_b_signal_crc_persplit{PROBE_TAG}{proto}.csv")
        pd.DataFrame(persplit).to_csv(outp, index=False)
        print("[saved]", outp)
    print(df.to_string(index=False))

    print("\n=== commit @ alpha=0.10: input-only vs probe vs combined ===")
    nb = comb_gt_in = probe_gt_in = 0
    for bb in BACKBONES:
        dd = df[df.backbone == bb].set_index("signal")
        if not {"input-only", "combined", "probe"}.issubset(dd.index):
            continue
        nb += 1
        ci, cc, cp = dd.loc["input-only", "commit"], dd.loc["combined", "commit"], dd.loc["probe", "commit"]
        comb_gt_in += int(cc > ci); probe_gt_in += int(cp > ci)
        print(f"  {bb:16s} input {ci:.3f} | probe {cp:.3f} | combined {cc:.3f}  "
              f"{'comb>input' if cc > ci else 'comb<=input'}")
    if nb:
        print(f"\n  combined > input-only on {comb_gt_in}/{nb};  probe > input-only on {probe_gt_in}/{nb}")
        print("  (feeds main Table IV + Fig 2; H2 = does internal info raise certified"
              " coverage beyond clinical input?)")


if __name__ == "__main__":
    main()
