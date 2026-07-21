#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0x_deployed_surface_runner.py — E0X: deployed-protocol multi-backbone
surface-signal screening on the MAIN n=2,000 subsample.

Motivation (design: proposals/E0X_deployed_surface_design.md):
the manuscript's negative result (four surface signals at chance) rests on the
E0 screen — single backbone (Qwen), n=500, per-source majority-vote protocol.
E0X re-measures every deployable surface signal
    U_lp   : 1 - exp(mean token logprob)  [deployed greedy generation]
    U_sc / U_lab : self-consistency instability [N samples, deployed prompt]
    S_max / S_mean : per-source retrieval sufficiency (cosine)
    D_jac / D_lab / D_dec / D_high : cross-source conflict (E0 definition;
        per-source generation — a per-source signal by definition)
    pred_card : deployable cardinality |ŷ| of the deployed greedy prediction
on the SAME rows as the MAIN run (df.sample(n=2000, random_state=42) on
cohort_v2.csv — identical convention to all e0_* runners), with error labels
(ct_miss / generic / setloss) computed against gold exactly as e0_crc.py does,
ON THE DEPLOYED GREEDY PREDICTION (not the E0 majority vote).

Evidence modes (--evidence_mode):
    concat : per-source "### Source P/W/G" blocks, top-3/source
             (= e0_probe_runner protocol that produced the MAIN probe npz)
    fused  : min-max + k-means upper-cluster global top-4
             (= imaging_runner_v2 msrc fusion, manuscript §4.1 description)
Pass --main_npz to verify subsample alignment (stay_id sets) and report
greedy-prediction agreement with the MAIN pred_bits — this settles which
evidence mode reproduces the MAIN generation (resolves open item #1, §4.2).

Generation calls per stay: greedy 1 + self-consistency 1 batched (N seqs)
+ per-source 3  →  9 sequences / stay; n=2000 → 18,000 sequences per backbone.

Usage (server):
  CRMSR_ENV=server MILVUS_TOKEN=... python e0x_deployed_surface_runner.py \
    --out_csv runs_e0x/qwen14b_deployed_surface.csv \
    --llm_model_path /path/to/models/Qwen/Qwen2.5-14B-Instruct \
    --main_npz runs_main/probe_qwen14b_n2000_seed42.npz \
    --gpu_id 1 --n 2000 --seed 42 --resume
"""
from __future__ import annotations
import os, sys, time, argparse, itertools, random
from pathlib import Path

# ── Pin CUDA before any torch import. Default GPU 1 (user directive 2026-07-14).
#    GPU 6 is reserved for CR-MSR M1.5 runs — do NOT use it for E0X. ──
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _gpu = "1"
    if "--gpu_id" in sys.argv:
        try: _gpu = sys.argv[sys.argv.index("--gpu_id") + 1]
        except (IndexError, ValueError): pass
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import numpy as np
import pandas as pd

import paths
import imaging_runner_v2 as ir
import e0_conflict_runner as e0r      # LLM loader, cosine search, parsing, label space
import e0_uncertainty_runner as e0u   # sample_llm, text_to_set, uncertainty_from_samples

LABELS = e0r.LABELS
NO_IMG = e0r.NO_IMG
ALL_LABELS = e0r.ALL_LABELS
SRC_COLL = e0r.SRC_COLL               # {"P": PubMed, "W": MedWiki, "G": Guideline}
SRC_ORDER = e0r.SRC_ORDER
CT_LABELS = [L for L in LABELS if L.startswith("CT_")]
# search_cosine key -> imaging_runner_v2 msrc source key (for SOURCE_WEIGHTS)
SRC_TO_MSRC = {"P": "PubMed", "W": "MedWiki", "G": "Guideline"}
TOP_K_CONCAT = ir.TOP_K_SINGLE        # 3 — E0 concat / conflict / sufficiency arm
TOP_K_FUSE = ir.TOP_K_PER_SOURCE      # 5 — msrc fusion arm


def parse_bits(raw: str):
    """One generation → (bits over ALL_LABELS, flag). Mirrors e0r.predict_pregate parse."""
    bits = {L: 0 for L in ALL_LABELS}
    if ir.is_refusal_text(raw):
        return bits, "refusal"
    obj = ir.extract_first_json_object(raw)
    if obj is None:
        return bits, "parse_fail"
    pb, meta = ir.normalize_imaging_json(obj, mode="multi_label")
    if meta.get("is_abstain_json"):
        return bits, "abstain_json"
    fixed, _ = ir.enforce_no_imaging_mutex(pb, label_space=LABELS)
    for L in ALL_LABELS:
        bits[L] = int(fixed.get(L, 0))
    return bits, "ok"


def concat_evidence(hits_by_src) -> str:
    """e0_probe_runner protocol: '### Source P/W/G' blocks, top-3 each."""
    parts = []
    for sk in SRC_ORDER:
        txt = ir._build_evidence_text(hits_by_src[sk][:TOP_K_CONCAT])
        if txt.strip():
            parts.append(f"### Source {sk}\n{txt}")
    return "\n\n".join(parts)


def fused_evidence(hits_by_src) -> str:
    """imaging_runner_v2 msrc fusion (manuscript §4.1): min-max per source,
    k-means K=2 upper cluster, weighted, global top-4. Cosine rescoring is a
    per-collection constant scale → monotone in raw IP, so min-max is unaffected."""
    per_source = {SRC_TO_MSRC[sk]: hits_by_src[sk] for sk in SRC_ORDER}
    return ir._build_msrc_evidence(per_source)


# ---------- conflict metrics — EXACT mirrors of e0_analyze.py ----------
def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0

def d_jac(sets3):
    return float(np.mean([1.0 - _jaccard(a, b)
                          for a, b in itertools.combinations(sets3, 2)]))

def d_lab(bits_by_src):
    dis = 0
    for L in LABELS:
        votes = {bits_by_src[sk][L] for sk in SRC_ORDER}
        if len(votes) > 1: dis += 1
    return dis / len(LABELS)

def d_dec(sets_by_src, noimg_by_src):
    decs = set()
    for sk in SRC_ORDER:
        decs.add("img" if len(sets_by_src[sk]) > 0
                 else ("no" if noimg_by_src[sk] == 1 else "empty"))
    return 1.0 if len(decs) > 1 else 0.0

def d_high(sets_by_src):
    has = {any(L in sets_by_src[sk] for L in CT_LABELS) for sk in SRC_ORDER}
    return 1.0 if len(has) > 1 else 0.0


def error_labels(row: dict, pred_set: set) -> dict:
    """Gold comparison exactly as e0_crc.run_real (window=full)."""
    gold = {L for L in LABELS if int(row.get(f"y_full__{L}", 0) or 0) == 1}
    ctg = {L for L in CT_LABELS if L in gold}
    n_miss, n_over = len(gold - pred_set), len(pred_set - gold)
    denom = 2 * len(gold) + len(pred_set)
    return {
        "y_err": int(pred_set != gold),
        "ct_miss": int(len(ctg - pred_set) > 0) if ctg else 0,
        "setloss": min(1.0, (2 * n_miss + n_over) / denom) if denom else 0.0,
        "gold_card": len(gold),
        "card_full": float(row.get("card_full", 0) or 0),
        "triage": int(row.get("triage", 0) or 0),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="E0X deployed-protocol surface signals")
    ap.add_argument("--in_csv", default=paths.COHORT_V2_CSV)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--llm_model_path", required=True,
                    help="explicit model path (no default — 8-backbone study)")
    ap.add_argument("--embed_model_path", default=ir.DEFAULT_EMBED_MODEL_PATH)
    ap.add_argument("--milvus_host", default=ir.MILVUS_HOST_DEFAULT)
    ap.add_argument("--milvus_port", type=int, default=ir.MILVUS_PORT_DEFAULT)
    ap.add_argument("--gpu_id", default="1",
                    help="default 1 (GPU 6 = CR-MSR, do not use)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42,
                    help="subsample seed — must match the MAIN run (42)")
    ap.add_argument("--evidence_mode", default="concat", choices=["concat", "fused"],
                    help="concat = e0_probe_runner protocol (MAIN npz); "
                         "fused = imaging_runner_v2 msrc (manuscript §4.1)")
    ap.add_argument("--main_npz", default="",
                    help="MAIN probe npz for stay-id alignment + prediction-agreement check")
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--resume", action="store_true",
                    help="skip stay_ids already present in out_csv")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    print(f"[E0X] deployed-protocol surface signals | env={paths.ENV} "
          f"seed={args.seed} n={args.n} evidence={args.evidence_mode}")
    print(f"  LLM={args.llm_model_path}")
    print(f"  Embed={args.embed_model_path}  Milvus={args.milvus_host}:{args.milvus_port}")

    df = ir.read_csv_robust(args.in_csv)
    print(f"  cohort rows={len(df):,}")
    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        print(f"  sampled={len(df):,} (df.sample(n, random_state=seed) — MAIN convention)")

    # ---- MAIN alignment check (design §2: mandatory before long runs) ----
    main_bits = {}
    if args.main_npz:
        z = np.load(args.main_npz, allow_pickle=True)
        m_sids = [int(s) for s in z["stay_ids"]]
        m_valid = z["valid"]; m_pred = z["pred_bits"]
        ours = set(int(s) for s in df["stay_id"])
        theirs = set(m_sids)
        inter = ours & theirs
        print(f"  [align] ours={len(ours)}  main_npz={len(theirs)}  overlap={len(inter)}")
        if ours != theirs:
            missing, extra = len(ours - theirs), len(theirs - ours)
            print(f"  [align][WARN] stay_id sets differ (ours-only={missing}, "
                  f"npz-only={extra}) — investigate before trusting same-sample claims")
        if len(inter) < 0.95 * len(ours):
            raise SystemExit("[align][FATAL] <95% overlap with MAIN npz — wrong "
                             "subsample (check --seed/--n/--in_csv). Aborting.")
        for i, sid in enumerate(m_sids):
            if int(m_valid[i]) == 1:
                main_bits[sid] = np.asarray(m_pred[i], dtype=np.int8)

    print("[1] load LLM ...", flush=True)
    e0r.load_llm(args.llm_model_path)
    print("[2] load embedder + Milvus ...", flush=True)
    ir.load_embedder(args.embed_model_path)
    ir.connect_milvus(args.milvus_host, args.milvus_port)
    e0r.measure_norm_consts()

    # ---- resume ----
    rows, done = [], set()
    if args.resume and Path(args.out_csv).exists():
        prev = pd.read_csv(args.out_csv)
        rows = prev.to_dict("records")
        done = set(int(s) for s in prev["stay_id"] if pd.notna(s))
        print(f"  [resume] loaded {len(done)} done stays from {args.out_csv}")

    def _save():
        out = pd.DataFrame(rows)
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)

    n_agree = n_cmp = 0
    jac_sum = 0.0
    print(f"[3] processing {len(df):,} stays ({len(done)} already done) ...", flush=True)
    t0 = time.time(); n_new = 0
    for row in df.to_dict("records"):
        sid = row.get("stay_id")
        try:
            if int(sid) in done:
                continue
        except (TypeError, ValueError):
            pass
        rec = {"stay_id": sid}
        try:
            case = ir._make_case_dict(row)
            case_text = ir.build_case_text_ed(case)
            qvec = ir.embed_query(case_text)

            # one retrieval per source (top-5); slices reused across arms
            hits_by_src = {sk: e0r.search_cosine(SRC_COLL[sk], qvec, TOP_K_FUSE)
                           for sk in SRC_ORDER}

            # retrieval sufficiency — E0 definition: max cosine over the source's
            # top-3 (TOP_K_SINGLE); S_max/S_mean aggregate across the 3 sources
            S = {sk: max((h["score"] for h in hits_by_src[sk][:TOP_K_CONCAT]),
                         default=0.0) for sk in SRC_ORDER}
            rec["S_max"] = float(max(S.values()))
            rec["S_mean"] = float(np.mean(list(S.values())))
            for sk in SRC_ORDER:
                rec[f"{sk}_strength"] = float(S[sk])

            # ---- deployed arm: greedy + token logprob + self-consistency ----
            ev = (fused_evidence(hits_by_src) if args.evidence_mode == "fused"
                  else concat_evidence(hits_by_src))
            prompt = ir.build_prompt(case_text, ev, mode="multi_label")

            raw, conf = e0r.call_llm_conf(prompt, max_new_tokens=args.max_new_tokens)
            bits, flag = parse_bits(raw)
            rec["dep_conf"] = float(conf)
            rec["U_lp"] = float(1.0 - conf)
            rec["dep_flag"] = flag
            for L in ALL_LABELS: rec[f"dep__{L}"] = bits[L]
            pred_set = {L for L in LABELS if bits[L] == 1}
            rec["pred_card"] = len(pred_set)

            # agreement with the MAIN greedy prediction (settles evidence_mode)
            try:
                mb = main_bits.get(int(sid))
            except (TypeError, ValueError):
                mb = None
            if mb is not None and flag == "ok":
                mset = {LABELS[j] for j in range(len(LABELS)) if int(mb[j]) == 1}
                n_cmp += 1
                n_agree += int(mset == pred_set)
                jac_sum += _jaccard(mset, pred_set)

            samples = e0u.sample_llm(prompt, args.n_samples, args.temperature,
                                     args.top_p, args.max_new_tokens)
            sets = [e0u.text_to_set(s) for s in samples]
            U_sc, U_lab, nv = e0u.uncertainty_from_samples(sets)
            rec["U_sc"] = U_sc; rec["U_lab"] = U_lab; rec["n_valid_sc"] = nv

            # ---- per-source arm: conflict (E0 definition, top-3 evidence) ----
            sets_by_src, bits_by_src, noimg_by_src = {}, {}, {}
            for sk in SRC_ORDER:
                ev_s = ir._build_evidence_text(hits_by_src[sk][:TOP_K_CONCAT])
                raw_s, conf_s = e0r.call_llm_conf(ir.build_prompt(
                    case_text, ev_s, mode="multi_label"), args.max_new_tokens)
                b_s, fl_s = parse_bits(raw_s)
                rec[f"{sk}_conf"] = float(conf_s); rec[f"{sk}_flag"] = fl_s
                for L in ALL_LABELS: rec[f"{sk}__{L}"] = b_s[L]
                sets_by_src[sk] = {L for L in LABELS if b_s[L] == 1}
                bits_by_src[sk] = b_s
                noimg_by_src[sk] = b_s[NO_IMG]
            rec["D_jac"] = d_jac([sets_by_src[sk] for sk in SRC_ORDER])
            rec["D_lab"] = d_lab(bits_by_src)
            rec["D_dec"] = d_dec(sets_by_src, noimg_by_src)
            rec["D_high"] = d_high(sets_by_src)

            # ---- error labels on the DEPLOYED greedy prediction ----
            rec.update(error_labels(row, pred_set))
        except Exception as e:
            rec["error"] = str(e)[:200]
        rows.append(rec)
        n_new += 1
        if n_new % args.ckpt_every == 0 or len(rows) == len(df):
            _save()
            el = time.time() - t0; rate = n_new / max(el, 1e-6)
            print(f"    +{n_new} new (total={len(rows)}/{len(df)}, {el:.0f}s, "
                  f"{1/rate:.2f}s/stay, ETA {(len(df)-len(rows))/max(rate,1e-6)/3600:.1f}h)",
                  flush=True)

    _save()
    out = pd.DataFrame(rows)
    print(f"[4] saved {len(out):,} rows → {args.out_csv} ({time.time()-t0:.0f}s)")
    for c in ["dep_flag"] + [f"{sk}_flag" for sk in SRC_ORDER]:
        if c in out.columns:
            print(f"  {c}:", out[c].value_counts().to_dict())
    ok = out[out.get("dep_flag", pd.Series(dtype=str)) == "ok"]
    if len(ok):
        print(f"  base rates (deployed greedy, flag=ok, n={len(ok)}): "
              f"ct_miss={ok['ct_miss'].mean():.3f}  y_err={ok['y_err'].mean():.3f}  "
              f"pred_card={ok['pred_card'].mean():.2f}")
    if n_cmp:
        print(f"  [align] MAIN greedy agreement: exact {n_agree}/{n_cmp} "
              f"({100*n_agree/max(n_cmp,1):.1f}%), mean Jaccard {jac_sum/n_cmp:.3f} "
              f"(low agreement → wrong --evidence_mode for this backbone)")


if __name__ == "__main__":
    main()
