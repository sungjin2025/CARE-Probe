#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0_uncertainty_runner.py — CARE-MSR E0: proper uncertainty via self-consistency.

The first E0 pass used token-logprob-over-JSON as U, which is dominated by
boilerplate tokens (keys/braces/zeros) → AUC≈0.5 (useless). This recomputes a
clean uncertainty signal by SAMPLING the LLM N times on the multi-source-
conditioned prompt and measuring answer instability:
    U_sc  = mean pairwise Jaccard DISTANCE among the N sampled label sets
    U_lab = mean per-label vote-entropy over the union of predicted labels
(0 = perfectly self-consistent/certain ; high = unstable/uncertain)

Same 500 stays as e0_conflict_runner (same --n/--seed → identical df.sample).
Merge output with e0_preds_*.csv on stay_id, then re-run e0_analyze.py --u_csv.

Usage (server):
  CRMSR_ENV=server python e0_uncertainty_runner.py \
    --out_csv runs_e0/e0_unc_n500_seed42.csv \
    --llm_model_path   /path/to/models/Qwen/Qwen2.5-14B-Instruct \
    --embed_model_path /path/to/models/Alibaba-NLP/gte-large-en-v1.5 \
    --gpu_id 6 --n 500 --seed 42 --n_samples 5 --temperature 0.7
"""
from __future__ import annotations
import os, sys, time, argparse, itertools

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _gpu = "6"
    if "--gpu_id" in sys.argv:
        try: _gpu = sys.argv[sys.argv.index("--gpu_id") + 1]
        except (IndexError, ValueError): pass
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import numpy as np
import pandas as pd
from pathlib import Path

import paths
import imaging_runner_v2 as ir
import e0_conflict_runner as e0r   # reuse LLM loader, cosine search, label space

LABELS = e0r.LABELS
ALL_LABELS = e0r.ALL_LABELS
SRC_COLL = e0r.SRC_COLL
SRC_ORDER = e0r.SRC_ORDER


def sample_llm(prompt: str, n: int, temperature: float, top_p: float,
               max_new_tokens: int) -> list:
    """Sample n completions in one batched generate call. Returns list[str]."""
    import torch
    tok, model, device = e0r._LLM["tok"], e0r._LLM["model"], e0r._LLM["device"]
    text_in = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tok(text_in, return_tensors="pt", truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    plen = int(inputs["input_ids"].shape[-1])
    return [tok.decode(seq[plen:], skip_special_tokens=True).strip() for seq in out]


def text_to_set(raw: str):
    """Parse one sample → frozenset of imaging labels (pre-gate). None if invalid/abstain."""
    if ir.is_refusal_text(raw):
        return None
    obj = ir.extract_first_json_object(raw)
    if obj is None:
        return None
    pb, meta = ir.normalize_imaging_json(obj, mode="multi_label")
    if meta.get("is_abstain_json"):
        return None
    fixed, _ = ir.enforce_no_imaging_mutex(pb, label_space=LABELS)
    return frozenset(L for L in LABELS if int(fixed.get(L, 0)) == 1)


def jaccard_dist(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    return 1.0 - (len(a & b) / len(u) if u else 1.0)


def uncertainty_from_samples(sets: list) -> tuple:
    """sets: list of frozenset (or None for failed). Returns (U_sc, U_lab, n_valid)."""
    valid = [s for s in sets if s is not None]
    nv = len(valid)
    if nv <= 1:
        # all-failed → max uncertainty; single-valid → 0 pairs → treat as uncertain
        return (1.0 if nv == 0 else 0.0), (1.0 if nv == 0 else 0.0), nv
    # U_sc: mean pairwise Jaccard distance
    U_sc = float(np.mean([jaccard_dist(a, b) for a, b in itertools.combinations(valid, 2)]))
    # U_lab: mean Bernoulli entropy over union of predicted labels
    union = set().union(*valid)
    if not union:
        U_lab = 0.0
    else:
        ents = []
        for L in union:
            p = sum(1 for s in valid if L in s) / nv
            if p in (0.0, 1.0):
                ents.append(0.0)
            else:
                ents.append(-(p*np.log2(p) + (1-p)*np.log2(1-p)))
        U_lab = float(np.mean(ents))
    return U_sc, U_lab, nv


def build_multisource_evidence(qvec, top_k: int) -> str:
    parts = []
    for sk in SRC_ORDER:
        hits = e0r.search_cosine(SRC_COLL[sk], qvec, top_k)
        txt = ir._build_evidence_text(hits)
        if txt.strip():
            parts.append(f"### Source {sk}\n{txt}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="E0 uncertainty via self-consistency")
    ap.add_argument("--in_csv", default=paths.COHORT_V2_CSV)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--llm_model_path", default=ir.DEFAULT_LLM_MODEL_PATH)
    ap.add_argument("--embed_model_path", default=ir.DEFAULT_EMBED_MODEL_PATH)
    ap.add_argument("--milvus_host", default=ir.MILVUS_HOST_DEFAULT)
    ap.add_argument("--milvus_port", type=int, default=ir.MILVUS_PORT_DEFAULT)
    ap.add_argument("--gpu_id", default="6")
    ap.add_argument("--top_k", type=int, default=ir.TOP_K_SINGLE)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    args = ap.parse_args()

    import random, torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    print(f"[E0-U] self-consistency | env={paths.ENV} seed={args.seed} n={args.n} "
          f"N_samples={args.n_samples} temp={args.temperature}")

    df = ir.read_csv_robust(args.in_csv)
    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"  stays={len(df):,} (must match e0_conflict_runner sample)")

    print("[1] load LLM ...", flush=True)
    e0r.load_llm(args.llm_model_path)
    print("[2] load embedder + Milvus ...", flush=True)
    ir.load_embedder(args.embed_model_path)
    ir.connect_milvus(args.milvus_host, args.milvus_port)
    e0r.measure_norm_consts()

    print(f"[3] sampling {args.n_samples}× per stay over {len(df):,} stays ...", flush=True)
    t0 = time.time(); rows = []
    for i, row in enumerate(df.to_dict("records")):
        rec = {"stay_id": row.get("stay_id")}
        try:
            case = ir._make_case_dict(row)
            case_text = ir.build_case_text_ed(case)
            qvec = ir.embed_query(case_text)
            ev = build_multisource_evidence(qvec, args.top_k)
            prompt = ir.build_prompt(case_text, ev, mode="multi_label")
            samples = sample_llm(prompt, args.n_samples, args.temperature,
                                 args.top_p, args.max_new_tokens)
            sets = [text_to_set(s) for s in samples]
            U_sc, U_lab, nv = uncertainty_from_samples(sets)
            rec["U_sc"] = U_sc; rec["U_lab"] = U_lab; rec["n_valid"] = nv
        except Exception as e:
            rec["error"] = str(e)[:200]
            rec["U_sc"] = np.nan; rec["U_lab"] = np.nan; rec["n_valid"] = 0
        rows.append(rec)
        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            el = time.time() - t0; rate = (i + 1) / max(el, 1e-6)
            print(f"    {i+1}/{len(df)} ({el:.0f}s, {1/rate:.2f}s/stay, "
                  f"ETA {(len(df)-(i+1))/max(rate,1e-6):.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[4] saved {len(out):,} rows → {args.out_csv} ({time.time()-t0:.0f}s)")
    print(f"  U_sc  mean={out['U_sc'].mean():.3f} std={out['U_sc'].std():.3f}")
    print(f"  U_lab mean={out['U_lab'].mean():.3f} std={out['U_lab'].std():.3f}")
    print(f"  n_valid mean={out['n_valid'].mean():.2f}/{args.n_samples}")


if __name__ == "__main__":
    main()
