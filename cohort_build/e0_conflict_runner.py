#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0_conflict_runner.py — CARE-MSR E0 cross-source conflict analysis (data collection).

For each ED stay, produce PRE-GATE multi-label predictions from the SAME LLM
conditioned on each evidence source separately, plus vanilla (no evidence):
    y_∅ (vanilla) , y_P (PubMed) , y_W (Wikipedia) , y_G (Guideline)
Also records per-source retrieval sufficiency S (true cosine) and per-prediction
token-logprob confidence U. NO strength gate is applied → conflict reflects the
LLM's evidence-driven decision, isolated from the τ-gate / normalization bug.
(See proposals/E0_conflict_analysis_design.md)

Output CSV → e0_analyze.py computes D(x) conflict metrics + error correlation.

Usage (server):
  CRMSR_ENV=server python e0_conflict_runner.py \
    --out_csv runs_e0/e0_preds_n500_seed42.csv \
    --llm_model_path   /path/to/models/Qwen/Qwen2.5-14B-Instruct \
    --embed_model_path /path/to/models/Alibaba-NLP/gte-large-en-v1.5 \
    --gpu_id 6 --n 500 --seed 42
"""
from __future__ import annotations
import os, sys, time, json, argparse, random
from pathlib import Path

# ── Pin CUDA before any torch import (mirror imaging_runner_v2; default GPU 6) ──
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

import paths
import imaging_runner_v2 as ir   # reuse retrieval/prompt/parse helpers + label space

LABELS = list(ir.LABEL_SPACE_V2)
NO_IMG = ir.NO_IMAGING_LABEL
ALL_LABELS = LABELS + [NO_IMG]

# source key -> Milvus collection (reuse runner's mapping)
SRC_COLL = {
    "P": ir.COLLECTION_PUBMED,
    "W": ir.COLLECTION_MEDWIKI,
    "G": ir.COLLECTION_GUIDELINE,
}
SRC_ORDER = ["P", "W", "G"]

# Lightweight cosine recovery (E0-local; does NOT touch shared Milvus).
# Defaults from _check_norms.py (2026-06-02); remeasured at startup when possible.
NORM_CONST = {ir.COLLECTION_PUBMED: 23.72, ir.COLLECTION_MEDWIKI: 23.98,
              ir.COLLECTION_GUIDELINE: 1.00}

_LLM = {"tok": None, "model": None, "device": None}


# --- Fallback chat templates (Jinja) for models shipped without tokenizer.chat_template ---
# OpenBioLLM-8B (and some Llama-3 finetunes) omit chat_template -> apply_chat_template() crashes.
# We inject the canonical template for the base family so all 3 call sites work unchanged.
_LLAMA3_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    " + message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)
_PHI3_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|' + message['role'] + '|>\n' + message['content'] + '<|end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|assistant|>\n' }}{%- endif %}"
)
_CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)


def _ensure_chat_template(tok, model_path: str):
    """If the tokenizer ships no chat_template, inject the base family's template."""
    if getattr(tok, "chat_template", None):
        return
    name = model_path.lower()
    if any(k in name for k in ("llama", "med42", "openbio")):
        tok.chat_template = _LLAMA3_CHAT_TEMPLATE
        fam = "llama3"
    elif "phi" in name:
        tok.chat_template = _PHI3_CHAT_TEMPLATE
        fam = "phi3"
    else:
        tok.chat_template = _CHATML_TEMPLATE
        fam = "chatml"
    print(f"[load_llm] tokenizer had no chat_template; injected {fam} fallback for {model_path}",
          flush=True)


def _patch_dynamic_cache():
    """Phi-3 (and some) bundled remote code reads `past_key_values.seen_tokens`,
    removed from newer transformers DynamicCache (-> use get_seq_length()).
    Inject a read-only shim so old modeling code runs under the installed version.
    No-op if the attribute already exists or transformers lacks DynamicCache."""
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "seen_tokens"):
        return

    def _seen(self):
        try:
            return self.get_seq_length()
        except Exception:
            return 0
    DynamicCache.seen_tokens = property(_seen)
    print("[load_llm] patched DynamicCache.seen_tokens shim (old remote-code compat)", flush=True)


def load_llm(model_path: str):
    if _LLM["model"] is not None:
        return
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _patch_dynamic_cache()
    # Phi-3 ships modeling code that reads the removed DynamicCache.seen_tokens API.
    # Use the NATIVE transformers Phi3 impl (trust_remote_code=False) to bypass it.
    name = model_path.lower()
    trc = "phi" not in name
    print(f"[load_llm] transformers={transformers.__version__} trust_remote_code={trc} "
          f"for {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trc)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    _ensure_chat_template(tok, model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=trc)
    model.eval()
    _LLM.update(tok=tok, model=model, device=next(model.parameters()).device)


def _confidence_from_generate(gen, prompt_len: int) -> float:
    """exp(mean token logprob) of generated tokens ∈ (0,1]; 0.0 on failure."""
    try:
        import torch, torch.nn.functional as F
        if getattr(gen, "scores", None) is None or getattr(gen, "sequences", None) is None:
            return 0.0
        seq = gen.sequences[0]
        gen_tokens = seq[int(prompt_len):]
        T = len(gen.scores)
        if T <= 0 or gen_tokens.numel() < T:
            return 0.0
        logps = []
        for t, logits in enumerate(gen.scores):
            lp = F.log_softmax(logits[0], dim=-1)
            logps.append(float(lp[int(gen_tokens[t].item())].item()))
        if not logps:
            return 0.0
        avg = sum(logps) / len(logps)
        return float(torch.exp(torch.tensor(avg)).clamp(0, 1).item())
    except Exception:
        return 0.0


def call_llm_conf(prompt: str, max_new_tokens: int = 400) -> tuple:
    """Returns (text, confidence). Greedy decode + token-logprob confidence."""
    import torch
    tok, model, device = _LLM["tok"], _LLM["model"], _LLM["device"]
    text_in = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tok(text_in, return_tensors="pt", truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True, output_scores=True)
    plen = int(inputs["input_ids"].shape[-1])
    conf = _confidence_from_generate(gen, plen)
    txt = tok.decode(gen.sequences[0][plen:], skip_special_tokens=True).strip()
    return txt, float(conf)


def measure_norm_consts():
    """Best-effort remeasure of per-collection mean ‖d‖ (else keep defaults)."""
    try:
        from pymilvus import Collection, DataType
        vec_types = {DataType.FLOAT_VECTOR}
        for t in ("FLOAT16_VECTOR", "BFLOAT16_VECTOR"):
            if hasattr(DataType, t): vec_types.add(getattr(DataType, t))
        for name in SRC_COLL.values():
            c = Collection(name); c.load()
            pk = pkt = vec = None
            for f in c.schema.fields:
                if getattr(f, "is_primary", False): pk, pkt = f.name, f.dtype
                if f.dtype in vec_types: vec = f.name
            expr = f'{pk} != ""' if pkt == DataType.VARCHAR else f"{pk} >= 0"
            rows = c.query(expr=expr, output_fields=[vec], limit=5)
            norms = [float(np.linalg.norm(np.asarray(r[vec], dtype=np.float32))) for r in rows]
            if norms:
                NORM_CONST[name] = float(np.mean(norms))
        print(f"  norm consts = { {k: round(v,2) for k,v in NORM_CONST.items()} }")
    except Exception as e:
        print(f"  [warn] norm remeasure failed ({e}); using defaults {NORM_CONST}")


def search_cosine(coll_name: str, qvec, top_k: int) -> list:
    """Raw IP search → cosine = IP / ‖d‖ (per-collection const). Returns hits."""
    col = ir.get_collection(coll_name)
    res = col.search(data=[qvec.tolist()], anns_field=ir.MILVUS_EMBED_FIELD,
                     param=ir.MILVUS_SEARCH_PARAMS, limit=top_k,
                     output_fields=[ir.MILVUS_TEXT_FIELD])
    denom = max(NORM_CONST.get(coll_name, 1.0), 1e-6)
    hits = []
    for h in res[0]:
        try: text = h.entity.get(ir.MILVUS_TEXT_FIELD, "") or ""
        except Exception: text = ""
        cos = float(h.score) / denom
        hits.append({"text": text, "score": max(0.0, min(1.0, cos))})
    return hits


def predict_pregate(case_text: str, evidence_text: str, max_new_tokens: int) -> tuple:
    """LLM → pre-gate label bits (normalize + no_imaging mutex; NO strength gate).
    Returns (bits_dict, conf, flag) where flag ∈ {'ok','parse_fail','refusal','abstain_json'}."""
    prompt = ir.build_prompt(case_text, evidence_text, mode="multi_label")
    raw, conf = call_llm_conf(prompt, max_new_tokens=max_new_tokens)
    bits = {L: 0 for L in ALL_LABELS}
    if ir.is_refusal_text(raw):
        return bits, conf, "refusal"
    obj = ir.extract_first_json_object(raw)
    if obj is None:
        return bits, conf, "parse_fail"
    pred_bits, meta = ir.normalize_imaging_json(obj, mode="multi_label")
    if meta.get("is_abstain_json"):
        return bits, conf, "abstain_json"
    fixed, _ = ir.enforce_no_imaging_mutex(pred_bits, label_space=LABELS)
    for L in ALL_LABELS:
        bits[L] = int(fixed.get(L, 0))
    return bits, conf, "ok"


def parse_args():
    ap = argparse.ArgumentParser(description="E0 cross-source conflict data collection")
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
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--no_retrieval", action="store_true",
                    help="B-4 ablation: closed-book — compute ONLY the vanilla (no-evidence) "
                         "prediction + intrinsic logprob confidence; skip embedder/Milvus and the "
                         "per-source (P/W/G) retrieval. Sufficiency/cross-source conflict are "
                         "definitionally absent under no-retrieval. (Note: the vanilla columns are "
                         "identical to van_* in a full run — this flag is for cheap standalone runs.)")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    mode_str = "CLOSED-BOOK (no-retrieval; vanilla only)" if args.no_retrieval else "multi-source (vanilla + P/W/G)"
    print(f"[E0] conflict runner | env={paths.ENV} seed={args.seed} n={args.n} mode={mode_str}")
    print(f"  LLM={args.llm_model_path}")
    if not args.no_retrieval:
        print(f"  Embed={args.embed_model_path}  Milvus={args.milvus_host}:{args.milvus_port}")

    df = ir.read_csv_robust(args.in_csv)
    print(f"  cohort rows={len(df):,}")
    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        print(f"  sampled={len(df):,}")

    print("[1] load LLM ...", flush=True)
    load_llm(args.llm_model_path)
    if args.no_retrieval:
        print("[2] no_retrieval → SKIP embedder/Milvus (vanilla-only closed-book)", flush=True)
    else:
        print("[2] load embedder + Milvus ...", flush=True)
        ir.load_embedder(args.embed_model_path)
        ir.connect_milvus(args.milvus_host, args.milvus_port)
        measure_norm_consts()

    cond_desc = "vanilla only" if args.no_retrieval else "vanilla + 3 sources each"
    print(f"[3] processing {len(df):,} stays ({cond_desc}) ...", flush=True)
    t0 = time.time()
    rows = []
    src_key_to_label = {"P": "pubmed", "W": "wikipedia", "G": "guideline"}
    for i, row in enumerate(df.to_dict("records")):
        rec = {"stay_id": row.get("stay_id")}
        try:
            case = ir._make_case_dict(row)
            case_text = ir.build_case_text_ed(case)
            if not args.no_retrieval:
                qvec = ir.embed_query(case_text)

            # vanilla (no evidence) — always; this IS the closed-book intrinsic signal
            bits, conf, flag = predict_pregate(case_text, "", args.max_new_tokens)
            rec["van_conf"] = conf; rec["van_flag"] = flag
            for L in ALL_LABELS: rec[f"van__{L}"] = bits[L]

            # each source (skipped under --no_retrieval: sufficiency/conflict undefined)
            if not args.no_retrieval:
                for sk in SRC_ORDER:
                    coll = SRC_COLL[sk]
                    hits = search_cosine(coll, qvec, args.top_k)
                    S = max((h["score"] for h in hits), default=0.0)
                    ev = ir._build_evidence_text(hits)
                    bits, conf, flag = predict_pregate(case_text, ev, args.max_new_tokens)
                    rec[f"{sk}_strength"] = float(S)
                    rec[f"{sk}_conf"] = conf
                    rec[f"{sk}_flag"] = flag
                    for L in ALL_LABELS: rec[f"{sk}__{L}"] = bits[L]
        except Exception as e:
            rec["error"] = str(e)[:200]
        rows.append(rec)
        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            el = time.time() - t0; rate = (i + 1) / max(el, 1e-6)
            print(f"    {i+1}/{len(df)} ({el:.0f}s, {1/rate:.2f}s/stay, "
                  f"ETA {(len(df)-(i+1))/max(rate,1e-6):.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[4] saved {len(out):,} rows → {args.out_csv} ({time.time()-t0:.0f}s)")

    # quick QC
    for sk in ["van"] + SRC_ORDER:
        fl = out.get(f"{sk}_flag")
        if fl is not None:
            print(f"  {sk} flags:", fl.value_counts().to_dict())


if __name__ == "__main__":
    main()
