#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""imaging_runner_v2.py — CR-MSR unified runner (M1.5 baseline).

Supports 5 modes via --mode flag:
  vanilla        : LLM only (no retrieval)
  single_pubmed  : 1-source dense RAG (MedRAG_PubMed)
  single_medwiki : 1-source dense RAG (MedRAG_Wikipedia)
  single_guideline: 1-source dense RAG (Medical_Guidelines)
  msrc           : multi-source dense RAG (PubMed + MedWiki + Guideline, adaptive fusion)

Retrieval (M1.5 baseline = first paper stack FROZEN):
  - Dense-only via Milvus IP search
  - Embedder: Alibaba-NLP/gte-large-en-v1.5
  - No BM25, no reranker (those come in M2)

Decoder/Utils: policyA_decode_v2 + common_utils_v2 (V2 multi-label, no_imaging mutex, 3-way decision).

Default τ values preserve first paper semantics for V1→V2 mapping:
  - τ_CT = 0.60 (all CT_* labels)
  - τ_CXR = 0.72 (CXR_CHEST)
  - τ_US = 0.72 (all US_* labels)
  - τ_XMO = 0.72 (XR_*, MRI_*, OTHER_OTHER)
  - τ_none = 0.55 (pre-gate)
M3 conformal layer will REPLACE these heuristic τ with calibrated thresholds.

Usage (server):
  CRMSR_ENV=server python imaging_runner_v2.py \\
    --mode msrc \\
    --in_csv /path/to/cohort_v2.csv \\
    --out_csv /path/to/pred_msrc_seed1.csv \\
    --llm_model_path /mnt/.../Qwen2.5-14B-Instruct \\
    --embed_model_path /mnt/.../gte-large-en-v1.5 \\
    --milvus_host localhost --milvus_port 19530 \\
    --gpu_id 0 --n 1000 --seed 42
"""
from __future__ import annotations
import os, sys, time, json, argparse, traceback, random
from pathlib import Path

# ── Pin CUDA device BEFORE any torch/CUDA import (must precede all heavy imports) ──
# Priority: existing CUDA_VISIBLE_DEVICES env  >  --gpu_id arg  >  default GPU 6.
# (Setting it later, e.g. inside load_llm, is ignored once CUDA is initialized.)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _gpu_pin = "6"  # default: always GPU 6 unless overridden (user rule)
    if "--gpu_id" in sys.argv:
        try:
            _gpu_pin = sys.argv[sys.argv.index("--gpu_id") + 1]
        except (IndexError, ValueError):
            pass
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_pin

# Force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

# V2 modules (always importable on dev machine — no heavy deps)
import paths
from policyA_decode_v2 import (
    LABEL_SPACE_V2, NO_IMAGING_LABEL, decode_policyA_from_bits,
)
from common_utils_v2 import (
    read_csv_robust, build_case_text_ed, build_prompt, build_repair_prompt,
    extract_first_json_object, normalize_imaging_json,
    apply_none_gate, apply_label_gates, enforce_no_imaging_mutex,
)

# Offline-friendly HF env
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ====================================================================
# Configuration defaults (override via CLI)
# ====================================================================
DEFAULT_LLM_MODEL_PATH    = "/path/to/models/Qwen2.5-14B-Instruct"
DEFAULT_EMBED_MODEL_PATH  = "/path/to/models/Alibaba-NLP/gte-large-en-v1.5"

MILVUS_HOST_DEFAULT       = "localhost"
MILVUS_PORT_DEFAULT       = 19530
COLLECTION_PUBMED         = "MedRAG_PubMed"
COLLECTION_MEDWIKI        = "MedRAG_Wikipedia"
COLLECTION_GUIDELINE      = "Medical_Guidelines"
MILVUS_EMBED_FIELD        = "vector"
MILVUS_TEXT_FIELD         = "content"
MILVUS_SEARCH_PARAMS      = {"metric_type": "IP", "params": {"nprobe": 10}}
DOC_EMBEDDINGS_NORMALIZED = True  # IP == cosine when normalized

# First-paper τ values (preserved for V1→V2 mapping; M3 conformal will replace)
DEFAULT_TAU_NONE = 0.55
DEFAULT_TAU_CT   = 0.60
DEFAULT_TAU_CXR  = 0.72
DEFAULT_TAU_US   = 0.72
DEFAULT_TAU_XMO  = 0.72  # XR_*, MRI_*, OTHER_OTHER

# Retrieval params
TOP_K_SINGLE      = 3   # single-RAG modes
TOP_K_PER_SOURCE  = 5   # msrc per-source
TOP_K_GLOBAL      = 4   # msrc adaptive fusion global cap
MAX_EVIDENCE_CHARS_PER_DOC = 600

# Adaptive fusion weights (first paper; M2 will replace with RRF)
SOURCE_WEIGHTS = {"PubMed": 0.42, "MedWiki": 0.16, "Guideline": 0.42}
SOURCE_TO_COLLECTION = {
    "PubMed":   COLLECTION_PUBMED,
    "MedWiki":  COLLECTION_MEDWIKI,
    "Guideline": COLLECTION_GUIDELINE,
}

# Disagreement penalty α(σ) — first paper
def alpha_of_sigma(sigma: float) -> float:
    if sigma > 0.15: return 0.85
    if sigma > 0.10: return 0.90
    if sigma > 0.05: return 0.95
    return 1.00


def build_default_tau_dict() -> dict:
    """Per-label τ dict matching first paper semantics (V1 → V2)."""
    d = {}
    for L in LABEL_SPACE_V2:
        if L.startswith("CT_"):
            d[L] = DEFAULT_TAU_CT
        elif L == "CXR_CHEST":
            d[L] = DEFAULT_TAU_CXR
        elif L.startswith("US_"):
            d[L] = DEFAULT_TAU_US
        else:  # XR_*, MRI_*, OTHER_OTHER
            d[L] = DEFAULT_TAU_XMO
    return d


# ====================================================================
# Lazy-loaded heavy components (avoid import errors on dev machine)
# ====================================================================
_LLM   = {"tokenizer": None, "model": None, "device": None}
_EMBED = {"model": None}
_MILVUS_COLLECTIONS: dict = {}


def load_llm(model_path: str, gpu_id: str = "0"):
    if _LLM["model"] is not None:
        return
    # CUDA_VISIBLE_DEVICES is pinned at module top (before any torch import); do NOT override here.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    _LLM["tokenizer"] = tok
    _LLM["model"] = model
    _LLM["device"] = next(model.parameters()).device


def call_llm(prompt: str, max_new_tokens: int = 400, temperature: float = 0.0) -> str:
    import torch
    tok = _LLM["tokenizer"]; model = _LLM["model"]; device = _LLM["device"]
    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        text = prompt
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-5),
            pad_token_id=tok.pad_token_id,
        )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


def load_embedder(embed_model_path: str):
    if _EMBED["model"] is not None:
        return
    from sentence_transformers import SentenceTransformer
    _EMBED["model"] = SentenceTransformer(embed_model_path, trust_remote_code=True)


def embed_query(query: str):
    import numpy as np
    vec = _EMBED["model"].encode(query, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype="float32")


def connect_milvus(host: str, port: int):
    import os
    from pymilvus import connections
    token = os.environ.get("MILVUS_TOKEN", "")          # auth: inject at runtime, never hardcode
    kw = {"alias": "default", "host": host, "port": str(port)}
    if token:
        kw["token"] = token
    connections.connect(**kw)


def get_collection(name: str):
    if name not in _MILVUS_COLLECTIONS:
        from pymilvus import Collection
        col = Collection(name)
        col.load()
        _MILVUS_COLLECTIONS[name] = col
    return _MILVUS_COLLECTIONS[name]


def search_collection(name: str, query_vec, top_k: int):
    col = get_collection(name)
    res = col.search(
        data=[query_vec.tolist()],
        anns_field=MILVUS_EMBED_FIELD,
        param=MILVUS_SEARCH_PARAMS,
        limit=top_k,
        output_fields=[MILVUS_TEXT_FIELD],
    )
    hits = []
    for h in res[0]:
        text = ""
        try:
            text = h.entity.get(MILVUS_TEXT_FIELD, "") or ""
        except Exception:
            pass
        score = float(h.score)
        if DOC_EMBEDDINGS_NORMALIZED:
            score = max(0.0, min(1.0, score))
        hits.append({"text": text, "score": score})
    return hits


# ====================================================================
# Retrieval per mode + evidence text builder
# ====================================================================
def retrieve_for_mode(mode: str, query_vec) -> dict:
    """Returns {evidence_text, strength, passages, [sigma, alpha, m_max for msrc]}."""
    if mode == "vanilla":
        return {"evidence_text": "", "strength": 1.0, "passages": []}

    if mode.startswith("single_"):
        src_key_map = {"pubmed": "PubMed", "medwiki": "MedWiki", "guideline": "Guideline"}
        sub = mode.split("_", 1)[1]
        src_key = src_key_map.get(sub)
        if src_key is None:
            raise ValueError(f"Unknown single mode: {mode}")
        col_name = SOURCE_TO_COLLECTION[src_key]
        hits = search_collection(col_name, query_vec, top_k=TOP_K_SINGLE)
        strength = max((h["score"] for h in hits), default=0.0)
        strength = max(0.0, min(1.0, strength))
        return {"evidence_text": _build_evidence_text(hits),
                "strength": strength, "passages": hits}

    if mode == "msrc":
        import numpy as np
        per_source = {}
        for src_key, col_name in SOURCE_TO_COLLECTION.items():
            per_source[src_key] = search_collection(col_name, query_vec, top_k=TOP_K_PER_SOURCE)
        all_scores = [h["score"] for hs in per_source.values() for h in hs]
        m_max = max(all_scores) if all_scores else 0.0
        source_maxes = [max([h["score"] for h in per_source[s]], default=0.0)
                        for s in per_source]
        nonzero = [x for x in source_maxes if x > 0]
        sigma = float(np.std(nonzero, ddof=0)) if len(nonzero) >= 2 else 0.0
        alpha = alpha_of_sigma(sigma)
        strength = max(0.0, min(1.0, alpha * m_max))
        evidence_text = _build_msrc_evidence(per_source)
        return {"evidence_text": evidence_text, "strength": strength,
                "passages": per_source, "sigma": sigma, "alpha": alpha, "m_max": m_max}

    raise ValueError(f"Unknown mode: {mode}")


def _build_evidence_text(hits, max_chars: int = MAX_EVIDENCE_CHARS_PER_DOC) -> str:
    lines = []
    for i, h in enumerate(hits):
        t = (h.get("text") or "").strip()
        if not t: continue
        if len(t) > max_chars:
            t = t[:max_chars] + "..."
        lines.append(f"[{i+1}] {t}")
    return "\n\n".join(lines)


def _build_msrc_evidence(per_source, top_k_global: int = TOP_K_GLOBAL,
                         max_chars: int = MAX_EVIDENCE_CHARS_PER_DOC) -> str:
    """Adaptive fusion (first paper): min-max norm per source, k-means K=2 retain hi,
    weighted sum, global top-K_g."""
    import numpy as np
    fused = []
    for src_key, hits in per_source.items():
        if not hits: continue
        scores = np.array([h["score"] for h in hits], dtype=float)
        if scores.max() == scores.min():
            tilde = np.ones_like(scores)
        else:
            tilde = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels = km.fit_predict(tilde.reshape(-1, 1))
            cent = [tilde[labels == k].mean() if (labels == k).any() else -1
                    for k in (0, 1)]
            mask = labels == int(np.argmax(cent))
        except Exception:
            mask = tilde >= np.percentile(tilde, 50)
        w_k = SOURCE_WEIGHTS.get(src_key, 1.0 / 3)
        for h, t, m in zip(hits, tilde, mask):
            if not m: continue
            fused.append({"text": h["text"], "fused_score": float(w_k * t), "source": src_key})
    fused.sort(key=lambda x: -x["fused_score"])
    selected = fused[:top_k_global]
    lines = []
    for i, h in enumerate(selected):
        t = (h["text"] or "").strip()
        if not t: continue
        if len(t) > max_chars:
            t = t[:max_chars] + "..."
        lines.append(f"[{i+1}|{h['source']}] {t}")
    return "\n\n".join(lines)


# ====================================================================
# Refusal/policy text detection
# ====================================================================
_REFUSAL_PATTERNS = [
    "i can't", "i cannot", "i can’t", "can't fulfill", "cannot fulfill",
    "medical advice", "i'm not able", "i am not able", "i can't provide",
    "i cannot provide", "seek immediate medical attention",
]

def is_refusal_text(s: str) -> bool:
    t = (s or "").lower()
    return any(p in t for p in _REFUSAL_PATTERNS)


# ====================================================================
# Per-stay processing
# ====================================================================
CASE_FIELDS = [
    "chiefcomplaint", "triage_note", "hpi", "hpi_clean", "pmh",
    "medications", "allergies", "vitals", "physical_exam",
    "assessment", "plan", "diagnosis", "age", "gender", "race",
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
    "pain", "icd10_primary",
]


def _make_case_dict(row: dict) -> dict:
    out = {}
    for k in CASE_FIELDS:
        v = row.get(k)
        if v is None: continue
        if isinstance(v, float) and pd.isna(v): continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none"):
            out[k] = s
    return out


def process_stay(row: dict, mode: str, args, tau_dict: dict) -> dict:
    case = _make_case_dict(row)
    case_text = build_case_text_ed(case)
    error_log = ""

    # Retrieval
    retr = {"evidence_text": "", "strength": 1.0, "passages": []}
    try:
        if mode != "vanilla":
            qvec = embed_query(case_text)
            retr = retrieve_for_mode(mode, qvec)
    except Exception as e:
        error_log = f"retrieval_fail:{e}"
        retr = {"evidence_text": "", "strength": 0.0, "passages": []}

    # Pre-gate
    gated_pre, pre_reason = apply_none_gate(retr["strength"], args.tau_none)
    if gated_pre:
        return _build_row(row, mode, retr,
                          result=decode_policyA_from_bits({}, abstain_reason=pre_reason),
                          error_log=error_log)

    # LLM call
    prompt = build_prompt(case_text, retr["evidence_text"], mode="multi_label")
    raw = ""
    try:
        raw = call_llm(prompt, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature)
    except Exception as e:
        return _build_row(row, mode, retr,
                          result=decode_policyA_from_bits({}, abstain_reason="llm_failed"),
                          error_log=_append_err(error_log, str(e)), raw_text=raw)

    # Refusal sniff
    if is_refusal_text(raw):
        return _build_row(row, mode, retr,
                          result=decode_policyA_from_bits({}, abstain_reason="refusal"),
                          error_log=error_log, raw_text=raw)

    # Parse JSON
    obj = extract_first_json_object(raw)
    if obj is None:
        try:
            repair = build_repair_prompt(raw)
            raw2 = call_llm(repair, max_new_tokens=args.max_new_tokens, temperature=0.0)
            obj = extract_first_json_object(raw2)
            raw = raw + "\n---REPAIR---\n" + (raw2 or "")
        except Exception:
            pass
    if obj is None:
        return _build_row(row, mode, retr,
                          result=decode_policyA_from_bits({}, abstain_reason="parse_fail"),
                          error_log=error_log, raw_text=raw)

    # Normalize → bits
    pred_bits, meta = normalize_imaging_json(obj, mode="multi_label")

    # Explicit abstain JSON
    if meta.get("is_abstain_json"):
        ar = meta.get("abstain_reason") or "llm_abstain_json"
        return _build_row(row, mode, retr,
                          result=decode_policyA_from_bits(pred_bits, abstain_reason=ar),
                          error_log=error_log, raw_text=raw)

    # Post-gate per label
    gated, gate_reasons = apply_label_gates(
        pred_bits, retr["strength"], tau_dict, label_space=LABEL_SPACE_V2
    )

    # Mutex check (decoder also enforces, but record reasons)
    _, mutex_reasons = enforce_no_imaging_mutex(gated, label_space=LABEL_SPACE_V2)
    forced_abstain = "mutex_violation" if "mutex_violation" in mutex_reasons else ""

    # Decode
    result = decode_policyA_from_bits(gated, abstain_reason=forced_abstain,
                                       mode="multi_label")
    return _build_row(row, mode, retr, result=result,
                      error_log=error_log, raw_text=raw,
                      gate_reasons=gate_reasons,
                      meta_unknown=meta.get("unknown_keys") or [])


def _build_row(row: dict, mode: str, retr: dict, result: dict,
               error_log: str = "", raw_text: str = "",
               gate_reasons: list = None, meta_unknown: list = None) -> dict:
    bits = result.get("pred_bits") or {}
    out = {
        "stay_id":            row.get("stay_id"),
        "mode":               mode,
        "strength":           float(retr.get("strength", 0.0)),
        "sigma":              float(retr.get("sigma", 0.0)) if "sigma" in retr else None,
        "alpha":              float(retr.get("alpha", 1.0)) if "alpha" in retr else None,
        "evidence_chars":     len(retr.get("evidence_text") or ""),
        "decision":           result.get("decision", "abstain"),
        "pred_label_set":     ";".join(result.get("pred_label_set") or []),
        "pred_no_imaging":    int(result.get("pred_no_imaging", 0)),
        "is_commit_imaging":  int(bool(result.get("is_commit_imaging", False))),
        "is_commit_no_imaging": int(bool(result.get("is_commit_no_imaging", False))),
        "is_abstain":         int(bool(result.get("is_abstain", False))),
        "is_multi_pred":      int(bool(result.get("is_multi_pred", False))),
        "abstain_reason":     result.get("abstain_reason", "") or "",
        "gate_reasons":       ";".join(gate_reasons or []),
        "unknown_keys":       ";".join(meta_unknown or []),
        "error_log":          error_log,
        "raw_text":           (raw_text or "")[:2000],
    }
    for L in LABEL_SPACE_V2:
        out[f"pred_y__{L}"] = int(bits.get(L, 0))
    out[f"pred_y__{NO_IMAGING_LABEL}"] = int(bits.get(NO_IMAGING_LABEL, 0))
    return out


def _append_err(prev: str, msg: str) -> str:
    if not prev: return msg or ""
    if not msg: return prev
    return f"{prev} | {msg}"


# ====================================================================
# Main
# ====================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="CR-MSR unified runner (M1.5 baseline, V2 multi-label)")
    ap.add_argument("--mode", required=True,
                    choices=["vanilla", "single_pubmed", "single_medwiki",
                             "single_guideline", "msrc"])
    ap.add_argument("--in_csv", default=paths.COHORT_V2_CSV,
                    help=f"input cohort CSV (default: {paths.COHORT_V2_CSV})")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--llm_model_path",
                    default=os.environ.get("LOCAL_MODEL_PATH", DEFAULT_LLM_MODEL_PATH))
    ap.add_argument("--embed_model_path",
                    default=os.environ.get("EMBED_MODEL_PATH", DEFAULT_EMBED_MODEL_PATH))
    ap.add_argument("--milvus_host", default=MILVUS_HOST_DEFAULT)
    ap.add_argument("--milvus_port", type=int, default=MILVUS_PORT_DEFAULT)
    ap.add_argument("--gpu_id", default=os.environ.get("LOCAL_GPU_ID", "6"))
    ap.add_argument("--tau_none", type=float, default=DEFAULT_TAU_NONE)
    ap.add_argument("--tau_ct",   type=float, default=DEFAULT_TAU_CT)
    ap.add_argument("--tau_cxr",  type=float, default=DEFAULT_TAU_CXR)
    ap.add_argument("--tau_us",   type=float, default=DEFAULT_TAU_US)
    ap.add_argument("--tau_xmo",  type=float, default=DEFAULT_TAU_XMO,
                    help="τ for XR_*, MRI_*, OTHER_OTHER")
    ap.add_argument("--n",        type=int, default=-1,
                    help="number of stays to process (-1=all)")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--gold_set_col", default="gold_set_full",
                    help="gold set column for evaluation context (full or early)")
    ap.add_argument("--restrict_to", default="",
                    help="optional column filter, e.g. 'has_edstay=1'")
    return ap.parse_args()


def build_tau_dict_from_args(args) -> dict:
    d = {}
    for L in LABEL_SPACE_V2:
        if L.startswith("CT_"):
            d[L] = args.tau_ct
        elif L == "CXR_CHEST":
            d[L] = args.tau_cxr
        elif L.startswith("US_"):
            d[L] = args.tau_us
        else:
            d[L] = args.tau_xmo
    return d


def set_seeds(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main():
    args = parse_args()
    set_seeds(args.seed)

    print(f"[ENV={paths.ENV}] mode={args.mode}")
    print(f"  LLM:    {args.llm_model_path}")
    print(f"  Embed:  {args.embed_model_path}")
    print(f"  Milvus: {args.milvus_host}:{args.milvus_port}")
    print(f"  τ:      none={args.tau_none}, CT={args.tau_ct}, CXR={args.tau_cxr}, "
          f"US={args.tau_us}, XMO={args.tau_xmo}")
    print(f"  seed={args.seed}  n={args.n}  temperature={args.temperature}")

    # Load input
    print(f"\n[1] Loading input: {args.in_csv}", flush=True)
    df = read_csv_robust(args.in_csv)
    print(f"    {len(df):,} rows")

    if args.restrict_to:
        col, _, val = args.restrict_to.partition("=")
        col = col.strip(); val = val.strip()
        try:
            v = int(val)
        except Exception:
            v = val
        if col in df.columns:
            df = df[df[col] == v].copy()
            print(f"    after restrict {col}={v}: {len(df):,}")

    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
        print(f"    sampled {len(df):,}")

    # Load LLM
    print(f"\n[2] Loading LLM on GPU {args.gpu_id}...", flush=True)
    load_llm(args.llm_model_path, gpu_id=args.gpu_id)

    # Load embedder + Milvus if not vanilla
    if args.mode != "vanilla":
        print(f"\n[3] Loading embedder + Milvus...", flush=True)
        load_embedder(args.embed_model_path)
        connect_milvus(args.milvus_host, args.milvus_port)

    tau_dict = build_tau_dict_from_args(args)

    # Process
    print(f"\n[4] Processing {len(df):,} stays...", flush=True)
    t0 = time.time()
    out_rows = []
    n_total = len(df)
    for i, row in enumerate(df.to_dict("records")):
        try:
            out = process_stay(row, args.mode, args, tau_dict)
        except Exception as e:
            print(f"[ERROR] stay_id={row.get('stay_id')}: {e}")
            traceback.print_exc()
            out = {"stay_id": row.get("stay_id"), "mode": args.mode,
                   "decision": "abstain", "abstain_reason": "runner_exception",
                   "error_log": str(e), "is_abstain": 1,
                   "is_commit_imaging": 0, "is_commit_no_imaging": 0,
                   "is_multi_pred": 0, "pred_label_set": "",
                   "pred_no_imaging": 0, "strength": 0.0, "evidence_chars": 0,
                   "raw_text": "", "gate_reasons": "", "unknown_keys": ""}
            for L in LABEL_SPACE_V2 + [NO_IMAGING_LABEL]:
                out[f"pred_y__{L}"] = 0
        out_rows.append(out)
        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n_total - (i + 1)) / max(rate, 1e-6)
            print(f"    {i+1}/{n_total} ({elapsed:.0f}s, {1/rate:.2f}s/stay, "
                  f"ETA {eta:.0f}s)", flush=True)

    # Save
    print(f"\n[5] Saving output: {args.out_csv}", flush=True)
    out_df = pd.DataFrame(out_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"    Saved {len(out_df):,} rows ({time.time()-t0:.0f}s total)")

    # QC
    print(f"\n=== QC ===")
    print(f"decision distribution:")
    print(out_df["decision"].value_counts().to_string())
    if (out_df["decision"] == "abstain").any():
        print(f"\nabstain reasons (top 10):")
        print(out_df.loc[out_df["decision"] == "abstain", "abstain_reason"]
              .value_counts().head(10).to_string())
    n_commit_img = int((out_df["decision"] == "commit_imaging").sum())
    n_multi = int(((out_df["decision"] == "commit_imaging") &
                   (out_df["is_multi_pred"] == 1)).sum())
    if n_commit_img:
        print(f"\nmulti-label commits: {n_multi:,} / {n_commit_img:,} "
              f"({100*n_multi/n_commit_img:.1f}%)")
    print(f"\nlabel sums (top-15):")
    label_sums = {L: int(out_df[f"pred_y__{L}"].sum()) for L in LABEL_SPACE_V2}
    label_sums[NO_IMAGING_LABEL] = int(out_df[f"pred_y__{NO_IMAGING_LABEL}"].sum())
    for L, c in sorted(label_sums.items(), key=lambda x: -x[1])[:15]:
        print(f"  {c:>6}  {L}")


if __name__ == "__main__":
    main()
