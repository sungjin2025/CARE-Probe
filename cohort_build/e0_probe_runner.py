#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""e0_probe_runner.py — CARE-MSR S3: extract LLM internal states for error-probing.

E0 showed naive signals (logprob, self-consistency, sufficiency, label-conflict)
are all at chance for predicting error on cohort_v2. Literature (CHOKE/Simhi 2025,
Kadavath 2022) says INTERNAL-STATE PROBING is the most robust surviving signal.
This runs the multi-source-conditioned greedy prediction ONCE per stay and saves
the hidden-state vectors (last prompt token + last generated token, selected
layers) plus the prediction bits. e0_probe_analyze.py then trains a probe and
asks: does any layer/position predict error ABOVE chance?

Key: the probe label (error) is on the SAME prediction we extract states from →
no signal/prediction misalignment (the bug that muddied the U_sc test).

Output: <out_prefix>.npz (hidden states + pred bits + stay_ids).

Usage (server):
  CRMSR_ENV=server python e0_probe_runner.py \
    --out_prefix runs_e0/e0_probe_n500_seed42 \
    --llm_model_path   /path/to/models/Qwen/Qwen2.5-14B-Instruct \
    --embed_model_path /path/to/models/Alibaba-NLP/gte-large-en-v1.5 \
    --gpu_id 6 --n 500 --seed 42

  # B-4 closed-book ablation (no retrieval; embedder/Milvus not needed):
  CRMSR_ENV=server python e0_probe_runner.py \
    --out_prefix runs_e0/e0_probe_noRAG_n500_seed42 \
    --llm_model_path   /path/to/models/Qwen/Qwen2.5-14B-Instruct \
    --gpu_id 6 --n 500 --seed 42 --no_retrieval
"""
from __future__ import annotations
import os, sys, time, argparse

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
import e0_conflict_runner as e0r   # reuse LLM loader, cosine search, label space, parsing

LABELS = e0r.LABELS                # 15 imaging labels
SRC_COLL = e0r.SRC_COLL
SRC_ORDER = e0r.SRC_ORDER
DEFAULT_LAYER_FRACS = [0.5, 0.625, 0.75, 0.875, 1.0]


def build_multisource_evidence(qvec, top_k: int) -> str:
    parts = []
    for sk in SRC_ORDER:
        hits = e0r.search_cosine(SRC_COLL[sk], qvec, top_k)
        txt = ir._build_evidence_text(hits)
        if txt.strip():
            parts.append(f"### Source {sk}\n{txt}")
    return "\n\n".join(parts)


def text_to_bits15(raw: str):
    """Pre-gate imaging-label bits (15-dim) from one generation. None if invalid."""
    if ir.is_refusal_text(raw):
        return None
    obj = ir.extract_first_json_object(raw)
    if obj is None:
        return None
    pb, meta = ir.normalize_imaging_json(obj, mode="multi_label")
    if meta.get("is_abstain_json"):
        return None
    fixed, _ = ir.enforce_no_imaging_mutex(pb, label_space=LABELS)
    return np.array([int(fixed.get(L, 0)) for L in LABELS], dtype=np.int8)


def predict_and_probe(prompt: str, layer_fracs, max_new_tokens: int):
    """Greedy generate + one forward pass for hidden states.
    Returns (gen_text, layer_idx_list, prompt_vecs[n_layers,hidden], last_vecs[...])."""
    import torch
    tok, model, device = e0r._LLM["tok"], e0r._LLM["model"], e0r._LLM["device"]
    chat = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt", truncation=True, max_length=8192).to(device)
    plen = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             return_dict_in_generate=True)
    seq = out.sequences[0]
    gen_text = tok.decode(seq[plen:], skip_special_tokens=True).strip()
    with torch.no_grad():
        fo = model(input_ids=seq.unsqueeze(0), output_hidden_states=True)
    hs = fo.hidden_states                      # tuple length L+1
    L = len(hs) - 1
    layers = sorted(set(min(L, max(1, int(round(f * L)))) for f in layer_fracs))
    pvecs, lvecs = [], []
    for li in layers:
        pvecs.append(hs[li][0, plen - 1, :].float().cpu().numpy())
        lvecs.append(hs[li][0, -1, :].float().cpu().numpy())
    del fo, out
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    return gen_text, layers, np.stack(pvecs), np.stack(lvecs)


def main():
    ap = argparse.ArgumentParser(description="S3 internal-state probe extraction")
    ap.add_argument("--in_csv", default=paths.COHORT_V2_CSV)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--llm_model_path", default=ir.DEFAULT_LLM_MODEL_PATH)
    ap.add_argument("--embed_model_path", default=ir.DEFAULT_EMBED_MODEL_PATH)
    ap.add_argument("--milvus_host", default=ir.MILVUS_HOST_DEFAULT)
    ap.add_argument("--milvus_port", type=int, default=ir.MILVUS_PORT_DEFAULT)
    ap.add_argument("--gpu_id", default="6")
    ap.add_argument("--top_k", type=int, default=ir.TOP_K_SINGLE)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--ckpt_every", type=int, default=500, help="save partial .npz every K new stays")
    ap.add_argument("--resume", action="store_true", help="skip stays already in existing out .npz")
    ap.add_argument("--no_retrieval", action="store_true",
                    help="B-4 ablation: closed-book generation — skip multi-source RAG "
                         "(evidence=''); embedder/Milvus are not loaded. Use a distinct "
                         "--out_prefix (e.g. ..._noRAG_...) to avoid overwriting the RAG run.")
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed)

    mode_str = "CLOSED-BOOK (no-retrieval)" if args.no_retrieval else "multi-source RAG"
    print(f"[S3-probe] env={paths.ENV} seed={args.seed} n={args.n} mode={mode_str}")
    df = ir.read_csv_robust(args.in_csv)
    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"  stays={len(df):,} (same sample as e0_conflict_runner)")

    print("[1] load LLM ...", flush=True)
    e0r.load_llm(args.llm_model_path)
    if args.no_retrieval:
        print("[2] no_retrieval → SKIP embedder/Milvus (closed-book ablation)", flush=True)
    else:
        print("[2] load embedder + Milvus ...", flush=True)
        ir.load_embedder(args.embed_model_path)
        ir.connect_milvus(args.milvus_host, args.milvus_port)
        e0r.measure_norm_consts()

    out = args.out_prefix + ".npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # resume: reload prior checkpoint and skip already-done stays
    stay_ids, pred_bits, valid, prompt_all, last_all = [], [], [], [], []
    layer_idx, HID, done = None, None, set()
    if args.resume and Path(out).exists():
        z = np.load(out, allow_pickle=True)
        stay_ids = list(z["stay_ids"]); valid = list(z["valid"]); pred_bits = list(z["pred_bits"])
        prompt_all = list(z["prompt"]); last_all = list(z["last"])
        if z["layer_idx"].size:
            layer_idx = list(z["layer_idx"]); HID = z["prompt"].shape[2]
        done = set(int(s) for s in stay_ids)
        print(f"  [resume] loaded {len(done)} done stays from {out}")

    def _save():
        np.savez_compressed(
            out,
            stay_ids=np.array(stay_ids), valid=np.array(valid, np.int8),
            pred_bits=np.stack(pred_bits) if pred_bits else np.zeros((0, len(LABELS)), np.int8),
            labels=np.array(LABELS),
            layer_idx=np.array(layer_idx if layer_idx is not None else []),
            prompt=np.stack(prompt_all) if prompt_all else np.zeros((0, 0, 0), np.float16),
            last=np.stack(last_all) if last_all else np.zeros((0, 0, 0), np.float16))

    print(f"[3] extract hidden states over {len(df):,} stays "
          f"({len(done)} already done) ...", flush=True)
    t0 = time.time(); n_new = 0
    for row in df.to_dict("records"):
        sid = row.get("stay_id")
        try:
            if int(sid) in done:
                continue
        except (TypeError, ValueError):
            pass
        try:
            case = ir._make_case_dict(row)
            case_text = ir.build_case_text_ed(case)
            if args.no_retrieval:
                ev = ""                                   # B-4: closed-book (no retrieval)
            else:
                qvec = ir.embed_query(case_text)
                ev = build_multisource_evidence(qvec, args.top_k)
            prompt = ir.build_prompt(case_text, ev, mode="multi_label")
            gen, layers, pvecs, lvecs = predict_and_probe(
                prompt, DEFAULT_LAYER_FRACS, args.max_new_tokens)
            if layer_idx is None:
                layer_idx = layers; HID = pvecs.shape[1]
            bits = text_to_bits15(gen)
            stay_ids.append(sid)
            pred_bits.append(bits if bits is not None else np.zeros(len(LABELS), np.int8))
            valid.append(1 if bits is not None else 0)
            prompt_all.append(pvecs.astype(np.float16))
            last_all.append(lvecs.astype(np.float16))
        except Exception as e:
            print(f"  [ERR] stay {sid}: {str(e)[:150]}")
            if HID is None:
                continue   # no shape established yet → skip to avoid misalignment
            stay_ids.append(sid); valid.append(0)
            pred_bits.append(np.zeros(len(LABELS), np.int8))
            prompt_all.append(np.zeros((len(layer_idx), HID), np.float16))
            last_all.append(np.zeros((len(layer_idx), HID), np.float16))
        n_new += 1
        if n_new % 25 == 0:
            el = time.time() - t0; rate = n_new / max(el, 1e-6)
            print(f"    +{n_new} new ({el:.0f}s, {1/rate:.2f}s/stay, "
                  f"total={len(stay_ids)}/{len(df)})", flush=True)
        if n_new % args.ckpt_every == 0:
            _save(); print(f"    [ckpt saved @ total={len(stay_ids)}]", flush=True)

    _save()
    print(f"[4] saved → {out} ({time.time()-t0:.0f}s)  total={len(stay_ids)} "
          f"valid={int(np.sum(valid)) if valid else 0}  layers={layer_idx}  hidden={HID}")


if __name__ == "__main__":
    main()
