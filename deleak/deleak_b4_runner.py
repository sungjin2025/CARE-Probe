#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""deleak_b4_runner.py -- de-leak P(True)/semantic-entropy regeneration (GPU).

Same idea as deleak_probe_runner.py, but wraps b4_semantic_ptrue_runner.py (the
P(True) + semantic-entropy generator) instead of e0_probe_runner. Removes the
post-hoc prompt fields so the behavioral uncertainty signals are computed on
presentation-time information alone:

    REMOVE = icd10_primary, assessment, plan, diagnosis

b4_semantic_ptrue_runner builds the prompt via imaging_runner_v2 (ir._make_case_dict
-> ir.build_case_text_ed), so the runtime CASE_FIELDS monkeypatch flows through to
the samples and the P(True) self-eval -> the signals are de-leaked. Frozen-code
principle: b4 runner / imaging_runner_v2 are NOT edited.

=====================================================================
HOW TO RUN (server, per backbone; mirror the original b4 runs). Use orchestration
run_deleak_b4.sh, or directly:
  CRMSR_ENV=server MILVUS_TOKEN=... USE_TF=0 USE_TORCH=1 \
  python deleak_b4_runner.py \
    --out_csv runs_e0/b4_semantic_ptrue_qwen14b-deleak_n2000_seed42.csv \
    --llm_model_path   /mnt/nvme01/huggingface/models/Qwen/Qwen2.5-14B-Instruct \
    --embed_model_path /mnt/nvme01/huggingface/models/Alibaba-NLP/gte-large-en-v1.5 \
    --gpu_id 9 --n 2000 --seed 42 --n_samples 5

  * out_csv MUST contain "{bb}-deleak" so work_b_signal_crc.py (PROBE_TAG=-deleak)
    picks it up as the de-leak P(True)/SemEnt source.
  * N=5 samples + P(True) self-eval -> ~6-11 h/backbone on Qwen-14B (like b4).
  * gemma2-9b: bf16 + TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 if it collapses.
=====================================================================
"""
from __future__ import annotations

import os
import sys

# -- CUDA pin BEFORE importing imaging_runner_v2 (mirror deleak_probe_runner) --
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _gpu = "9"
    if "--gpu_id" in sys.argv:
        try:
            _gpu = sys.argv[sys.argv.index("--gpu_id") + 1]
        except (IndexError, ValueError):
            pass
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import imaging_runner_v2 as ir  # noqa: E402

REMOVE = ["icd10_primary", "assessment", "plan", "diagnosis"]

_BEFORE = list(ir.CASE_FIELDS)
present = [f for f in REMOVE if f in _BEFORE]
missing = [f for f in REMOVE if f not in _BEFORE]
ir.CASE_FIELDS = [f for f in ir.CASE_FIELDS if f not in REMOVE]
for f in REMOVE:
    assert f not in ir.CASE_FIELDS, f"patch failed: {f} still present"
assert len(ir.CASE_FIELDS) == len(_BEFORE) - len(present), "unexpected field removal count"
if missing:
    print(f"[deleak-b4][WARN] target field(s) not in CASE_FIELDS: {missing}", flush=True)
print(f"[deleak-b4] removed {len(present)} field(s): {', '.join(present)}", flush=True)
print(f"[deleak-b4] fields now ({len(ir.CASE_FIELDS)}): {', '.join(ir.CASE_FIELDS)}", flush=True)

import b4_semantic_ptrue_runner as b4  # noqa: E402  (after patch; shares the same ir module)

if __name__ == "__main__":
    b4.main()
