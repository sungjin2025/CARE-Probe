#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""deleak_probe_runner.py -- full de-leak hidden-state regeneration (GPU).

Extends noicd_probe_runner.py: instead of removing only `icd10_primary`, it
removes ALL post-hoc (non-presentation-time) prompt fields so every signal is
compared on presentation-time information alone:

    REMOVE = icd10_primary, assessment, plan, diagnosis

Rationale: a prompt-field audit (Supp. S28) found the discharge-time ICD-10 code
inadvertently included; `assessment`, `plan`, and `diagnosis` are likewise
discharge-type (post-hoc) fields. Because these enter the LLM prompt, removing
them changes the generation, the retrieval query, and the HIDDEN STATES -> the
probe .npz must be regenerated (GPU). This is the expensive Phase-2 job.

Frozen-code principle (same as noicd_probe_runner): imaging_runner_v2 and
e0_probe_runner are NOT edited; fields are removed by runtime monkeypatch only
(diff 0, auditable). All MAIN-protocol args pass straight through to e0_probe_runner.

=====================================================================
HOW TO RUN (server, per backbone; mirror your MAIN / noicd invocation):
  Place THIS FILE inside cohort_build_v2/ then, for each of the 8 backbones:

    CRMSR_ENV=server MILVUS_TOKEN=... python deleak_probe_runner.py \
      --out_prefix runs_main/probe_qwen14b-deleak_n2000_seed42 \
      --llm_model_path /mnt/nvme01/huggingface/models/Qwen/Qwen2.5-14B-Instruct \
      --embed_model_path /mnt/nvme01/huggingface/models/Alibaba-NLP/gte-large-en-v1.5 \
      --gpu_id 9 --n 2000 --seed 42 --ckpt_every 500 --resume

  * IMPORTANT: use out_prefix suffix "-deleak" (i.e. probe_{bb}-deleak_n2000_seed42)
    so the analysis step picks it up via PROBE_TAG=-deleak.
  * gemma2-9b: if the loader hits fp16 activation overflow / torch.compile recompile
    (silent collapse), run with bf16 + TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1,
    exactly as for the P(True) / MAIN gemma2 reruns.
  * GPU id: set to the current assignment (e.g. 9). ~a few hours per backbone.

THEN (CPU) run the analysis on the de-leaked npz:
    PROBE_TAG=-deleak python work_b_signal_crc.py
  -> work_b_signal_crc_results-deleak.csv  (directly comparable to the WITH-icd10 run;
     input-only is unchanged by de-leak, so the comparison isolates the hidden-state
     effect). Send back BOTH csv files.
=====================================================================
"""
from __future__ import annotations

import os
import sys

# -- CUDA pin: must run BEFORE importing imaging_runner_v2 (same as noicd runner) --
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

import imaging_runner_v2 as ir  # noqa: E402  (after CUDA pin)

# fields to strip from the prompt (post-hoc / discharge-time)
REMOVE = ["icd10_primary", "assessment", "plan", "diagnosis"]

_BEFORE = list(ir.CASE_FIELDS)
present = [f for f in REMOVE if f in _BEFORE]
missing = [f for f in REMOVE if f not in _BEFORE]
ir.CASE_FIELDS = [f for f in ir.CASE_FIELDS if f not in REMOVE]

# safety: none of the target fields remain
for f in REMOVE:
    assert f not in ir.CASE_FIELDS, f"patch failed: {f} still present"
assert len(ir.CASE_FIELDS) == len(_BEFORE) - len(present), "unexpected field removal count"

if missing:
    print(f"[deleak][WARN] target field(s) not in CASE_FIELDS (name mismatch?): {missing}",
          flush=True)
print(f"[deleak] removed {len(present)} field(s): {', '.join(present)}", flush=True)
print(f"[deleak] fields now ({len(ir.CASE_FIELDS)}): {', '.join(ir.CASE_FIELDS)}", flush=True)

import e0_probe_runner as pr  # noqa: E402  (import AFTER patch; shares the same ir module)

if __name__ == "__main__":
    pr.main()
