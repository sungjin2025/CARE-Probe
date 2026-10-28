#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""deleak_surface_runner.py -- de-leak deployed surface signals (GPU, orig. S11/E0X).

Same runtime monkeypatch as deleak_probe_runner.py, applied to
e0x_deployed_surface_runner.py: the four post-hoc prompt fields

    REMOVE = icd10_primary, assessment, plan, diagnosis

are stripped from imaging_runner_v2.CASE_FIELDS before the E0X runner is
imported. E0X builds every prompt (greedy, 5 self-consistency samples, 3
per-source generations) through ir._make_case_dict, which re-reads
CASE_FIELDS on each call, so the patch reaches all of them.

Frozen-code principle: imaging_runner_v2 and e0x_deployed_surface_runner are
NOT edited (diff 0). All E0X args pass straight through.

HOW TO RUN (server, cohort_build_v2/; normally via run_deleak_extra_gpu.sh JOB=surface):
    CRMSR_ENV=server MILVUS_TOKEN=... python deleak_surface_runner.py \
      --out_csv runs_e0x/qwen14b-deleak_deployed_surface.csv \
      --llm_model_path /mnt/nvme01/huggingface/models/Qwen/Qwen2.5-14B-Instruct \
      --gpu_id 9 --n 2000 --seed 42 --evidence_mode concat \
      --main_npz runs_main/probe_qwen14b-deleak_n2000_seed42.npz --resume
"""
from __future__ import annotations

import os
import sys

# -- CUDA pin: must run BEFORE importing imaging_runner_v2 (torch) --
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

# fields to strip from the prompt (post-hoc / discharge-time) -- identical to deleak_probe_runner
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

import e0x_deployed_surface_runner as e0x  # noqa: E402  (import AFTER patch; shares the same ir module)

# the E0X module must see the patched list (it holds a reference to the same ir module)
assert e0x.ir is ir and e0x.ir.CASE_FIELDS == ir.CASE_FIELDS, "E0X does not share patched ir"

if __name__ == "__main__":
    e0x.main()
