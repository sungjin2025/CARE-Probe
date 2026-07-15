#!/usr/bin/env bash
# =====================================================================
# C2 — head-to-head baseline vs CARE-Probe (mock-review R2-W3/Q5).
# Step 1: collect the SAME surface signals used in Table 1, but at the
#         MAIN n=2000 subsample (so they align stay-for-stay with the
#         frozen probe npz). Reuses the frozen collectors unchanged.
# Step 2: run the head-to-head (CARE-Probe vs Conf/SelfCons/Suff/Conflict/
#         Cardinality/Random) through the SAME CRC/AURC machinery.
#
# NOTE: signal collection is LLM generation (Conf/Conflict = vanilla+P/W/G,
#       SelfCons = 5 samples/case) -> GPU. CARE-Probe + Cardinality + Random
#       need NO regeneration, so c2_baseline_headtohead.py already runs a
#       partial head-to-head even before Step 1 completes.
# Place in cohort_build_v2/ ; run: bash run_c2_baselines.sh
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

EMBED=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
MROOT=/path/to/models
GPU=6
N=2000
SUB_SEEDS="42 43 44"
N_SAMPLES=5            # self-consistency samples/case (matches Table 1 signal)
TEMP=0.7
TOPP=0.9
OUTDIR=runs_e0
IN_CSV=cohort_v2.csv  # conflict_runner --in_csv (same cohort as probe_runner)

export CRMSR_ENV=server
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mkdir -p "$OUTDIR/logs"

declare -A MODELS=(
  [qwen14b]="$MROOT/Qwen/Qwen2.5-14B-Instruct"
  [llama31-8b]="$MROOT/MetaAI/Llama_3.1_8B-Instruct"
  [gemma2-9b]="$MROOT/Google/gemma-2-9b-Instruct"
  [mistral7b-v03]="$MROOT/MistralAI/Mistral-7B-Instruct-v0.3"
  [med42-8b]="$MROOT/m42-health/Llama3-Med42-8B"
  [openbio-8b]="$MROOT/Aaditya/Llama3-OpenBioLLM-8B"
  [phi3mini-128k]="$MROOT/Microsoft/Phi-3-mini-128k-Instruct"
  [mediphi]="$MROOT/Microsoft/MediPhi-Instruct"
)
# To limit cost, start with a subset e.g. ORDER="qwen14b mediphi phi3mini-128k"
ORDER="qwen14b llama31-8b gemma2-9b mistral7b-v03 med42-8b openbio-8b phi3mini-128k mediphi"

for m in $ORDER; do
  path="${MODELS[$m]:-}"
  [[ -z "$path" || ! -d "$path" ]] && { echo "[skip] $m path missing"; continue; }
  for sub in $SUB_SEEDS; do
    preds="$OUTDIR/e0_preds_n2000_${m}_seed${sub}.csv"
    unc="$OUTDIR/e0_unc_n2000_${m}_seed${sub}.csv"
    # (a) conflict runner -> van_conf (logprob), {P,W,G}_strength (sufficiency), per-source bits (conflict)
    if [[ -f "$preds" ]]; then echo "[preds] $m sub$sub exists, skip"; else
      echo "[preds] $m sub$sub -> $preds ($(date '+%T'))"
      python e0_conflict_runner.py --in_csv "$IN_CSV" --out_csv "$preds" \
        --llm_model_path "$path" --embed_model_path "$EMBED" \
        --gpu_id "$GPU" --n "$N" --seed "$sub" \
        > "$OUTDIR/logs/preds_${m}_seed${sub}.log" 2>&1 || echo "[warn] preds $m sub$sub failed"
    fi
    # (b) uncertainty runner -> U_sc (self-consistency)
    if [[ -f "$unc" ]]; then echo "[unc] $m sub$sub exists, skip"; else
      echo "[unc] $m sub$sub -> $unc ($(date '+%T'))"
      python e0_uncertainty_runner.py --out_csv "$unc" \
        --llm_model_path "$path" --embed_model_path "$EMBED" \
        --gpu_id "$GPU" --n "$N" --seed "$sub" \
        --n_samples "$N_SAMPLES" --temperature "$TEMP" --top_p "$TOPP" \
        > "$OUTDIR/logs/unc_${m}_seed${sub}.log" 2>&1 || echo "[warn] unc $m sub$sub failed"
    fi
  done
done

echo "[C2] signals collected. Running head-to-head ..."
python c2_baseline_headtohead.py --npz_dir runs_main --sig_dir runs_e0 \
  --alpha 0.10 --delta 0.10 --out c2_headtohead_results.csv --backbones "$ORDER"
echo "[C2] DONE -> c2_headtohead_results.csv"
