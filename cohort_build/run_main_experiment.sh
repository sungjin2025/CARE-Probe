#!/usr/bin/env bash
# =====================================================================
# CARE-Probe MAIN experiment — reproducible batch over backbones × seeds.
# Run ONCE after the preliminary (single-seed) runs.
#   - extracts probe hidden states (resume-safe) at SUB_SEEDS (subsample seeds)
#   - runs CRC across SPLIT_SEEDS (calib/test split) -> appends to results CSV  (multi-seed CI)
#   - PRIMARY model additionally gets E-shift(temporal) + E-sev(severity)
#   - LAYER fixed a-priori (--layer_idx), NOT tuned on test  (reviewer-proof)
# Aggregate: python e0_aggregate.py --results_csv runs_main/results_main.csv
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ===================== CONFIG (edit me) =====================
EMBED=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
MROOT=/path/to/models
GPU=6
N=2000                                          # -1 = full-n (for headline)
SUB_SEEDS="42 43 44"                             # subsample seeds -> separate extraction (subsample variation)
SPLIT_SEEDS="42 43 44 45 46 47 48 49 50 51"      # calib/test split seeds -> CI (CPU, cheap)
LAYER_IDX=3                                      # a-priori fixed layer position 0..4 (=depth 0.5..1.0); not tuned on test
LOSS=ct_miss
ALPHA=0.10
DELTA=0.10
SHIFT_CSV=stay_anchor_year.csv
OUTDIR=runs_main
RESULTS=$OUTDIR/results_main.csv
PRIMARY=qwen14b                                 # model that additionally gets E-shift+E-sev

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
ORDER="qwen14b llama31-8b gemma2-9b mistral7b-v03 med42-8b openbio-8b phi3mini-128k mediphi"
# ============================================================

export CRMSR_ENV=server
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # guard for SentencePiece tokenizer (Mistral/Phi)
mkdir -p "$OUTDIR/logs"
echo "[main] N=$N sub=($SUB_SEEDS) split=($SPLIT_SEEDS) layer_idx=$LAYER_IDX loss=$LOSS -> $RESULTS"
echo "[main] started $(date '+%F %T')"

for m in $ORDER; do
  path="${MODELS[$m]:-}"
  if [[ -z "$path" || ! -d "$path" ]]; then echo "[skip] $m : path missing ($path)"; continue; fi
  for sub in $SUB_SEEDS; do
    pref="$OUTDIR/probe_${m}_n${N}_seed${sub}"
    npz="${pref}.npz"
    # 1) extract (resume-safe; skip if done)
    if [[ -f "$npz" && -z "${FORCE:-}" ]]; then
      echo "[extract] $m sub=$sub : exists, skip"
    else
      echo "[extract] $m sub=$sub -> $npz  ($(date '+%T'))"
      python e0_probe_runner.py --out_prefix "$pref" \
        --llm_model_path "$path" --embed_model_path "$EMBED" \
        --gpu_id "$GPU" --n "$N" --seed "$sub" --ckpt_every 500 --resume \
        > "$OUTDIR/logs/extract_${m}_seed${sub}.log" 2>&1
    fi
    [[ -f "$npz" ]] || { echo "[fail] $m sub=$sub : no npz (see log)"; continue; }

    # 2) probe AUC (logged once per extraction)
    python e0_probe_analyze.py --npz "$npz" --window full \
      > "$OUTDIR/logs/analyze_${m}_seed${sub}.log" 2>&1 || true

    # 3) CRC random split × SPLIT_SEEDS -> results CSV (E-guar + E-win)
    for sp in $SPLIT_SEEDS; do
      python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
        --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
        --tag "${m}|sub${sub}|main" --results_csv "$RESULTS" \
        >> "$OUTDIR/logs/crc_${m}_seed${sub}.log" 2>&1
    done

    # 4) PRIMARY only: E-shift(temporal) × SPLIT_SEEDS + E-sev(severity, logged)
    if [[ "$m" == "$PRIMARY" && -f "$SHIFT_CSV" ]]; then
      for sp in $SPLIT_SEEDS; do
        python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
          --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
          --split temporal --shift_csv "$SHIFT_CSV" \
          --tag "${m}|sub${sub}|shift" --results_csv "$RESULTS" \
          >> "$OUTDIR/logs/crc_${m}_shift_seed${sub}.log" 2>&1
      done
      # E-sev: per-group goes to the log, not results_csv (per-group guarantee is parsed from the log)
      python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
        --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed 42 \
        --mondrian --alpha_high 0.05 --alpha_low 0.15 \
        > "$OUTDIR/logs/esev_${m}_seed${sub}.log" 2>&1 || true
    fi
  done
done
echo "[main] DONE $(date '+%F %T'). aggregate -> python e0_aggregate.py --results_csv $RESULTS"
