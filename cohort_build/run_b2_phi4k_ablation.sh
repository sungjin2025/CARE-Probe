#!/usr/bin/env bash
# =====================================================================
# CARE-Probe B-2 — 4k Phi-3-mini ablation (isolating the 128k LongRoPE confound)
# Purpose: is phi3mini-128k's last-token CT-miss collapse (≈0.53) due to
#        (a) 128k LongRoPE long-context adaptation, or (b) plain grounding/scale?
# Method: run a non-128k '4k' Phi-3-mini through the same pipeline and compare last/prompt ct_miss AUC.
#   Verdict: if 4k's last is also chance (≈0.5) => not 128k = grounding/scale factor (state honestly as a limitation)
#            if 4k's last recovers (>0.6)       => LongRoPE(128k) contributes to the signal collapse (confound confirmed)
# Outputs: runs_b2/{probe,analyze,crc logs} + results_b2.csv + comparison table printed.
# Run: bash run_b2_phi4k_ablation.sh        (server, GPU required)
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ===================== CONFIG (verify for your server environment) =====================
EMBED=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
MROOT=/path/to/models
# 4k Phi-3-mini weights path. [verify manually] If not on the server, download first:
#   huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir "$MROOT/Microsoft/Phi-3-mini-4k-Instruct"
PHI4K="$MROOT/Microsoft/Phi-3-mini-4k-Instruct"
GPU=6
N=2000                                       # same as MAIN (for direct comparison with phi3mini-128k)
SUB_SEEDS="42 43 44"
SPLIT_SEEDS="42 43 44 45 46 47 48 49 50 51"
LAYER_IDX=3                                   # same a-priori fixed layer as MAIN
LOSS=ct_miss; ALPHA=0.10; DELTA=0.10
OUTDIR=runs_b2
RESULTS=$OUTDIR/results_b2.csv
MODEL_KEY=phi3mini-4k
# ========================================================================

export CRMSR_ENV=server
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # guard for Phi SentencePiece tokenizer
mkdir -p "$OUTDIR/logs"

if [[ ! -d "$PHI4K" ]]; then
  echo "[FATAL] 4k Phi-3-mini weights missing: $PHI4K"
  echo "  first: huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir \"$PHI4K\""
  exit 1
fi
echo "[B-2] N=$N sub=($SUB_SEEDS) split=($SPLIT_SEEDS) layer_idx=$LAYER_IDX -> $RESULTS"
echo "[B-2] started $(date '+%F %T')"

for sub in $SUB_SEEDS; do
  pref="$OUTDIR/probe_${MODEL_KEY}_n${N}_seed${sub}"
  npz="${pref}.npz"
  # 1) extract hidden states (resume-safe)
  if [[ -f "$npz" && -z "${FORCE:-}" ]]; then
    echo "[extract] sub=$sub : exists, skip"
  else
    echo "[extract] sub=$sub -> $npz  ($(date '+%T'))"
    python e0_probe_runner.py --out_prefix "$pref" \
      --llm_model_path "$PHI4K" --embed_model_path "$EMBED" \
      --gpu_id "$GPU" --n "$N" --seed "$sub" --ckpt_every 500 --resume \
      > "$OUTDIR/logs/extract_${MODEL_KEY}_seed${sub}.log" 2>&1
  fi
  [[ -f "$npz" ]] || { echo "[fail] sub=$sub : no npz (see log)"; continue; }

  # 2) probe AUC grid (last vs prompt; key output)
  python e0_probe_analyze.py --npz "$npz" --window full \
    > "$OUTDIR/logs/analyze_${MODEL_KEY}_seed${sub}.log" 2>&1 || true

  # 3) CRC random split × seeds -> results_b2.csv (for gap%/joint_risk comparison)
  for sp in $SPLIT_SEEDS; do
    python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
      --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
      --tag "${MODEL_KEY}|sub${sub}|main" --results_csv "$RESULTS" \
      >> "$OUTDIR/logs/crc_${MODEL_KEY}_seed${sub}.log" 2>&1
  done
done

echo ""
echo "================ B-2 comparison: 4k vs 128k Phi-3-mini (seed42 ct_miss) ================"
for tag in "4k:$OUTDIR/logs/analyze_${MODEL_KEY}_seed42.log" \
           "128k:runs_main/logs/analyze_phi3mini-128k_seed42.log"; do
  name="${tag%%:*}"; f="${tag#*:}"
  echo "---- Phi-3-mini-$name ----"
  if [[ -f "$f" ]]; then
    awk '/\[ct_miss\]/{c=1} c&&/prompt:|last/{print} /best ct_miss/{print; c=0}' "$f"
  else
    echo "   (log missing: $f)"
  fi
done
echo ""
echo "[verdict guide] max AUC of 4k's last row:"
echo "  ≈0.50  -> not 128k: grounding/scale factor (keep honest Limitation statement)"
echo "  >0.60  -> LongRoPE(128k) contributes to last-token collapse (confound confirmed, reinforces §5.3/Limitation)"
echo "[B-2] DONE $(date '+%F %T')"
