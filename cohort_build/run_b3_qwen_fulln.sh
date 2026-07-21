#!/usr/bin/env bash
# =====================================================================
# CARE-Probe B-3 — Qwen2.5-14B full-cohort headline run
# Purpose: the draft refers to PRIMARY as "full cohort" -> current MAIN is unified at N=2000.
#        One-off run to decide whether to report headline numbers on full-n (entire cohort).
# Note: full-n has no subsample variation, so a single extraction seed suffices (the rest is split-seed CI).
#        N=-1 => e0_probe_runner uses the entire cohort (no subsampling).
# Outputs: runs_b3/{probe,analyze,crc,shift,esev} + results_b3.csv + n2000 comparison table.
# Run: bash run_b3_qwen_fulln.sh        (server, GPU required; 14B full-n extraction = ~4x MAIN's time)
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ===================== CONFIG =====================
EMBED=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
MROOT=/path/to/models
QWEN="$MROOT/Qwen/Qwen2.5-14B-Instruct"
GPU=6
N=-1                                          # -1 = full cohort
SUB_SEED=42                                    # for full-n, a single extraction suffices
SPLIT_SEEDS="42 43 44 45 46 47 48 49 50 51"    # calib/test split for CI
LAYER_IDX=3
LOSS=ct_miss; ALPHA=0.10; DELTA=0.10
SHIFT_CSV=stay_anchor_year.csv
OUTDIR=runs_b3
RESULTS=$OUTDIR/results_b3.csv
MODEL_KEY=qwen14b
# =================================================

export CRMSR_ENV=server
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mkdir -p "$OUTDIR/logs"

if [[ ! -d "$QWEN" ]]; then echo "[FATAL] Qwen weights missing: $QWEN"; exit 1; fi
echo "[B-3] N=$N(full) sub=$SUB_SEED split=($SPLIT_SEEDS) layer_idx=$LAYER_IDX -> $RESULTS"
echo "[B-3] started $(date '+%F %T')"

pref="$OUTDIR/probe_${MODEL_KEY}_full_seed${SUB_SEED}"
npz="${pref}.npz"

# 1) extract (full cohort; resume-safe — if interrupted, resumes on rerun)
if [[ -f "$npz" && -z "${FORCE:-}" ]]; then
  echo "[extract] exists, skip ($npz)"
else
  echo "[extract] full-n -> $npz  ($(date '+%T'))  # 14B full-n: takes several hours"
  python e0_probe_runner.py --out_prefix "$pref" \
    --llm_model_path "$QWEN" --embed_model_path "$EMBED" \
    --gpu_id "$GPU" --n "$N" --seed "$SUB_SEED" --ckpt_every 500 --resume \
    > "$OUTDIR/logs/extract_${MODEL_KEY}_full.log" 2>&1
fi
[[ -f "$npz" ]] || { echo "[fail] no npz (see log)"; exit 1; }

# 2) probe AUC grid
python e0_probe_analyze.py --npz "$npz" --window full \
  > "$OUTDIR/logs/analyze_${MODEL_KEY}_full.log" 2>&1 || true

# 3) CRC random split × seeds -> results_b3.csv (E-guar + E-win)
for sp in $SPLIT_SEEDS; do
  python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
    --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
    --tag "${MODEL_KEY}|full|main" --results_csv "$RESULTS" \
    >> "$OUTDIR/logs/crc_${MODEL_KEY}_full.log" 2>&1
done

# 4) E-shift(temporal) × seeds + E-sev(severity) — PRIMARY additional guarantee (re-verified on full-n too)
if [[ -f "$SHIFT_CSV" ]]; then
  for sp in $SPLIT_SEEDS; do
    python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
      --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
      --split temporal --shift_csv "$SHIFT_CSV" \
      --tag "${MODEL_KEY}|full|shift" --results_csv "$RESULTS" \
      >> "$OUTDIR/logs/crc_${MODEL_KEY}_full_shift.log" 2>&1
  done
  python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
    --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed 42 \
    --mondrian --alpha_high 0.05 --alpha_low 0.15 \
    > "$OUTDIR/logs/esev_${MODEL_KEY}_full.log" 2>&1 || true
fi

# 5) full-n vs n=2000 headline comparison (gap%/joint_risk mean)
echo ""
echo "================ B-3 comparison: Qwen full-n vs n=2000 (main split) ================"
awk -F',' '
  $1=="qwen14b|full|main"   {fg+=$17; fj+=$12; fc+=$11; fn++}
  $1 ~ /^qwen14b\|sub..\|main$/ {sg+=$17; sj+=$12; sc+=$11; sn++}
  END{
    if(fn) printf "  full-n  : gap%%=%.1f  joint_risk=%.4f  commit=%.3f  (n=%d runs)\n",100*fg/fn,fj/fn,fc/fn,fn;
    if(sn) printf "  n=2000  : gap%%=%.1f  joint_risk=%.4f  commit=%.3f  (n=%d runs)\n",100*sg/sn,sj/sn,sc/sn,sn;
  }' "$RESULTS" runs_main/results_main.csv
echo "[verdict] if full-n gap%/joint_risk differs significantly from n=2000, switch the headline to full-n,"
echo "       if similar, keep 'unified N=2000' (current draft state)."
echo "[B-3] DONE $(date '+%F %T')"
