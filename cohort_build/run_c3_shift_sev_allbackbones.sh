#!/usr/bin/env bash
# =====================================================================
# C3 — extend E-shift (temporal) + E-sev (Mondrian) to ALL 8 backbones.
# The MAIN run applied these only to the primary (qwen14b); mock-review
# R2-W4/Q6 + R1-Q2 + meta-C6 ask for multi-backbone evidence + per-group n.
# This is a DRIVER over the FROZEN e0_crc.py on the EXISTING probe npz —
# no new hidden-state extraction, CPU-only (the npz already hold states).
# Place in cohort_build_v2/ ; run: bash run_c3_shift_sev_allbackbones.sh
# Aggregate E-shift: python e0_aggregate.py --results_csv runs_c3/results_c3_shift.csv
# E-sev per-group guarantees are parsed from runs_c3/logs/esev_*.log
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

N=2000
SUB_SEEDS="42 43 44"
SPLIT_SEEDS="42 43 44 45 46 47 48 49 50 51"
LAYER_IDX=3
LOSS=ct_miss
ALPHA=0.10
DELTA=0.10
ALPHA_HIGH=0.05
ALPHA_LOW=0.15
SHIFT_CSV=stay_anchor_year.csv          # same sidecar the primary used
SHIFT_TEST_GROUP="2017 - 2019"          # clean max-shift test bin (quote: has spaces)
NPZDIR=runs_main
OUTDIR=runs_c3
RESULTS_SHIFT=$OUTDIR/results_c3_shift.csv
export CRMSR_ENV=server
mkdir -p "$OUTDIR/logs"

[[ -f "$SHIFT_CSV" ]] || { echo "[fatal] $SHIFT_CSV missing (run add_anchor_year.py first)"; exit 1; }

ORDER="qwen14b llama31-8b gemma2-9b mistral7b-v03 med42-8b openbio-8b phi3mini-128k mediphi"

echo "[C3] E-shift + E-sev over ALL backbones -> $OUTDIR ($(date '+%F %T'))"
for m in $ORDER; do
  for sub in $SUB_SEEDS; do
    npz="$NPZDIR/probe_${m}_n${N}_seed${sub}.npz"
    [[ -f "$npz" ]] || { echo "[skip] $m sub$sub : no npz ($npz)"; continue; }

    # ---- E-shift: temporal split, clean max-shift test group, across split-seeds ----
    for sp in $SPLIT_SEEDS; do
      python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
        --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
        --split temporal --shift_csv "$SHIFT_CSV" --shift_test_group "$SHIFT_TEST_GROUP" \
        --tag "${m}|sub${sub}|shift" --results_csv "$RESULTS_SHIFT" \
        >> "$OUTDIR/logs/crc_${m}_shift_seed${sub}.log" 2>&1
    done

    # ---- E-sev: Mondrian severity-adaptive (per-group guarantee logged; incl. group n) ----
    python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
      --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sub" \
      --mondrian --alpha_high "$ALPHA_HIGH" --alpha_low "$ALPHA_LOW" \
      > "$OUTDIR/logs/esev_${m}_seed${sub}.log" 2>&1 || true
    echo "[done] $m sub$sub ($(date '+%T'))"
  done
done
echo "[C3] DONE $(date '+%F %T')."
echo "  E-shift: python e0_aggregate.py --results_csv $RESULTS_SHIFT"
echo "  E-sev  : grep -H 'n_te=\\|joint_risk\\|committed_harm' $OUTDIR/logs/esev_*.log"
