#!/usr/bin/env bash
# =====================================================================
# run_deleak_extra_downstream.sh  (CPU) -- analyses for the GPU outputs of
# run_deleak_extra_gpu.sh. FROZEN e0_crc.py and analyzers are reused; scripts
# with hard-coded paths are copied with a path-only sed (copies end in _deleak.py).
# Place in cohort_build_v2/ and run after the GPU jobs:
#   bash run_deleak_extra_downstream.sh > runs_deleak/logs/extra_downstream.out 2>&1
# Each section skips itself if its inputs are missing (rerunnable).
#
# Outputs (send back):
#   runs_deleak/results_deleak_full.csv        (1) full-cohort main/shift
#   runs_deleak/logs/esev_qwen14b-deleak_full.log
#   runs_deleak/b4_3seed-deleak.csv            (2) no-RAG vs RAG, 3 gen x 10 split
#   runs_deleak/logs/noRAG-deleak_auc.log      (2) probe AUC no-RAG vs RAG
#   runs_e0x/e0x_summary-deleak.csv            (3) surface signals
#   runs_e0x/confound_summary-deleak.csv       (3) pred_card vs gold_card
#   runs_deleak/c2_local_headtohead_results-deleak.csv, reconcile_card_auc-deleak.csv
#   runs_deleak/logs/extra_downstream.out      (this console, incl. summaries)
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

SPLIT_SEEDS="42 43 44 45 46 47 48 49 50 51"
LAYER_IDX=3; LOSS=ct_miss; ALPHA=0.10; DELTA=0.10
SHIFT_CSV=stay_anchor_year.csv
SHIFT_TEST_GROUP="2017 - 2019"
OUT=runs_deleak
export CRMSR_ENV=server
mkdir -p "$OUT/logs"
BBS="qwen14b mediphi gemma2-9b llama31-8b med42-8b mistral7b-v03 openbio-8b phi3mini-128k"

# ---------- (0) anchor: rerun one existing de-leak MAIN cell, must match ----------
echo "=== (0) anchor: qwen14b-deleak sub42 split42 vs results_deleak_main.csv ==="
A=$OUT/_anchor_extra.csv; rm -f "$A"
python e0_crc.py --npz runs_main/probe_qwen14b-deleak_n2000_seed42.npz --window full --loss "$LOSS" \
  --probe_target loss --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed 42 \
  --tag "qwen14b-deleak|sub42|main" --results_csv "$A" > /dev/null 2>&1
python - "$A" "$OUT/results_deleak_main.csv" <<'PY'
import csv, sys
new = list(csv.reader(open(sys.argv[1])))[1]
ref = [r for r in csv.reader(open(sys.argv[2])) if r[0] == new[0] and r[5] == new[5]]
ok = bool(ref) and ref[0][10:] == new[10:]
print("[anchor]", "OK (pipeline reproduces)" if ok else f"MISMATCH new={new[10:]} ref={ref[0][10:] if ref else None}")
PY

# ---------- (1) full-cohort scale-up ----------
NPZ_FULL=runs_b3/probe_qwen14b-deleak_full_seed42.npz
RES_FULL=$OUT/results_deleak_full.csv
if [[ -f "$NPZ_FULL" ]]; then
  echo "=== (1) full-cohort ($(date '+%T')) ==="
  rm -f "$RES_FULL"
  python e0_probe_analyze.py --npz "$NPZ_FULL" --window full \
    > "$OUT/logs/analyze_qwen14b-deleak_full.log" 2>&1 || true
  for sp in $SPLIT_SEEDS; do
    python e0_crc.py --npz "$NPZ_FULL" --window full --loss "$LOSS" --probe_target loss \
      --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
      --tag "qwen14b-deleak|full|main" --results_csv "$RES_FULL" \
      >> "$OUT/logs/crc_qwen14b-deleak_full.log" 2>&1
    python e0_crc.py --npz "$NPZ_FULL" --window full --loss "$LOSS" --probe_target loss \
      --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
      --split temporal --shift_csv "$SHIFT_CSV" --shift_test_group "$SHIFT_TEST_GROUP" \
      --tag "qwen14b-deleak|full|shift" --results_csv "$RES_FULL" \
      >> "$OUT/logs/crc_qwen14b-deleak_full_shift.log" 2>&1
  done
  python e0_crc.py --npz "$NPZ_FULL" --window full --loss "$LOSS" --probe_target loss \
    --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed 42 \
    --mondrian --alpha_high 0.05 --alpha_low 0.15 \
    > "$OUT/logs/esev_qwen14b-deleak_full.log" 2>&1 || true
  echo "--- full-n vs n=2000 (main split; mean over split seeds) ---"
  awk -F',' '
    $1=="qwen14b-deleak|full|main"            {fg+=$17; fj+=$12; fc+=$11; fn++; fu=$9}
    $1 ~ /^qwen14b-deleak\|sub42\|main$/      {sg+=$17; sj+=$12; sc+=$11; sn++}
    $1=="qwen14b|full|main"                   {wg+=$17; wj+=$12; wc+=$11; wn++}
    END{
      if(fn) printf "  de-leak full-n (n_usable=%s): gap%%=%.1f joint=%.4f commit=%.3f (%d runs)\n",fu,100*fg/fn,fj/fn,fc/fn,fn;
      if(sn) printf "  de-leak n=2000              : gap%%=%.1f joint=%.4f commit=%.3f (%d runs)\n",100*sg/sn,sj/sn,sc/sn,sn;
      if(wn) printf "  with-field full-n (ref)     : gap%%=%.1f joint=%.4f commit=%.3f (%d runs)\n",100*wg/wn,wj/wn,wc/wn,wn;
    }' "$RES_FULL" "$OUT/results_deleak_main.csv" runs_b3/results_b3.csv
  awk -F',' '$1=="qwen14b-deleak|full|shift"{n++; ok+=($12<=$7)} END{if(n) printf "  full-n temporal: %d/%d within alpha\n",ok,n}' "$RES_FULL"
else
  echo "[skip] (1) $NPZ_FULL missing"
fi

# ---------- (2) closed-book ablation (mirrors run_b4_3seed.sh CRC loop) ----------
RES_B4=$OUT/b4_3seed-deleak.csv
P=runs_e0/noRAG-deleak_2k_s
Q=runs_main/probe_qwen14b-deleak_n2000_seed
if ls ${P}*.npz > /dev/null 2>&1; then
  echo "=== (2) no-RAG ($(date '+%T')) ==="
  rm -f "$RES_B4"
  for g in 42 43 44; do
    [[ -f ${P}$g.npz && -f ${Q}$g.npz ]] || { echo "[skip] gen seed $g (npz missing)"; continue; }
    for S in $SPLIT_SEEDS; do
      python e0_crc.py --npz ${P}$g.npz --layer_idx 3 --probe_target loss --mondrian --seed $S \
        --tag noRAG-deleak_g$g --results_csv "$RES_B4" > /dev/null 2>&1
      python e0_crc.py --npz ${Q}$g.npz --layer_idx 3 --probe_target loss --mondrian --seed $S \
        --tag RAG-deleak_g$g --results_csv "$RES_B4" > /dev/null 2>&1
    done
  done
  awk -F',' 'NR>1{k=($1 ~ /^noRAG/)?"noRAG":"RAG"; g[k]+=$17; j[k]+=$12; c[k]+=$11; n[k]++}
    END{for(k in n) printf "  %-5s: gap%%=%.1f joint=%.4f commit=%.3f (%d runs)\n",k,100*g[k]/n[k],j[k]/n[k],c[k]/n[k],n[k]}' "$RES_B4"
  # probe AUC: path-only copy of _recompute_noRAG_auc.py
  sed -e 's/f"noRAG_2k_s{sd}.npz"/f"noRAG-deleak_2k_s{sd}.npz"/' \
      -e 's/R.run_model("qwen14b")/R.run_model("qwen14b-deleak")/' \
      _recompute_noRAG_auc.py > _recompute_noRAG_auc_deleak.py
  if grep -q 'noRAG-deleak_2k_s' _recompute_noRAG_auc_deleak.py && grep -q 'run_model("qwen14b-deleak")' _recompute_noRAG_auc_deleak.py; then
    python _recompute_noRAG_auc_deleak.py 2>&1 | tee "$OUT/logs/noRAG-deleak_auc.log"
  else
    echo "[FAIL] sed patch of _recompute_noRAG_auc.py did not apply"
  fi
else
  echo "[skip] (2) no ${P}*.npz"
fi

# ---------- (3) surface signals ----------
SUM=runs_e0x/e0x_summary-deleak.csv
if ls runs_e0x/*-deleak_deployed_surface.csv > /dev/null 2>&1; then
  echo "=== (3) surface ($(date '+%T')) ==="
  for m in $BBS; do
    f=runs_e0x/${m}-deleak_deployed_surface.csv
    [[ -f "$f" ]] || { echo "[skip] $m (no $f)"; continue; }
    python e0x_analyze.py --pred_csv "$f" --tag "${m}-deleak" --summary_csv "$SUM" \
      > "$OUT/logs/e0x_analyze_${m}-deleak.log" 2>&1 || echo "[warn] e0x_analyze $m"
  done
  python e0x_analyze.py --latex --pred_csv unused --tag unused --summary_csv "$SUM" 2>&1 | tail -40
  # confound (pred_card vs gold_card; gold-card reference AUC): path-only copy
  sed -e 's/f"{bb}_deployed_surface.csv"/f"{bb}-deleak_deployed_surface.csv"/' \
      -e 's/"confound_summary.csv"/"confound_summary-deleak.csv"/g' \
      runs_e0x/confound_analysis.py > runs_e0x/confound_analysis_deleak.py
  if grep -q 'deleak_deployed_surface' runs_e0x/confound_analysis_deleak.py; then
    python runs_e0x/confound_analysis_deleak.py 2>&1 | tail -20 \
      || echo "[warn] confound failed (needs all 8 surface csv)"
  else
    echo "[FAIL] sed patch of confound_analysis.py did not apply"
  fi
else
  echo "[skip] (3) no runs_e0x/*-deleak_deployed_surface.csv"
fi

# ---------- (3b) cardinality under CRC (CPU, from existing de-leak npz) ----------
echo "=== (3b) cardinality ($(date '+%T')) ==="
python _deleak_card_analyses.py 2>&1 | tail -30 || echo "[warn] card analyses failed"

echo "[deleak-extra-cpu] DONE $(date '+%F %T')"
