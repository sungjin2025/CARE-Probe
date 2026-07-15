#!/usr/bin/env bash
# =====================================================================
# CR-MSR M1.5 baseline batch runner
#   5 modes × 3 seeds = 15 runs, then eval_v2 for each.
#
# Usage:
#   # 1) edit paths.py server block (see SERVER_SETUP.md)
#   # 2) export CRMSR_ENV=server
#   # 3) (recommended) run sanity check first:
#   #      python _check_milvus.py --embed_model_path $EMBED_MODEL_PATH
#   # 4) launch:
#   #      bash run_all_modes.sh
#
# Environment variables you may override:
#   N_STAYS         number of stays per run        (default 1000)
#   GPU_ID          CUDA device id                 (default 0)
#   LLM_PATH        path to Qwen2.5-14B-Instruct
#   EMBED_PATH      path to gte-large-en-v1.5
#   MILVUS_HOST     Milvus host                    (default localhost)
#   MILVUS_PORT     Milvus port                    (default 19530)
#   OUT_DIR         output dir                     (default $WORK_DIR/runs_m15)
#   COHORT_CSV      cohort csv                     (default $WORK_DIR/cohort_v2.csv)
#   SEEDS           space-separated seeds          (default "42 43 44")
#   MODES           space-separated modes          (default 5 modes)
#   SKIP_CHECK      "1" to skip Milvus sanity check (default 0)
#   RESUME          "1" to skip runs whose out_csv already exists (default 1)
#
# Logs:
#   $OUT_DIR/logs/{mode}_seed{N}.log         runner stdout/stderr
#   $OUT_DIR/logs/eval_{mode}_seed{N}.log    eval stdout/stderr
#   $OUT_DIR/BATCH_SUMMARY.txt               final summary
# =====================================================================

set -uo pipefail   # -e off: we want partial failures to continue

# ---------- defaults (override via env) ----------
: "${CRMSR_ENV:=server}"
: "${N_STAYS:=1000}"
: "${GPU_ID:=0}"
: "${MILVUS_HOST:=localhost}"
: "${MILVUS_PORT:=19530}"
: "${SEEDS:=42 43 44}"
: "${MODES:=vanilla single_pubmed single_medwiki single_guideline msrc}"
: "${SKIP_CHECK:=0}"
: "${RESUME:=1}"

export CRMSR_ENV

# ---------- resolve paths from paths.py ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-read WORK_DIR from paths.py (single source of truth)
WORK_DIR_FROM_PATHS=$(python -c "import paths; print(paths.WORK_DIR)" 2>/dev/null)
if [[ -z "$WORK_DIR_FROM_PATHS" ]]; then
  echo "[FATAL] failed to import paths.py — check CRMSR_ENV=$CRMSR_ENV and server-block TODOs"
  exit 1
fi

: "${COHORT_CSV:=$WORK_DIR_FROM_PATHS/cohort_v2.csv}"
: "${OUT_DIR:=$WORK_DIR_FROM_PATHS/runs_m15}"

if [[ -z "${LLM_PATH:-}" ]]; then
  echo "[FATAL] LLM_PATH not set. Example:"
  echo "  export LLM_PATH=/path/to/models/Qwen2.5-14B-Instruct"
  exit 1
fi
if [[ -z "${EMBED_PATH:-}" ]]; then
  echo "[FATAL] EMBED_PATH not set. Example:"
  echo "  export EMBED_PATH=/path/to/models/Alibaba-NLP/gte-large-en-v1.5"
  exit 1
fi
if [[ ! -f "$COHORT_CSV" ]]; then
  echo "[FATAL] cohort csv not found: $COHORT_CSV"
  exit 1
fi

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

SUMMARY="$OUT_DIR/BATCH_SUMMARY.txt"
: > "$SUMMARY"

echo "=========================================================" | tee -a "$SUMMARY"
echo "CR-MSR M1.5 batch runner"                                  | tee -a "$SUMMARY"
echo "  CRMSR_ENV       = $CRMSR_ENV"                            | tee -a "$SUMMARY"
echo "  WORK_DIR        = $WORK_DIR_FROM_PATHS"                  | tee -a "$SUMMARY"
echo "  OUT_DIR         = $OUT_DIR"                              | tee -a "$SUMMARY"
echo "  COHORT_CSV      = $COHORT_CSV"                           | tee -a "$SUMMARY"
echo "  LLM_PATH        = $LLM_PATH"                             | tee -a "$SUMMARY"
echo "  EMBED_PATH      = $EMBED_PATH"                           | tee -a "$SUMMARY"
echo "  MILVUS          = $MILVUS_HOST:$MILVUS_PORT"              | tee -a "$SUMMARY"
echo "  GPU_ID          = $GPU_ID"                               | tee -a "$SUMMARY"
echo "  N_STAYS         = $N_STAYS"                              | tee -a "$SUMMARY"
echo "  MODES           = $MODES"                                | tee -a "$SUMMARY"
echo "  SEEDS           = $SEEDS"                                | tee -a "$SUMMARY"
echo "  RESUME          = $RESUME"                               | tee -a "$SUMMARY"
echo "  Started         = $(date '+%F %T')"                       | tee -a "$SUMMARY"
echo "=========================================================" | tee -a "$SUMMARY"

# ---------- optional Milvus sanity check ----------
if [[ "$SKIP_CHECK" != "1" ]]; then
  echo ""
  echo "[STEP] Milvus sanity check ..."
  if python _check_milvus.py \
        --embed_model_path "$EMBED_PATH" \
        --milvus_host "$MILVUS_HOST" \
        --milvus_port "$MILVUS_PORT" 2>&1 | tee "$LOG_DIR/_check_milvus.log"; then
    echo "[OK] Milvus check passed"   | tee -a "$SUMMARY"
  else
    echo "[FAIL] Milvus check failed (see $LOG_DIR/_check_milvus.log)" | tee -a "$SUMMARY"
    echo "Aborting. Set SKIP_CHECK=1 to bypass."
    exit 1
  fi
fi

# ---------- batch loop ----------
N_TOTAL=0
N_OK=0
N_FAIL=0
N_SKIP=0
START_EPOCH=$(date +%s)

for mode in $MODES; do
  for seed in $SEEDS; do
    N_TOTAL=$((N_TOTAL + 1))
    TAG="${mode}_qwen14b_seed${seed}_n${N_STAYS}"
    OUT_CSV="$OUT_DIR/pred_${TAG}.csv"
    RUN_LOG="$LOG_DIR/${TAG}.log"
    EVAL_LOG="$LOG_DIR/eval_${TAG}.log"

    echo "" | tee -a "$SUMMARY"
    echo "--- [$N_TOTAL] mode=$mode seed=$seed ---" | tee -a "$SUMMARY"

    # resume: skip if pred already exists
    if [[ "$RESUME" == "1" && -f "$OUT_CSV" ]]; then
      ROWS=$(wc -l < "$OUT_CSV" 2>/dev/null || echo 0)
      echo "  [SKIP] $OUT_CSV exists ($ROWS lines). Set RESUME=0 to rerun." | tee -a "$SUMMARY"
      N_SKIP=$((N_SKIP + 1))
    else
      T0=$(date +%s)
      python imaging_runner_v2.py \
        --mode "$mode" \
        --seed "$seed" \
        --in_csv  "$COHORT_CSV" \
        --out_csv "$OUT_CSV" \
        --llm_model_path   "$LLM_PATH" \
        --embed_model_path "$EMBED_PATH" \
        --milvus_host "$MILVUS_HOST" \
        --milvus_port "$MILVUS_PORT" \
        --gpu_id "$GPU_ID" \
        --n "$N_STAYS" \
        > "$RUN_LOG" 2>&1
      RC=$?
      T1=$(date +%s)
      ELAPSED=$((T1 - T0))

      if [[ $RC -eq 0 && -f "$OUT_CSV" ]]; then
        echo "  [OK]   runner finished in ${ELAPSED}s → $OUT_CSV" | tee -a "$SUMMARY"
        N_OK=$((N_OK + 1))
      else
        echo "  [FAIL] runner rc=$RC (see $RUN_LOG)" | tee -a "$SUMMARY"
        N_FAIL=$((N_FAIL + 1))
        continue
      fi
    fi

    # ---------- eval ----------
    METRICS_JSON="$OUT_DIR/${TAG}_metrics.json"
    if [[ "$RESUME" == "1" && -f "$METRICS_JSON" ]]; then
      echo "  [SKIP] eval: $METRICS_JSON exists" | tee -a "$SUMMARY"
    else
      python eval_v2.py \
        --pred_csv   "$OUT_CSV" \
        --cohort_csv "$COHORT_CSV" \
        --window full \
        --out_dir    "$OUT_DIR" \
        --tag        "$TAG" \
        > "$EVAL_LOG" 2>&1
      RC=$?
      if [[ $RC -eq 0 ]]; then
        echo "  [OK]   eval → ${TAG}_metrics.json, ${TAG}_report.txt" | tee -a "$SUMMARY"
      else
        echo "  [FAIL] eval rc=$RC (see $EVAL_LOG)" | tee -a "$SUMMARY"
        N_FAIL=$((N_FAIL + 1))
      fi
    fi
  done
done

END_EPOCH=$(date +%s)
TOTAL_ELAPSED=$((END_EPOCH - START_EPOCH))

echo ""                                                                         | tee -a "$SUMMARY"
echo "========================================================="               | tee -a "$SUMMARY"
echo "Batch complete: $(date '+%F %T')  (elapsed=${TOTAL_ELAPSED}s)"            | tee -a "$SUMMARY"
echo "  total = $N_TOTAL"                                                       | tee -a "$SUMMARY"
echo "  ok    = $N_OK"                                                          | tee -a "$SUMMARY"
echo "  skip  = $N_SKIP  (already-existing artifacts, RESUME=1)"                | tee -a "$SUMMARY"
echo "  fail  = $N_FAIL"                                                        | tee -a "$SUMMARY"
echo ""                                                                         | tee -a "$SUMMARY"
echo "Artifacts in: $OUT_DIR"                                                   | tee -a "$SUMMARY"
echo "  pred_*.csv"                                                             | tee -a "$SUMMARY"
echo "  *_metrics.json"                                                         | tee -a "$SUMMARY"
echo "  *_report.txt"                                                           | tee -a "$SUMMARY"
echo "Logs in:      $LOG_DIR"                                                   | tee -a "$SUMMARY"
echo "Summary file: $SUMMARY"                                                   | tee -a "$SUMMARY"
echo "========================================================="               | tee -a "$SUMMARY"

if [[ $N_FAIL -gt 0 ]]; then
  exit 1
fi
exit 0
