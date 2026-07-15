#!/usr/bin/env bash
# run_e0x.sh — E0X: deployed-protocol surface-signal screening, 3 backbones.
# Design: proposals/99.단계별_진행자료/E0X_deployed_surface_design.md (approved 2026-07-14)
# GPU 1 (user directive 2026-07-14). GPU 6 = CR-MSR — DO NOT USE.
#
# Server usage:
#   cd ~/Research/PJT1/cohort_build_v2
#   export MILVUS_TOKEN=...            # required since 06-03 (auth enabled)
#   nohup bash run_e0x.sh > runs_e0x/run_e0x.out 2>&1 &
#
# NOTE model paths: verify with `ls /path/to/models/` before
# launch — MediPhi/Gemma dir names below are [직접 확인 필요] placeholders.
set -u

export CRMSR_ENV=server
GPU=1
N=2000
SEED=42
MODELROOT=/path/to/models
OUTDIR=runs_e0x
mkdir -p "$OUTDIR" "$OUTDIR/logs"

if [ -z "${MILVUS_TOKEN:-}" ]; then
  echo "[FATAL] MILVUS_TOKEN not set (Milvus auth enabled since 06-03)"; exit 1
fi

# tag | model path | MAIN probe npz (alignment + prediction-agreement check)
# paths verified against server 2026-07-14 (vendor dirs are Capitalized; gemma
# npz uses a hyphen: gemma2-9b, and the model dir is gemma-2-9b-Instruct).
BACKBONES=(
  "qwen14b|$MODELROOT/Qwen/Qwen2.5-14B-Instruct|runs_main/probe_qwen14b_n2000_seed42.npz"
  "mediphi|$MODELROOT/Microsoft/MediPhi-Instruct|runs_main/probe_mediphi_n2000_seed42.npz"
  "gemma2-9b|$MODELROOT/Google/gemma-2-9b-Instruct|runs_main/probe_gemma2-9b_n2000_seed42.npz"
)

for spec in "${BACKBONES[@]}"; do
  IFS='|' read -r TAG MODEL NPZ <<< "$spec"
  OUT="$OUTDIR/${TAG}_deployed_surface.csv"
  LOG="$OUTDIR/logs/${TAG}.log"
  echo "=== [E0X] $TAG ($(date '+%F %T')) ==="

  if [ ! -d "$MODEL" ]; then
    echo "[SKIP] model path not found: $MODEL — fix run_e0x.sh"; continue
  fi
  NPZ_ARG=()
  if [ -f "$NPZ" ]; then
    NPZ_ARG=(--main_npz "$NPZ")
  else
    echo "[WARN] MAIN npz not found ($NPZ) — running WITHOUT alignment check"
  fi

  python e0x_deployed_surface_runner.py \
    --out_csv "$OUT" \
    --llm_model_path "$MODEL" \
    --gpu_id "$GPU" --n "$N" --seed "$SEED" \
    --evidence_mode concat \
    "${NPZ_ARG[@]}" \
    --resume \
    2>&1 | tee "$LOG"

  # low MAIN-agreement means the MAIN generation was not concat-protocol:
  # re-check with --evidence_mode fused on a small -n before burning full runs.
  python e0x_analyze.py --pred_csv "$OUT" --tag "$TAG" \
    --summary_csv "$OUTDIR/e0x_summary.csv" 2>&1 | tee -a "$LOG"
done

echo "=== [E0X] all done ($(date '+%F %T')) ==="
python e0x_analyze.py --latex --pred_csv unused --tag unused \
  --summary_csv "$OUTDIR/e0x_summary.csv" || true
touch "$OUTDIR/DONE_E0X"
