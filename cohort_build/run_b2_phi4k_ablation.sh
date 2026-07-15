#!/usr/bin/env bash
# =====================================================================
# CARE-Probe B-2 — 4k Phi-3-mini ablation (128k LongRoPE confound 분리)
# 목적: phi3mini-128k의 last-token CT-miss 붕괴(≈0.53)가
#        (a) 128k LongRoPE 장문맥 적응 때문인가, (b) 단순 grounding/scale 때문인가.
# 방법: 비-128k '4k' Phi-3-mini를 동일 파이프라인에 통과시켜 last/prompt ct_miss AUC 비교.
#   판정: 4k도 last가 chance(≈0.5)면  => 128k 아님 = grounding/scale 요인 (한계로 정직 기술)
#         4k에서 last가 회복(>0.6)되면 => LongRoPE(128k)가 신호 붕괴에 기여 (confound 입증)
# 산출: runs_b2/{probe,analyze,crc 로그} + results_b2.csv + 비교표 출력.
# 실행: bash run_b2_phi4k_ablation.sh        (서버, GPU 필요)
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ===================== CONFIG (서버 환경에 맞게 확인) =====================
EMBED=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
MROOT=/path/to/models
# 4k Phi-3-mini 가중치 경로. [직접 확인 필요] 서버에 없으면 먼저 받아야 함:
#   huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir "$MROOT/Microsoft/Phi-3-mini-4k-Instruct"
PHI4K="$MROOT/Microsoft/Phi-3-mini-4k-Instruct"
GPU=6
N=2000                                       # MAIN과 동일 (phi3mini-128k와 직접 비교 위해)
SUB_SEEDS="42 43 44"
SPLIT_SEEDS="42 43 44 45 46 47 48 49 50 51"
LAYER_IDX=3                                   # MAIN과 동일 a-priori 고정 레이어
LOSS=ct_miss; ALPHA=0.10; DELTA=0.10
OUTDIR=runs_b2
RESULTS=$OUTDIR/results_b2.csv
MODEL_KEY=phi3mini-4k
# ========================================================================

export CRMSR_ENV=server
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # Phi SentencePiece 토크나이저 대비
mkdir -p "$OUTDIR/logs"

if [[ ! -d "$PHI4K" ]]; then
  echo "[FATAL] 4k Phi-3-mini 가중치 없음: $PHI4K"
  echo "  먼저: huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir \"$PHI4K\""
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

  # 2) probe AUC grid (last vs prompt; 핵심 산출)
  python e0_probe_analyze.py --npz "$npz" --window full \
    > "$OUTDIR/logs/analyze_${MODEL_KEY}_seed${sub}.log" 2>&1 || true

  # 3) CRC random split × seeds -> results_b2.csv (gap%/joint_risk 비교용)
  for sp in $SPLIT_SEEDS; do
    python e0_crc.py --npz "$npz" --window full --loss "$LOSS" --probe_target loss \
      --layer_idx "$LAYER_IDX" --alpha "$ALPHA" --delta "$DELTA" --seed "$sp" \
      --tag "${MODEL_KEY}|sub${sub}|main" --results_csv "$RESULTS" \
      >> "$OUTDIR/logs/crc_${MODEL_KEY}_seed${sub}.log" 2>&1
  done
done

echo ""
echo "================ B-2 비교: 4k vs 128k Phi-3-mini (seed42 ct_miss) ================"
for tag in "4k:$OUTDIR/logs/analyze_${MODEL_KEY}_seed42.log" \
           "128k:runs_main/logs/analyze_phi3mini-128k_seed42.log"; do
  name="${tag%%:*}"; f="${tag#*:}"
  echo "---- Phi-3-mini-$name ----"
  if [[ -f "$f" ]]; then
    awk '/\[ct_miss\]/{c=1} c&&/prompt:|last/{print} /best ct_miss/{print; c=0}' "$f"
  else
    echo "   (로그 없음: $f)"
  fi
done
echo ""
echo "[판정 가이드] 4k의 last 행 최대 AUC가:"
echo "  ≈0.50  -> 128k 아님: grounding/scale 요인 (Limitation 정직 기술 유지)"
echo "  >0.60  -> LongRoPE(128k)가 last-token 붕괴에 기여 (confound 입증, §5.3/Limitation 보강)"
echo "[B-2] DONE $(date '+%F %T')"
