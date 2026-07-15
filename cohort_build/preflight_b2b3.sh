#!/usr/bin/env bash
# =====================================================================
# B-2/B-3 사전점검 — 수 시간 GPU 런 전에 필수 전제조건을 한 번에 확인.
# (중간에 깨지는 것 방지: 가중치·Milvus·GPU·디스크·deps)
# 실행: bash preflight_b2b3.sh
# =====================================================================
cd "$(dirname "$0")"
export CRMSR_ENV=server
MROOT=/path/to/models
EMBED="$MROOT/Alibaba-NLP/gte-large-en-v1.5"
PHI4K="$MROOT/Microsoft/Phi-3-mini-4k-Instruct"
QWEN="$MROOT/Qwen/Qwen2.5-14B-Instruct"
GPU=6
FAIL_B2=0; FAIL_B3=0
ok(){ echo "  [OK]  $1"; }
no(){ echo "  [!!]  $1"; }

echo "==================== 공통 ===================="
[ -f cohort_v2.csv ]        && ok "cohort_v2.csv" || { no "cohort_v2.csv 없음"; FAIL_B2=1; FAIL_B3=1; }
[ -f stay_anchor_year.csv ] && ok "stay_anchor_year.csv (E-shift/E-sev)" || { no "stay_anchor_year.csv 없음"; FAIL_B3=1; }
[ -d "$EMBED" ] && ok "embed model: $EMBED" || { no "embed model 없음: $EMBED"; FAIL_B2=1; FAIL_B3=1; }
python -c "import sklearn,numpy,scipy,torch,transformers" 2>/dev/null \
  && ok "py deps (sklearn/torch/transformers)" || { no "py deps 부족"; FAIL_B2=1; FAIL_B3=1; }

echo "-- Milvus (검색 필수; 3개 컬렉션) --"
if python _check_milvus.py --milvus_host localhost --milvus_port 19530 2>&1 | grep -qE "\[FAIL\]"; then
  no "Milvus 점검 실패 → 검색 단계가 멈춤. Milvus 기동 후 재시도."
  FAIL_B2=1; FAIL_B3=1
else
  ok "Milvus 연결/컬렉션 OK"
fi

echo "-- GPU $GPU 여유 --"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$GPU" 2>/dev/null \
    && ok "GPU $GPU 조회됨(위 사용량 확인)" || no "GPU $GPU 조회 실패(번호 확인)"
else
  no "nvidia-smi 없음"
fi

echo "-- 디스크 여유(npz 저장) --"
df -h . | awk 'NR==1||NR==2'

echo ""
echo "==================== B-2 (4k Phi ablation) ===================="
[ -d "$PHI4K" ] && ok "4k Phi 가중치: $PHI4K" \
  || { no "4k 가중치 없음 → huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir \"$PHI4K\""; FAIL_B2=1; }
[ -f runs_main/logs/analyze_phi3mini-128k_seed42.log ] && ok "128k 비교기준 로그 존재" || no "128k 비교 로그 없음(비교표 일부 생략됨)"

echo ""
echo "==================== B-3 (Qwen full-n) ===================="
[ -d "$QWEN" ] && ok "Qwen 가중치: $QWEN" || { no "Qwen 가중치 없음: $QWEN"; FAIL_B3=1; }
[ -f runs_main/results_main.csv ] && ok "n=2000 대비기준(results_main.csv) 존재" || no "results_main.csv 없음(대비표 생략)"

echo ""
echo "==================== 판정 ===================="
[ "$FAIL_B2" -eq 0 ] && echo "  B-2: ✅ GO   (bash run_b2_phi4k_ablation.sh)" || echo "  B-2: ❌ 위 [!!] 해결 후 실행"
[ "$FAIL_B3" -eq 0 ] && echo "  B-3: ✅ GO   (bash run_b3_qwen_fulln.sh)"   || echo "  B-3: ❌ 위 [!!] 해결 후 실행"
