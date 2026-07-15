#!/usr/bin/env bash
# =====================================================================
# preflight_check.sh — C2/C3 실행 전 전제조건 점검 (읽기 전용, 아무것도 실행/생성 안 함)
# 사용: cohort_build_v2/ 에 두고  ->  bash preflight_check.sh
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

PASS=0; FAIL=0; WARN=0
ok(){   printf "  [OK]   %s\n" "$1"; PASS=$((PASS+1)); }
bad(){  printf "  [FAIL] %s\n" "$1"; FAIL=$((FAIL+1)); }
warn(){ printf "  [WARN] %s\n" "$1"; WARN=$((WARN+1)); }

echo "== 0. 현재 위치 =="
echo "  cwd = $(pwd)"

echo "== 1. 업로드한 스크립트 3개 =="
for f in c2_baseline_headtohead.py run_c2_baselines.sh run_c3_shift_sev_allbackbones.sh; do
  [[ -f "$f" ]] && ok "$f" || bad "$f  (업로드 누락)"
done

echo "== 2. import/호출되는 기존 런너·모듈 =="
for f in e0_crc.py imaging_runner_v2.py paths.py e0_conflict_runner.py e0_uncertainty_runner.py e0_aggregate.py; do
  [[ -f "$f" ]] && ok "$f" || bad "$f  (없음 -> 해당 실험 실패)"
done

echo "== 3. Python import & 심볼 확인 (실행 아님, import만) =="
python - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import e0_crc
    for s in ("select_lambda","risk_coverage","aurc_random"):
        print(("[OK]   e0_crc."+s) if hasattr(e0_crc,s) else ("[FAIL] e0_crc."+s+" 없음"))
except Exception as e:
    print("[FAIL] import e0_crc ->", repr(e))
try:
    import imaging_runner_v2 as ir
    print("[OK]   imaging_runner_v2.read_csv_robust" if hasattr(ir,"read_csv_robust") else "[FAIL] read_csv_robust 없음")
except Exception as e:
    print("[FAIL] import imaging_runner_v2 ->", repr(e))
try:
    import paths
    p=getattr(paths,"COHORT_V2_CSV",None)
    print("[OK]   paths.COHORT_V2_CSV =", p) if p else print("[FAIL] paths.COHORT_V2_CSV 없음")
except Exception as e:
    print("[FAIL] import paths ->", repr(e))
for mod in ("numpy","pandas","sklearn"):
    try:
        __import__(mod); print("[OK]   pydep", mod)
    except Exception as e:
        print("[FAIL] pydep", mod, "->", repr(e))
PY

echo "== 4. 데이터: cohort CSV =="
CV=$(python - <<'PY' 2>/dev/null
try:
    import paths; print(getattr(paths,"COHORT_V2_CSV",""))
except Exception: print("")
PY
)
[[ -n "$CV" && -f "$CV" ]] && ok "paths.COHORT_V2_CSV 파일 존재: $CV" || bad "paths.COHORT_V2_CSV 파일 없음: '${CV:-<unset>}'"
[[ -f cohort_v2.csv ]] && ok "cohort_v2.csv (run_c2 IN_CSV, cwd)" || warn "cohort_v2.csv 없음 (run_c2 Step1 --in_csv 기본값). paths와 다른 파일이면 준비 필요"

echo "== 5. 데이터: probe npz (runs_main) =="
NPZ_OK=0; NPZ_MISS=0
BBS="qwen14b llama31-8b gemma2-9b mistral7b-v03 med42-8b openbio-8b phi3mini-128k mediphi"
for m in $BBS; do
  for sub in 42 43 44; do
    f="runs_main/probe_${m}_n2000_seed${sub}.npz"
    if [[ -f "$f" ]]; then NPZ_OK=$((NPZ_OK+1)); else NPZ_MISS=$((NPZ_MISS+1)); echo "     - 없음: $f"; fi
  done
done
if   [[ $NPZ_OK -eq 24 ]]; then ok "probe npz 24/24 (8백본 x 3서브시드) 모두 존재"
elif [[ $NPZ_OK -gt 0  ]]; then warn "probe npz $NPZ_OK/24 존재 (일부 백본만 가능). 없는 백본은 자동 skip됨"
else bad "probe npz 0/24 — C2/C3 둘 다 아무것도 못 돌림 (MAIN run 산출물 필요)"; fi

echo "== 6. C3 전용: 앵커연도 sidecar =="
[[ -f stay_anchor_year.csv ]] && ok "stay_anchor_year.csv" || { \
  [[ -f add_anchor_year.py ]] && warn "stay_anchor_year.csv 없음 -> 먼저 'python add_anchor_year.py' 실행 필요 (add_anchor_year.py는 있음)" \
                              || bad "stay_anchor_year.csv 없고 add_anchor_year.py도 없음 -> C3 E-shift 즉시 fatal"; }

echo "== 7. C2 Step1(GPU) 모델 경로 (있으면 좋음, 없으면 해당 백본 신호 skip) =="
MROOT=/path/to/models
EMBED=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
[[ -d "$EMBED" ]] && ok "embed: gte-large-en-v1.5" || warn "embed 모델 경로 없음: $EMBED"
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
for m in "${!MODELS[@]}"; do
  [[ -d "${MODELS[$m]}" ]] && ok "model $m" || warn "model $m 경로 없음 (${MODELS[$m]})"
done

echo "== 8. GPU 상태 (run_c2 GPU=6 사용) =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | sed 's/^/  GPU /'
else warn "nvidia-smi 없음 (C2 Step1 GPU 신호수집 불가; C2 partial/C3는 가능)"; fi

echo
echo "==================== 요약 ===================="
echo "  PASS=$PASS  WARN=$WARN  FAIL=$FAIL"
[[ $FAIL -eq 0 ]] && echo "  => FAIL 0. 실행 진행 가능 (WARN은 비용/부분실행 관련이니 내용 확인)." \
                  || echo "  => FAIL 있음. 위 [FAIL] 항목 먼저 해결 후 실행."
echo "=============================================="
