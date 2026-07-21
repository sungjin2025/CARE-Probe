#!/usr/bin/env bash
# =====================================================================
# preflight_check.sh — Precondition check before running C2/C3 (read-only, nothing is executed/created)
# Usage: place in cohort_build_v2/  ->  bash preflight_check.sh
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

PASS=0; FAIL=0; WARN=0
ok(){   printf "  [OK]   %s\n" "$1"; PASS=$((PASS+1)); }
bad(){  printf "  [FAIL] %s\n" "$1"; FAIL=$((FAIL+1)); }
warn(){ printf "  [WARN] %s\n" "$1"; WARN=$((WARN+1)); }

echo "== 0. Current location =="
echo "  cwd = $(pwd)"

echo "== 1. The 3 uploaded scripts =="
for f in c2_baseline_headtohead.py run_c2_baselines.sh run_c3_shift_sev_allbackbones.sh; do
  [[ -f "$f" ]] && ok "$f" || bad "$f  (upload missing)"
done

echo "== 2. Existing runners/modules that are imported/called =="
for f in e0_crc.py imaging_runner_v2.py paths.py e0_conflict_runner.py e0_uncertainty_runner.py e0_aggregate.py; do
  [[ -f "$f" ]] && ok "$f" || bad "$f  (missing -> corresponding experiment fails)"
done

echo "== 3. Python import & symbol check (not execution, import only) =="
python - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import e0_crc
    for s in ("select_lambda","risk_coverage","aurc_random"):
        print(("[OK]   e0_crc."+s) if hasattr(e0_crc,s) else ("[FAIL] e0_crc."+s+" missing"))
except Exception as e:
    print("[FAIL] import e0_crc ->", repr(e))
try:
    import imaging_runner_v2 as ir
    print("[OK]   imaging_runner_v2.read_csv_robust" if hasattr(ir,"read_csv_robust") else "[FAIL] read_csv_robust missing")
except Exception as e:
    print("[FAIL] import imaging_runner_v2 ->", repr(e))
try:
    import paths
    p=getattr(paths,"COHORT_V2_CSV",None)
    print("[OK]   paths.COHORT_V2_CSV =", p) if p else print("[FAIL] paths.COHORT_V2_CSV missing")
except Exception as e:
    print("[FAIL] import paths ->", repr(e))
for mod in ("numpy","pandas","sklearn"):
    try:
        __import__(mod); print("[OK]   pydep", mod)
    except Exception as e:
        print("[FAIL] pydep", mod, "->", repr(e))
PY

echo "== 4. Data: cohort CSV =="
CV=$(python - <<'PY' 2>/dev/null
try:
    import paths; print(getattr(paths,"COHORT_V2_CSV",""))
except Exception: print("")
PY
)
[[ -n "$CV" && -f "$CV" ]] && ok "paths.COHORT_V2_CSV file exists: $CV" || bad "paths.COHORT_V2_CSV file missing: '${CV:-<unset>}'"
[[ -f cohort_v2.csv ]] && ok "cohort_v2.csv (run_c2 IN_CSV, cwd)" || warn "cohort_v2.csv missing (run_c2 Step1 --in_csv default). If different from paths, needs preparation"

echo "== 5. Data: probe npz (runs_main) =="
NPZ_OK=0; NPZ_MISS=0
BBS="qwen14b llama31-8b gemma2-9b mistral7b-v03 med42-8b openbio-8b phi3mini-128k mediphi"
for m in $BBS; do
  for sub in 42 43 44; do
    f="runs_main/probe_${m}_n2000_seed${sub}.npz"
    if [[ -f "$f" ]]; then NPZ_OK=$((NPZ_OK+1)); else NPZ_MISS=$((NPZ_MISS+1)); echo "     - missing: $f"; fi
  done
done
if   [[ $NPZ_OK -eq 24 ]]; then ok "probe npz 24/24 (8 backbones x 3 sub-seeds) all present"
elif [[ $NPZ_OK -gt 0  ]]; then warn "probe npz $NPZ_OK/24 present (only some backbones available). Missing backbones are auto-skipped"
else bad "probe npz 0/24 — neither C2 nor C3 can run anything (MAIN run outputs required)"; fi

echo "== 6. C3 only: anchor-year sidecar =="
[[ -f stay_anchor_year.csv ]] && ok "stay_anchor_year.csv" || { \
  [[ -f add_anchor_year.py ]] && warn "stay_anchor_year.csv missing -> run 'python add_anchor_year.py' first (add_anchor_year.py is present)" \
                              || bad "stay_anchor_year.csv missing and add_anchor_year.py also missing -> C3 E-shift immediately fatal"; }

echo "== 7. C2 Step1(GPU) model paths (nice to have; if missing, that backbone's signal is skipped) =="
MROOT=/path/to/models
EMBED=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
[[ -d "$EMBED" ]] && ok "embed: gte-large-en-v1.5" || warn "embed model path missing: $EMBED"
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
  [[ -d "${MODELS[$m]}" ]] && ok "model $m" || warn "model $m path missing (${MODELS[$m]})"
done

echo "== 8. GPU status (run_c2 uses GPU=6) =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | sed 's/^/  GPU /'
else warn "nvidia-smi missing (C2 Step1 GPU signal collection not possible; C2 partial/C3 still possible)"; fi

echo
echo "==================== Summary ===================="
echo "  PASS=$PASS  WARN=$WARN  FAIL=$FAIL"
[[ $FAIL -eq 0 ]] && echo "  => FAIL 0. OK to proceed (WARN relates to cost/partial runs, so review the details)." \
                  || echo "  => FAIL present. Resolve the [FAIL] items above first, then run."
echo "=============================================="
