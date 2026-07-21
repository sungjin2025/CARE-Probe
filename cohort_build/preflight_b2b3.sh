#!/usr/bin/env bash
# =====================================================================
# B-2/B-3 preflight — check all required preconditions at once before multi-hour GPU runs.
# (Prevents mid-run breakage: weights, Milvus, GPU, disk, deps)
# Run: bash preflight_b2b3.sh
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

echo "==================== Common ===================="
[ -f cohort_v2.csv ]        && ok "cohort_v2.csv" || { no "cohort_v2.csv missing"; FAIL_B2=1; FAIL_B3=1; }
[ -f stay_anchor_year.csv ] && ok "stay_anchor_year.csv (E-shift/E-sev)" || { no "stay_anchor_year.csv missing"; FAIL_B3=1; }
[ -d "$EMBED" ] && ok "embed model: $EMBED" || { no "embed model missing: $EMBED"; FAIL_B2=1; FAIL_B3=1; }
python -c "import sklearn,numpy,scipy,torch,transformers" 2>/dev/null \
  && ok "py deps (sklearn/torch/transformers)" || { no "py deps incomplete"; FAIL_B2=1; FAIL_B3=1; }

echo "-- Milvus (required for retrieval; 3 collections) --"
if python _check_milvus.py --milvus_host localhost --milvus_port 19530 2>&1 | grep -qE "\[FAIL\]"; then
  no "Milvus check failed -> retrieval stage will stall. Start Milvus and retry."
  FAIL_B2=1; FAIL_B3=1
else
  ok "Milvus connection/collections OK"
fi

echo "-- GPU $GPU availability --"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$GPU" 2>/dev/null \
    && ok "GPU $GPU queried (check usage above)" || no "GPU $GPU query failed (check the index)"
else
  no "nvidia-smi missing"
fi

echo "-- Disk availability (npz storage) --"
df -h . | awk 'NR==1||NR==2'

echo ""
echo "==================== B-2 (4k Phi ablation) ===================="
[ -d "$PHI4K" ] && ok "4k Phi weights: $PHI4K" \
  || { no "4k weights missing -> huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir \"$PHI4K\""; FAIL_B2=1; }
[ -f runs_main/logs/analyze_phi3mini-128k_seed42.log ] && ok "128k comparison-baseline log present" || no "128k comparison log missing (some comparison-table rows omitted)"

echo ""
echo "==================== B-3 (Qwen full-n) ===================="
[ -d "$QWEN" ] && ok "Qwen weights: $QWEN" || { no "Qwen weights missing: $QWEN"; FAIL_B3=1; }
[ -f runs_main/results_main.csv ] && ok "n=2000 comparison baseline (results_main.csv) present" || no "results_main.csv missing (comparison table omitted)"

echo ""
echo "==================== Verdict ===================="
[ "$FAIL_B2" -eq 0 ] && echo "  B-2: ✅ GO   (bash run_b2_phi4k_ablation.sh)" || echo "  B-2: ❌ Resolve the [!!] items above first"
[ "$FAIL_B3" -eq 0 ] && echo "  B-3: ✅ GO   (bash run_b3_qwen_fulln.sh)"   || echo "  B-3: ❌ Resolve the [!!] items above first"
