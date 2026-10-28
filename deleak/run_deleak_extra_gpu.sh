#!/usr/bin/env bash
# =====================================================================
# run_deleak_extra_gpu.sh  (GPU) -- the three analyses still "with-field" in the
# manuscript, regenerated under de-leak (icd10_primary/assessment/plan/diagnosis
# removed). Reuses deleak_probe_runner.py / deleak_surface_runner.py (monkeypatch
# only, frozen code diff 0). Place all deleak_extra/ files in cohort_build_v2/.
#
#   export MILVUS_TOKEN=...
#   JOB=full     GPU=9 nohup bash run_deleak_extra_gpu.sh > runs_deleak/logs/extra_full.out 2>&1 &
#   JOB=norag    GPU=9 ...      # no-RAG Qwen, gen seeds 42 43 44
#   JOB=ragseeds GPU=9 ...      # RAG Qwen de-leak gen seeds 43 44 (pairs no-RAG 3 seeds)
#   JOB=surface  GPU=9 BACKBONES="qwen14b mediphi" ...   # surface signals (default: all 8)
#   JOB=all      GPU=9 ...      # everything sequentially (~6 GPU-days)
#
#   SMOKE=1 -> n=20, outputs under runs_smoke/ (never mixed with real outputs).
#
# Measured runtimes of the with-field originals (logs): full-cohort Qwen 17.3 h;
# no-RAG Qwen 4.0 h/seed; surface Qwen 22 h, Llama 12.4 h, MediPhi 8 h.
# Gemma-2: if output collapses, rerun with bf16 + TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

JOB="${JOB:?set JOB=full|norag|ragseeds|surface|all}"
GPU="${GPU:-9}"                                    # confirm with nvidia-smi
SMOKE="${SMOKE:-}"
EMBED=/mnt/nvme01/huggingface/models/Alibaba-NLP/gte-large-en-v1.5
MROOT=/mnt/nvme01/huggingface/models

export CRMSR_ENV=server
export USE_TF=0 USE_TORCH=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

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
QWEN="${MODELS[qwen14b]}"

for f in deleak_probe_runner.py deleak_surface_runner.py e0_probe_runner.py e0x_deployed_surface_runner.py; do
  [[ -f "$f" ]] || { echo "[fatal] $f missing in $(pwd)"; exit 1; }
done

# smoke mode: tiny n, separate output root
if [[ -n "$SMOKE" ]]; then
  ROOT=runs_smoke; N2K=20; NFULL=20
else
  ROOT=""; N2K=2000; NFULL=-1
fi
d() { local p="${ROOT:+$ROOT/}$1"; mkdir -p "$p/logs"; echo "$p"; }

need_milvus() { export MILVUS_TOKEN="${MILVUS_TOKEN:?set MILVUS_TOKEN (RAG required)}"; }

job_full() {   # (1) full-cohort scale-up, Qwen, de-leak
  need_milvus
  local o; o=$(d runs_b3)
  local pref="$o/probe_qwen14b-deleak_full_seed42"
  echo "[full] n=$NFULL -> $pref.npz  ($(date '+%F %T'))"
  python deleak_probe_runner.py --out_prefix "$pref" \
    --llm_model_path "$QWEN" --embed_model_path "$EMBED" \
    --gpu_id "$GPU" --n "$NFULL" --seed 42 --ckpt_every 500 --resume \
    > "$o/logs/extract_qwen14b-deleak_full.log" 2>&1 \
    && echo "[done] full $(date '+%T')" || echo "[FAIL] full (see $o/logs)"
}

job_norag() {  # (2) closed-book ablation, Qwen, de-leak, gen seeds 42-44 (no Milvus needed)
  local o; o=$(d runs_e0)
  for g in 42 43 44; do
    local pref="$o/noRAG-deleak_2k_s$g"
    echo "[norag] seed $g -> $pref.npz  ($(date '+%T'))"
    python deleak_probe_runner.py --out_prefix "$pref" \
      --llm_model_path "$QWEN" --gpu_id "$GPU" --n "$N2K" --seed "$g" \
      --no_retrieval --ckpt_every 500 --resume \
      > "$o/logs/noRAG-deleak_s$g.log" 2>&1 \
      && echo "[done] norag s$g $(date '+%T')" || echo "[FAIL] norag s$g"
    [[ -n "$SMOKE" ]] && break
  done
}

job_ragseeds() {  # (2b) RAG arm for no-RAG pairing: Qwen de-leak gen seeds 43 44
  need_milvus
  local o; o=$(d runs_main)
  for g in 43 44; do
    local pref="$o/probe_qwen14b-deleak_n2000_seed$g"
    [[ -n "$SMOKE" ]] && pref="$o/probe_qwen14b-deleak_smoke_seed$g"
    echo "[rag] seed $g -> $pref.npz  ($(date '+%T'))"
    python deleak_probe_runner.py --out_prefix "$pref" \
      --llm_model_path "$QWEN" --embed_model_path "$EMBED" \
      --gpu_id "$GPU" --n "$N2K" --seed "$g" --ckpt_every 500 --resume \
      > "$o/logs/deleak_qwen14b_seed$g.log" 2>&1 \
      && echo "[done] rag s$g $(date '+%T')" || echo "[FAIL] rag s$g"
    [[ -n "$SMOKE" ]] && break
  done
}

job_surface() {  # (3) deployed surface signals, 8 backbones, de-leak
  need_milvus
  local o; o=$(d runs_e0x)
  local order="${BACKBONES:-mediphi phi3mini-128k med42-8b openbio-8b llama31-8b mistral7b-v03 gemma2-9b qwen14b}"
  for m in $order; do
    local path="${MODELS[$m]:-}"
    [[ -n "$path" && -d "$path" ]] || { echo "[skip] $m : model path missing ($path)"; continue; }
    local npz="runs_main/probe_${m}-deleak_n2000_seed42.npz"
    local npz_arg=()
    if [[ -f "$npz" ]]; then npz_arg=(--main_npz "$npz"); else echo "[WARN] $npz missing -> no alignment check"; fi
    local out="$o/${m}-deleak_deployed_surface.csv"
    echo "[surface] $m -> $out  ($(date '+%T'))"
    python deleak_surface_runner.py --out_csv "$out" \
      --llm_model_path "$path" --embed_model_path "$EMBED" \
      --gpu_id "$GPU" --n "$N2K" --seed 42 --evidence_mode concat \
      "${npz_arg[@]}" --resume \
      > "$o/logs/${m}-deleak.log" 2>&1 \
      && echo "[done] surface $m $(date '+%T')" || echo "[FAIL] surface $m"
    [[ -n "$SMOKE" ]] && break
  done
}

echo "[deleak-extra] JOB=$JOB GPU=$GPU SMOKE=${SMOKE:-0}  $(date '+%F %T')"
case "$JOB" in
  full)     job_full ;;
  norag)    job_norag ;;
  ragseeds) job_ragseeds ;;
  surface)  job_surface ;;
  all)      job_full; job_norag; job_ragseeds; job_surface ;;
  *) echo "[fatal] unknown JOB=$JOB"; exit 1 ;;
esac
echo "[deleak-extra] JOB=$JOB finished $(date '+%F %T')"
[[ -z "$SMOKE" ]] && echo "next (CPU, after all GPU jobs): bash run_deleak_extra_downstream.sh"
