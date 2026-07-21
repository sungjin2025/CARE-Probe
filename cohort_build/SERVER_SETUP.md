# CR-MSR M1.5 — Server Setup Guide

> Goal: run `imaging_runner_v2.py` in 5 modes × 3 seeds on a GPU server → produce the baseline table
> Assumptions: on the dev side (this PC), code/data live under `/path/to/cohort_build\`; the server is Linux + CUDA GPU with the Milvus daemon running
> Output location: `$WORK_DIR/runs_m15/` (pred_*.csv + *_metrics.json + *_report.txt × 15)

---

## 0. Pre-flight checks (on the server)

```bash
# 1) Python version
python --version          # 3.10 or 3.11 recommended

# 2) GPU visibility
nvidia-smi                # memory >= 40GB required (Qwen-14B fp16)

# 3) Is the Milvus daemon up?
# usually run via systemd or a docker container
docker ps | grep milvus   # or: systemctl status milvus
# confirm port 19530 is listening
ss -tlnp | grep 19530

# 4) Confirm model files exist
ls -lh /path/to/models/Qwen2.5-14B-Instruct
ls -lh /path/to/models/Alibaba-NLP/gte-large-en-v1.5
# (adjust paths to match your server)
```

---

## 1. Transfer code + data to the server

Copy the entire `cohort_build_v2/` folder to the server. **Must include**:

```
cohort_build_v2/
├── paths.py                       # environment switching
├── policyA_decode_v2.py
├── common_utils_v2.py
├── imaging_runner_v2.py
├── eval_v2.py
├── _check_milvus.py               # ← M1.5 sanity check
├── requirements_server.txt        # ← dependencies
├── run_all_modes.sh               # ← batch wrapper
├── SERVER_SETUP.md                # ← this document
├── exam_name_mapping.csv          # (optional, for paper appendix)
└── cohort_v2.csv                  # ← gold cohort (required)
```

Transfer example (rsync):
```bash
# dev → server
rsync -avz --exclude='__pycache__' \
  /path/to/cohort_build/ \
  user@server:/path/to/cohort_build/
```

Or with git:
```bash
# on the server
cd /path/to/workdir/
git clone <your-repo>
cd cohort_build_v2
```

---

## 2. Python environment setup

```bash
cd /path/to/cohort_build/

# virtual environment (just activate if it already exists)
python -m venv .venv
source .venv/bin/activate

# install torch separately to match your CUDA version
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# (or use the torch already installed on the server)

# remaining deps
pip install -r requirements_server.txt
```

---

## 3. Fill in the server block of paths.py (★ key step)

Replace the 6 TODOs in the `elif ENV == "server":` block (lines 32–43) of `paths.py` with your own server paths.

### 3.1 Which paths to fill in

| Variable | What it is | Example (change these) |
|---|---|---|
| `RAD_DETAIL_GZ_PATH` | Absolute path to MIMIC-IV-Note `radiology_detail.csv.gz` | `/path/to/mimic-iv-note/note/radiology_detail.csv.gz` |
| `RADIOLOGY_CSV` | MIMIC-IV-Note `radiology.csv` (report text body) | `/path/to/mimic-iv-note/note/radiology.csv` |
| `EDSTAYS_CSV` | MIMIC-IV-ED `edstays.csv` | `/path/to/mimic-iv-ed/edstays.csv` |
| `MIMIC_ED_INPUT_HPI_CSV` | HPI extraction CSV from the first paper (reused) | `/path/to/preprocessed/mimic_ed_input_hpi.csv` |
| `LEGACY_COHORT_CSV` | 4-class cohort from the first paper (for cross-validation, optional) | `/path/to/preprocessed/cohort_imaging_labels_4class.csv` |
| `WORK_DIR` | Directory where outputs (`cohort_v2.csv`, etc.) are stored | `/path/to/cohort_build` |

### 3.2 (Optional) If you only run M1.5

If you already built `cohort_v2.csv` on dev and copied it to the server, **no rebuild is needed**. Only `WORK_DIR` needs to be correct (`cohort_v2.csv` must be at `$WORK_DIR/cohort_v2.csv`).
The other 5 variables are only needed when rebuilding data → leaving them as `None` or a placeholder does not affect M1.5 execution.

### 3.3 smoke test

```bash
export CRMSR_ENV=server
python paths.py
# OK if it prints something like:
# CRMSR_ENV=server
# WORK_DIR=/path/to/cohort_build
# ...
```

---

## 4. Milvus collection sanity check

Confirm the 3 collections (`MedRAG_PubMed`, `MedRAG_Wikipedia`, `Medical_Guidelines`) are up and the embedding dimensions match:

```bash
python _check_milvus.py \
    --embed_model_path /path/to/models/Alibaba-NLP/gte-large-en-v1.5 \
    --milvus_host localhost \
    --milvus_port 19530
```

Expected output (gist):
```
[OK]   connected to Milvus at localhost:19530
[OK]   collection found: MedRAG_PubMed  (entities=...)
[OK]   collection found: MedRAG_Wikipedia
[OK]   collection found: Medical_Guidelines
[OK]   embedder loaded: gte-large-en-v1.5  (out_dim=1024)
[OK]   MedRAG_PubMed: dim match (1024)
...
[OK]   MedRAG_PubMed: search OK  (top-1 score=0.78xx, preview='...')
=== ALL CHECKS PASSED ===
```

**If you get a FAIL**:
- `collection missing` → the collection name may differ. Compare the `COLLECTION_*` constants at `imaging_runner_v2.py:75–77` with the actual Milvus collection names.
- `dim mismatch` → the embedder may be a different model than the first paper. gte-large-en-v1.5 is 1024-dim.
- `connect failed` → the Milvus daemon is not running or is on a different port.

---

## 5. (Optional) Single-mode dry run

Before the full batch, verify behavior with the `vanilla` mode on a small sample (n=10):

```bash
export CRMSR_ENV=server
python imaging_runner_v2.py \
  --mode vanilla \
  --seed 42 \
  --out_csv /tmp/dry_vanilla.csv \
  --llm_model_path   /path/to/models/Qwen2.5-14B-Instruct \
  --embed_model_path /path/to/models/Alibaba-NLP/gte-large-en-v1.5 \
  --gpu_id 0 \
  --n 10
# when done:
head -2 /tmp/dry_vanilla.csv
# OK if you see stay_id, decision, pred_label_set, ...
```

---

## 6. Run the full batch

```bash
export CRMSR_ENV=server
export LLM_PATH=/path/to/models/Qwen2.5-14B-Instruct
export EMBED_PATH=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
export GPU_ID=0
export N_STAYS=1000        # or 3000 (experiment scale)

bash run_all_modes.sh
# or in the background:
# nohup bash run_all_modes.sh > /tmp/crmsr_batch.out 2>&1 &
```

Monitoring progress:
```bash
tail -f $WORK_DIR/runs_m15/BATCH_SUMMARY.txt
# individual run logs:
tail -f $WORK_DIR/runs_m15/logs/vanilla_qwen14b_seed42_n1000.log
```

**Estimated runtime (rough)**:
- vanilla:        fastest (no retrieval)
- single_*:       medium
- msrc:           slowest (3 source × top-k × fusion)
- total 15 runs × ~tens of minutes/run = a few hours to 1 day (depends on GPU/LLM throughput)

---

## 7. Collect results (server → dev)

```bash
# on the server
tar -czf m15_results.tar.gz -C $WORK_DIR runs_m15/

# on dev
scp user@server:/path/to/cohort_build/m15_results.tar.gz .
tar -xzf m15_results.tar.gz -C /path/to/cohort_build/
```

After collection, aggregate results on the dev side:
```python
# (example) aggregate metrics for 5 modes × 3 seeds
import json, glob, pandas as pd
rows = []
for f in glob.glob("runs_m15/*_metrics.json"):
    m = json.load(open(f))
    rows.append({"tag": f, **m["aggregate"]})
print(pd.DataFrame(rows).sort_values("tag"))
```

---

## 8. Re-run / Resume

- `run_all_modes.sh` uses `RESUME=1` by default — existing `pred_*.csv` / `*_metrics.json` are skipped.
- To re-run only a specific mode:
  ```bash
  rm $WORK_DIR/runs_m15/pred_msrc_*.csv
  bash run_all_modes.sh
  ```
- To re-run only a specific seed:
  ```bash
  SEEDS="44" MODES="msrc" bash run_all_modes.sh
  ```

---

## 9. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `OOM (CUDA out of memory)` | Qwen-14B fp16 needs about 28GB. Move to a larger GPU via `--gpu_id`, or kill other processes |
| `HF offline` error | Env var can't find the cached model. Temporarily disable the offline flag at `imaging_runner_v2.py:60-64`: `export HF_HUB_OFFLINE=0` |
| Milvus connect timeout | Daemon not running / wrong port. Re-run `_check_milvus.py` |
| `cohort_v2.csv not found` | Check that the `WORK_DIR` path matches paths.py |
| abstain ratio 100% for every seed | τ value too high → confirm the first paper's defaults are kept (CT 0.60, others 0.72) |
| `pymilvus` version mismatch | The `pymilvus` client minor version must match the server's Milvus version |

---

## 10. M1.5 outputs → M2 comparison anchor

When the batch finishes, you should have the following table (15 runs):

```
runs_m15/
├── BATCH_SUMMARY.txt
├── pred_vanilla_qwen14b_seed42_n1000.csv
├── pred_vanilla_qwen14b_seed43_n1000.csv
├── pred_vanilla_qwen14b_seed44_n1000.csv
├── pred_single_pubmed_qwen14b_seed42_n1000.csv
│   ... (15 pred files total)
├── vanilla_qwen14b_seed42_n1000_metrics.json
├── vanilla_qwen14b_seed42_n1000_report.txt
│   ... (15 metrics + 15 reports total)
└── logs/
    └── *.log
```

→ This table is the **anchor for comparing M2 (retriever swap) results**. Without it, M2 results cannot be interpreted.

---

## Appendix: environment variables quick reference

| Variable | Meaning | Default |
|---|---|---|
| `CRMSR_ENV` | dev/server switch | `dev` (must be `server` on the server) |
| `LLM_PATH` | LLM model path | (required) |
| `EMBED_PATH` | embedder model path | (required) |
| `GPU_ID` | CUDA device | `0` |
| `N_STAYS` | stays per run | `1000` |
| `SEEDS` | seed list | `"42 43 44"` |
| `MODES` | mode list | 5 modes |
| `MILVUS_HOST` | Milvus host | `localhost` |
| `MILVUS_PORT` | Milvus port | `19530` |
| `OUT_DIR` | output directory | `$WORK_DIR/runs_m15` |
| `COHORT_CSV` | gold cohort path | `$WORK_DIR/cohort_v2.csv` |
| `SKIP_CHECK` | skip Milvus check | `0` |
| `RESUME` | skip existing results | `1` |

Written: 2026-05-29 (M1.5 deliverable bundle)
