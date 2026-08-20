# CARE-Probe

Code and cohort-build scripts for:

> **CARE-Probe: A conformal risk-control framework for physician deferral with bounded committed error in open-weight medical LLMs**
> Sungjin Jang, Woosung Jung, Beakcheol Jang. Graduate School of Information, Yonsei University; Department of Emergency Medicine, Gupo Sungshim Hospital.
> *Artificial Intelligence in Medicine* (under review, 2026).

CARE-Probe is a deployable safety mechanism for governing LLM-based clinical decision
support. It reads a model's internal state with a lightweight linear probe and wraps the
resulting score in distribution-free risk control (RCPS/Learn-Then-Test), bounding the
joint committed proxy-error rate (an under-imaging discordance proxy) below a preset
target with a finite-sample guarantee.

---

## ⚠️ Data availability and privacy

This repository contains **code only**. It does **not** contain any patient data or
derived patient-level artifacts.

The study uses **MIMIC-IV** and **MIMIC-IV-ED**, which are available via
[PhysioNet](https://physionet.org/) to credentialed users under the PhysioNet
Credentialed Health Data Use Agreement. To reproduce the results you must obtain
credentialed access and regenerate the cohort locally with the scripts here. Per the
DUA, no MIMIC-derived data (cohort tables, hidden-state vectors, per-encounter
predictions) is redistributed in this repository.

`cohort_build/exam_name_mapping.csv` is a deterministic exam-name → (modality, body
part) mapping vocabulary and contains no patient information.

---

## Repository layout

```
cohort_build/          all runnable code
  _m12a..e_*.py        cohort construction (exam-name survey → mapping → imaging events → cohort → cross-validation)
  add_anchor_year.py   chronology recovery (stay → subject → patient anchor_year_group)
  imaging_runner_v2.py multi-source retrieval + LLM generation (single / msrc modes)
  e0_probe_runner.py   MAIN internal-state probe extraction (per-source top-3 concat evidence)
  e0_conflict_runner.py / e0_uncertainty_runner.py   surface-signal computation
  e0_analyze.py / e0_probe_analyze.py                surface-signal & probe analysis
  e0_crc.py            Conformal Risk Control (Hoeffding–Bentkus, fixed-sequence testing, Mondrian)
  e0_aggregate.py      result aggregation
  e0x_deployed_surface_runner.py / e0x_analyze.py    deployed-surface diagnostic (concat vs fused)
  run_*.sh             experiment drivers (B-2/B-3/B-4, C-2/C-3, E0X)
  b1_setloss_cert.py               two-sided set-loss certification, alpha grid (S4)
  power_note_sz.py                 calibration-set size power note (S8)
  _matched_pair.py                 matched general<->medical pair analysis (S9)
  _delta_sweep.py                  delta-sensitivity sweep (S10)
  c2_local_headtohead.py           lightweight head-to-head: probe vs cardinality vs random (S12)
  _e0_pilot_layer_replication.py   pilot layer-grid basis (S20)
  _shift_probe_auc_replication.py  probe discrimination under temporal shift (S21)
  _random_score_crc_baseline.py    random-score CRC non-vacuity baseline (S16)
  _single_label_dilution.py        single-label control (S17)
  _subgroup_audit.py               demographic equity audit (S18)
  _esi3_threegroup_mondrian.py     ESI-3 three-group severity budget (S19)
  paths.py             environment-dependent paths (dev / server)
  requirements_server.txt   Python dependencies
  SERVER_SETUP.md      GPU + Milvus setup notes
results/               aggregate result tables (no patient data; map to Online Resources)
```

---

## Reproduction pipeline

Prerequisites: credentialed MIMIC-IV / MIMIC-IV-ED access, a GPU server, a running
[Milvus](https://milvus.io/) instance with the three MedRAG collections
(PubMed / Wikipedia / Guidelines) embedded with `gte-large-en-v1.5`, and the eight
open-weight backbones downloaded locally.

```bash
pip install -r cohort_build/requirements_server.txt
export MILVUS_TOKEN=<user:pass>      # Milvus auth injected at runtime (never hardcoded)

# 1. Build cohort (regenerates from your local MIMIC copy)
python cohort_build/_m12a_survey_exam_names.py
python cohort_build/_m12b_build_mapping.py
python cohort_build/_m12c_build_imaging_events_v2.py
python cohort_build/_m12d_build_cohort_v2.py
python cohort_build/add_anchor_year.py
python cohort_build/_m12e_cross_validate.py        # reliability (Cohen's kappa)

# 2. Surface signals + internal-state probe extraction (MAIN)
CRMSR_ENV=server python cohort_build/e0_conflict_runner.py --gpu_id 6 --n 500 --seed 42
CRMSR_ENV=server python cohort_build/e0_probe_runner.py \
    --out_prefix runs_main/probe_qwen14b_n2000_seed42 \
    --llm_model_path /path/to/Qwen/Qwen2.5-14B-Instruct \
    --embed_model_path /path/to/Alibaba-NLP/gte-large-en-v1.5 \
    --gpu_id 6 --n 2000 --seed 42

# 3. Probe training + CRC certification (deployed protocol: --layer_idx 3 --probe_target loss)
bash cohort_build/run_main_experiment.sh    # all 8 backbones, 3 sub-seeds x 10 split-seeds
# single-run example:
CRMSR_ENV=server python cohort_build/e0_crc.py \
    --npz runs_main/probe_qwen14b_n2000_seed42.npz --layer_idx 3 --probe_target loss
# surface-signal / probe analyses take their inputs explicitly:
#   e0_analyze.py --pred_csv <e0 predictions csv>;  e0_probe_analyze.py --npz <probe npz>

# 4. Ablations (Supplementary)
bash cohort_build/run_b2_phi4k_ablation.sh          # B-2: Phi-3-mini-4k confound
bash cohort_build/run_b3_qwen_fulln.sh              # B-3: full-cohort scale-up
bash cohort_build/run_b4_3seed.sh                   # B-4: no-retrieval ablation
bash cohort_build/run_c2_baselines.sh               # C-2: head-to-head baselines
bash cohort_build/run_c3_shift_sev_allbackbones.sh  # C-3: temporal shift + severity (8 backbones)
bash cohort_build/run_e0x.sh                        # E0X: deployed-surface diagnostic

# 5. Supplementary certifications and baselines (aggregate outputs in results/)
python cohort_build/b1_setloss_cert.py                 # S4:  two-sided set-loss certification (alpha grid)
python cohort_build/power_note_sz.py                   # S8:  calibration-set size power note
python cohort_build/_matched_pair.py                   # S9:  matched general<->medical pair analysis
python cohort_build/_delta_sweep.py                    # S10: delta-sensitivity of the guarantee
python cohort_build/c2_local_headtohead.py             # S12: probe vs cardinality vs random under frozen CRC

# 6. Internal-review reproductions (aggregate outputs in results/, mapped to Online Resources)
python cohort_build/_e0_pilot_layer_replication.py     # S20: probe-layer pilot grid basis
python cohort_build/_shift_probe_auc_replication.py    # S21: probe discrimination measured under temporal shift
python cohort_build/_random_score_crc_baseline.py      # S16: random-score CRC baseline (non-vacuity)
python cohort_build/_single_label_dilution.py          # S17: single-label control (failure is not multi-label dilution)
python cohort_build/_subgroup_audit.py                 # S18: demographic equity audit of the global policy
python cohort_build/_esi3_threegroup_mondrian.py       # S19: ESI-3 three-group severity budget
python cohort_build/_conditional_sgr_card.py           # S24: conditional-risk certification (SGR)
```

The S15 α-sweep (`results/alpha_sweep_B5_results.csv`) is generated by looping
`e0_crc.py --alpha` over {0.02, 0.05, 0.10, 0.15, 0.20}; there is no separate driver.

Reproducibility: greedy decoding (deterministic); the probe layer is fixed at relative
depth 0.875 — selected on a single-backbone pilot screen (Qwen2.5-14B, n=500;
`results/pilot_layer_grid.csv`) and frozen before the multi-seed MAIN protocol and all
test-set evaluations (not tuned to the certified CT-miss loss); subsample/split seeds
{42, 43, 44} × {42…51}; α = δ = 0.10. See `SERVER_SETUP.md` for environment details.

---

## Aggregate results (`results/`)

The `results/` directory holds **aggregate statistics only** (per-backbone / per-group /
per-seed; no patient-level rows), matching the manuscript's supplementary tables. These
are the reproduction targets — running the scripts above on your locally regenerated
cohort should recover them.

| File | Online Resource | Content |
|------|-----------------|---------|
| `results_b3.csv` | S2 | full-cohort scale-up (N = 7,814), per split seed |
| `b4_3seed.csv` | S3 | no-retrieval ablation, 3 sub-seeds × 10 split-seeds per arm |
| `b1_setloss_results.csv` | S4 | two-sided set-loss certification over the α grid |
| `delta_sweep_results.csv` | S10 | δ-sensitivity of the guarantee |
| `accuracy_8backbone.csv` | S13 | off-the-shelf generation accuracy, 8 backbones |
| `alpha_sweep_B5_results.csv` | S15 | risk-budget (α) sweep |
| `pilot_layer_grid.csv` | S20 | pilot position×layer probe AUC grid |
| `random_score_crc_baseline.csv` | S16 | probe vs random-score commit at the same certified joint risk |
| `single_label_dilution.csv` | S17 | surface-signal AUC: multi-label vs single-label decision |
| `subgroup_audit_{qwen14b,8backbone}.csv` | S18 | commit / joint risk decomposed by age, sex, race |
| `esi3_threegroup_{qwen14b,8backbone}.csv` | S19 | ESI-3 isolated three-group Mondrian budget |
| `shift_probe_auc_replication.csv` | S21 | probe discrimination trained on earlier bins, tested on 2017–2019 |
| `conditional_sgr_results.csv` | S24 | conditional-risk (committed-case error) SGR certification |

---

## Citation

```bibtex
@article{jang2026careprobe,
  title  = {CARE-Probe: A conformal risk-control framework for physician deferral
            with bounded committed error in open-weight medical LLMs},
  author = {Jang, Sungjin and Jung, Woosung and Jang, Beakcheol},
  journal= {Artificial Intelligence in Medicine},
  year   = {2026},
  note   = {Under review}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). MIMIC-IV / MIMIC-IV-ED
data are governed separately by the PhysioNet Credentialed Health Data Use Agreement and
are **not** redistributed here.
