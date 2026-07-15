  #!/usr/bin/env bash
  M=/path/to/models/Qwen/Qwen2.5-14B-Instruct
  P=runs_e0/noRAG_2k_s
  Q=runs_main/probe_qwen14b_n2000_seed
  C=runs_e0/b4_3seed.csv
  for g in 43 44; do
    CRMSR_ENV=server python e0_probe_runner.py --llm_model_path "$M" --out_prefix ${P}$g --n 2000 --seed $g --no_retrieval
  done
  for g in 42 43 44; do
    for S in $(seq 42 51); do
      CRMSR_ENV=server python e0_crc.py --npz ${P}$g.npz --layer_idx 3 --probe_target loss --mondrian --seed $S --tag noRAG_g$g --results_csv $C
      CRMSR_ENV=server python e0_crc.py --npz ${Q}$g.npz --layer_idx 3 --probe_target loss --mondrian --seed $S --tag RAG_g$g --results_csv $C
    done
  done
  echo DONE_B4_3SEED
