# CR-MSR M1.5 — Server Setup Guide

> 대상: GPU 서버에서 `imaging_runner_v2.py` 5 modes × 3 seeds 실행 → baseline 표 산출
> 가정: dev side(이 PC) 코드/데이터는 `/path/to/cohort_build\`에서 작업됨, 서버는 Linux + CUDA GPU + Milvus 데몬 가동 중
> 산출물 위치: `$WORK_DIR/runs_m15/` (pred_*.csv + *_metrics.json + *_report.txt × 15)

---

## 0. 사전 점검 (서버에서)

```bash
# 1) Python 버전
python --version          # 3.10 또는 3.11 권장

# 2) GPU 가시화
nvidia-smi                # 메모리 ≥ 40GB 필요 (Qwen-14B fp16)

# 3) Milvus 데몬 떠 있는지
# 보통 systemd 또는 docker 컨테이너로 실행됨
docker ps | grep milvus   # 또는: systemctl status milvus
# 포트 19530 listening 확인
ss -tlnp | grep 19530

# 4) 모델 파일 존재 확인
ls -lh /path/to/models/Qwen2.5-14B-Instruct
ls -lh /path/to/models/Alibaba-NLP/gte-large-en-v1.5
# (경로는 본인 서버에 맞게)
```

---

## 1. 코드 + 데이터 서버로 전송

`cohort_build_v2/` 폴더 전체를 서버에 복사. **반드시 포함해야 할 것**:

```
cohort_build_v2/
├── paths.py                       # 환경 분기
├── policyA_decode_v2.py
├── common_utils_v2.py
├── imaging_runner_v2.py
├── eval_v2.py
├── _check_milvus.py               # ← M1.5 sanity check
├── requirements_server.txt        # ← 의존성
├── run_all_modes.sh               # ← 배치 래퍼
├── SERVER_SETUP.md                # ← 이 문서
├── exam_name_mapping.csv          # (선택, 페이퍼 부록용)
└── cohort_v2.csv                  # ← gold cohort (필수)
```

전송 예시 (rsync):
```bash
# dev → 서버
rsync -avz --exclude='__pycache__' \
  /path/to/cohort_build/ \
  user@server:/path/to/cohort_build/
```

또는 git 사용 시:
```bash
# 서버에서
cd /path/to/workdir/
git clone <your-repo>
cd cohort_build_v2
```

---

## 2. Python 환경 설정

```bash
cd /path/to/cohort_build/

# 가상환경 (이미 있으면 활성화만)
python -m venv .venv
source .venv/bin/activate

# torch는 CUDA 버전에 맞춰 별도 설치
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# (또는 서버에 이미 깔린 torch 사용)

# 나머지 deps
pip install -r requirements_server.txt
```

---

## 3. paths.py 서버 블록 채우기 (★ 핵심)

`paths.py` 의 `elif ENV == "server":` 블록(line 32–43)의 6개 TODO를 본인 서버 경로로 교체.

### 3.1 어떤 경로를 채우는가

| 변수 | 무엇 | 예시 (변경하세요) |
|---|---|---|
| `RAD_DETAIL_GZ_PATH` | MIMIC-IV-Note `radiology_detail.csv.gz` 절대경로 | `/mnt/data/mimic-iv-note/note/radiology_detail.csv.gz` |
| `RADIOLOGY_CSV` | MIMIC-IV-Note `radiology.csv` (텍스트 본문) | `/mnt/data/mimic-iv-note/note/radiology.csv` |
| `EDSTAYS_CSV` | MIMIC-IV-ED `edstays.csv` | `/mnt/data/mimic-iv-ed/edstays.csv` |
| `MIMIC_ED_INPUT_HPI_CSV` | 첫 페이퍼 HPI 추출 CSV (재사용) | `/mnt/data/preprocessed/mimic_ed_input_hpi.csv` |
| `LEGACY_COHORT_CSV` | 첫 페이퍼 4-class cohort (cross-validate용, 선택) | `/mnt/data/preprocessed/cohort_imaging_labels_4class.csv` |
| `WORK_DIR` | 산출물(`cohort_v2.csv` 등) 저장 디렉터리 | `/path/to/cohort_build` |

### 3.2 (선택) M1.5만 실행할 거라면

`cohort_v2.csv`를 이미 dev에서 만들어 서버로 옮겼다면, **재빌드 불필요**. `WORK_DIR`만 정확하면 됨 (cohort_v2.csv가 `$WORK_DIR/cohort_v2.csv`에 있어야 함).
나머지 5개 변수는 데이터 재빌드 시에만 필요 → `None` 두거나 placeholder로 둬도 M1.5 실행에는 영향 없음.

### 3.3 smoke test

```bash
export CRMSR_ENV=server
python paths.py
# 다음과 같이 출력되면 OK:
# CRMSR_ENV=server
# WORK_DIR=/path/to/cohort_build
# ...
```

---

## 4. Milvus collection sanity check

3개 컬렉션 (`MedRAG_PubMed`, `MedRAG_Wikipedia`, `Medical_Guidelines`) 이 떠 있고 임베딩 차원이 일치하는지 확인:

```bash
python _check_milvus.py \
    --embed_model_path /path/to/models/Alibaba-NLP/gte-large-en-v1.5 \
    --milvus_host localhost \
    --milvus_port 19530
```

기대 출력 (요지):
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

**FAIL이 나오면**:
- `collection missing` → 컬렉션 이름이 다를 수 있음. `imaging_runner_v2.py:75–77`의 `COLLECTION_*` 상수와 실제 Milvus 컬렉션명 비교.
- `dim mismatch` → embedder가 첫 페이퍼와 다른 모델일 가능성. gte-large-en-v1.5는 1024-dim.
- `connect failed` → Milvus 데몬 미가동 또는 다른 포트.

---

## 5. (선택) 단일 mode dry-run

전체 batch 전에 `vanilla` 모드 작은 표본(n=10)으로 동작 확인:

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
# 끝나면:
head -2 /tmp/dry_vanilla.csv
# stay_id, decision, pred_label_set, ... 가 보이면 OK
```

---

## 6. 전체 배치 실행

```bash
export CRMSR_ENV=server
export LLM_PATH=/path/to/models/Qwen2.5-14B-Instruct
export EMBED_PATH=/path/to/models/Alibaba-NLP/gte-large-en-v1.5
export GPU_ID=0
export N_STAYS=1000        # 또는 3000 (실험 규모)

bash run_all_modes.sh
# 또는 백그라운드:
# nohup bash run_all_modes.sh > /tmp/crmsr_batch.out 2>&1 &
```

진행 상황 모니터링:
```bash
tail -f $WORK_DIR/runs_m15/BATCH_SUMMARY.txt
# 개별 run 로그:
tail -f $WORK_DIR/runs_m15/logs/vanilla_qwen14b_seed42_n1000.log
```

**예상 소요시간 (대략)**:
- vanilla:        가장 빠름 (retrieval 없음)
- single_*:       중간
- msrc:           가장 느림 (3 source × top-k × fusion)
- 총 15 run × ~수십 분/run = 수 시간 ~ 1일 (GPU·LLM throughput 의존)

---

## 7. 결과 수거 (서버 → dev)

```bash
# 서버에서
tar -czf m15_results.tar.gz -C $WORK_DIR runs_m15/

# dev에서
scp user@server:/path/to/cohort_build/m15_results.tar.gz .
tar -xzf m15_results.tar.gz -C /path/to/cohort_build/
```

수거 후 dev side에서 결과 통합:
```python
# (예시) 5 modes × 3 seeds 메트릭 집계
import json, glob, pandas as pd
rows = []
for f in glob.glob("runs_m15/*_metrics.json"):
    m = json.load(open(f))
    rows.append({"tag": f, **m["aggregate"]})
print(pd.DataFrame(rows).sort_values("tag"))
```

---

## 8. 재실행 / Resume

- `run_all_modes.sh` 는 기본적으로 `RESUME=1` — 이미 존재하는 `pred_*.csv` / `*_metrics.json` 은 건너뜀.
- 특정 mode만 다시 돌리려면:
  ```bash
  rm $WORK_DIR/runs_m15/pred_msrc_*.csv
  bash run_all_modes.sh
  ```
- 시드만 다시 돌리려면:
  ```bash
  SEEDS="44" MODES="msrc" bash run_all_modes.sh
  ```

---

## 9. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `OOM (CUDA out of memory)` | Qwen-14B fp16은 약 28GB 필요. `--gpu_id` 큰 GPU로 옮기거나, 다른 프로세스 종료 |
| `HF offline` 오류 | 환경변수가 cached 모델 못 찾음. `imaging_runner_v2.py:60-64`의 offline 플래그를 일시 해제: `export HF_HUB_OFFLINE=0` |
| Milvus connect timeout | 데몬 미기동 / 포트 오류. `_check_milvus.py` 다시 실행 |
| `cohort_v2.csv not found` | `WORK_DIR` 경로가 paths.py와 일치하는지 확인 |
| 모든 seed에서 abstain 비율 100% | τ 값 너무 높음 → 첫 페이퍼 기본값 유지인지 확인 (CT 0.60, 나머지 0.72) |
| `pymilvus` 버전 mismatch | 서버 Milvus 버전과 `pymilvus` 클라이언트 minor 일치 필요 |

---

## 10. M1.5 산출물 → M2 비교 anchor

배치 끝나면 다음 표가 있어야 함 (15 runs):

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

→ 이 표가 **M2(retriever 교체) 결과 비교의 anchor**. 그 전엔 M2 결과 해석 불가.

---

## 부록: 환경변수 빠른 정리

| 변수 | 의미 | 기본 |
|---|---|---|
| `CRMSR_ENV` | dev/server 분기 | `dev` (서버에선 반드시 `server`) |
| `LLM_PATH` | LLM 모델 경로 | (필수) |
| `EMBED_PATH` | 임베더 모델 경로 | (필수) |
| `GPU_ID` | CUDA device | `0` |
| `N_STAYS` | run당 stay 수 | `1000` |
| `SEEDS` | seed 목록 | `"42 43 44"` |
| `MODES` | mode 목록 | 5 modes |
| `MILVUS_HOST` | Milvus 호스트 | `localhost` |
| `MILVUS_PORT` | Milvus 포트 | `19530` |
| `OUT_DIR` | 결과 디렉터리 | `$WORK_DIR/runs_m15` |
| `COHORT_CSV` | gold cohort 경로 | `$WORK_DIR/cohort_v2.csv` |
| `SKIP_CHECK` | Milvus check 생략 | `0` |
| `RESUME` | 기존 결과 건너뛰기 | `1` |

작성: 2026-05-29 (M1.5 deliverable bundle)
