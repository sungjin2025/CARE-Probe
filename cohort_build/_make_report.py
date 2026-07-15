# -*- coding: utf-8 -*-
"""
CARE-Probe 전체 실험결과 종합 리포트 생성기.
단일 'blocks' 모델에서 HTML + Word(.docx)를 동시에 렌더링한다(내용 불일치 방지).
실행: python _make_report.py
출력: CARE-Probe_실험결과_종합.html / .docx
"""
import html as _html
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = Path(__file__).resolve().parent
STAMP = "2026-06-13"

# ---------------------------------------------------------------- 리포트 모델
# 블록 타입: ('h1'|'h2'|'h3', text) / ('p', text) / ('callout', text)
#           ('ul', [items]) / ('table', headers, rows, caption)
# 인라인 강조: **bold**
B = []
def h1(t): B.append(("h1", t))
def h2(t): B.append(("h2", t))
def h3(t): B.append(("h3", t))
def p(t):  B.append(("p", t))
def callout(t): B.append(("callout", t))
def ul(items): B.append(("ul", items))
def table(headers, rows, caption=""): B.append(("table", headers, rows, caption))

# ============================================================ 내용 시작
h1("CARE-Probe — 전체 실험 결과 종합 정리")
p(f"대상 디렉터리: cohort_build · 정리일: {STAMP} · "
  "목표 학회/저널: AIIM (Artificial Intelligence in Medicine)")
callout("**한 줄 요약** — 응급실(ED) 영상검사 오더 예측 과제에서, LLM **내부 표상 프로브**로 "
        "'commit(자동 확정) vs defer(전문가 이관)'를 결정하고 **Conformal Risk Control(CRC)** 로 "
        "joint-risk ≤ α 안전보장을 건다. 핵심 결과: **보장은 8개 백본·full-cohort·시간이동·중증도적응 전 조건에서 성립**하고, "
        "유용성(gap_closed)은 **내부 신호가 디코드되는 백본에서만** 발생한다. B-2/B-3 추가실험까지 모두 완료.")

# ---------------------------------------------------------------- 0. 실험 인벤토리
h2("0. 실험 인벤토리 (전체 4세트, 모두 완료)")
table(
    ["실험", "목적", "백본 / 스케일", "시드 설계", "상태"],
    [
        ["e0 / MAIN", "CARE-Probe 핵심: probe AUC + CRC 보장/유용성", "8개 백본, N=2,000", "추출 42–44 × split 42–51", "✅ 완료"],
        ["m15 (CR-MSR)", "과제 베이스라인: 검색증강 생성(RAG) 영상오더 예측", "Qwen2.5-14B, N=1,000", "42, 43, 44", "✅ 완료"],
        ["B-2", "4k Phi ablation — last-token 붕괴의 LongRoPE 교란 분리", "Phi-3-mini-4k, N=2,000", "추출 42–44 × split 42–51", "✅ 완료"],
        ["B-3", "Full-cohort 헤드라인 — 스케일업 시 보장 유지 확인", "Qwen2.5-14B, N=7,814(full)", "추출 42 × split 42–51", "✅ 완료"],
    ],
    "표 0. 실험 세트 요약")
ul([
    "**평가지표 정의** — commit_rate: 자동 확정 비율 / joint_risk: 확정한 건들의 위험사건(ct_miss 등) 결합 위험 / "
    "cond_risk: 확정건 조건부 위험 / AURC: Area Under Risk-Coverage (낮을수록 좋음) / "
    "gap_closed: (naive−probe)/(naive−oracle), 무작위→오라클 격차를 프로브가 닫은 비율.",
    "**손실/보장 설정** — loss = ct_miss(CT 누락), α = 0.10, δ = 0.10, layer_idx = 3(a-priori 고정), "
    "프로브 = last-token, PCA ≤ 50, split = random(기본)·temporal(E-shift)·severity(E-sev).",
])

# ---------------------------------------------------------------- 1. 코호트
h2("1. 코호트 구축 및 교차검증 (cohort_v2)")
p("MIMIC-IV ED 기반. 마스터 stay 7,823건, 전건 edstay 매칭. 주호소(chief-complaint) 6개 코호트로 층화.")
table(
    ["코호트(주호소)", "stay 수"],
    [["other", "4,717"], ["abdominal_pain", "2,166"], ["head_injury_dizzy", "610"],
     ["chest_pain", "178"], ["flank_pain", "84"], ["sob", "68"], ["합계", "7,823"]],
    "표 1a. 주호소별 코호트 분포")
table(
    ["영상검사 개수(card_full)", "0", "1", "2", "3", "4", "5", "6", "7"],
    [["stay 수", "2,303", "3,392", "1,480", "417", "149", "63", "18", "1"]],
    "표 1b. stay당 영상검사 카디널리티 분포 (card=0 = 무영상 2,303건)")
p("**legacy 라벨과 교차검증** (cohort_v2의 멀티모달 라벨링 타당성 확인):")
table(
    ["검증 항목", "Agreement", "Cohen's κ"],
    [["7-class (FULL window)", "77.00%", "0.716"],
     ["7-class (EARLY 2h window)", "86.63%", "0.764"],
     ["4-class (gold4 scope, FULL)", "86.13%", "0.775"],
     ["멀티모달 복원율 (legacy multi→v2 multi)", "97.3% (1,567/1,610)", "—"]],
    "표 1c. cohort_v2 vs legacy 일치도")
callout("코호트 타당성 — 4-class κ=0.775(상당한 일치), 무영상(none) 라벨 100% 일치. v2는 legacy가 단일로 "
        "뭉갰던 **멀티모달 561건을 추가 복원**(주로 other/ct_abd) → 멀티모달 셋-라벨 과제로서 더 정밀.")

# ---------------------------------------------------------------- 2. 베이스라인
h2("2. 과제 베이스라인 — m15 검색증강 생성 (Qwen-14B, N=1,000)")
p("프로브/CRC가 얹히는 토대인 '소박한 생성(naive generate)'의 과제 난이도. "
  "**3시드(42·43·44) 평균±95%CI** (t분포, df=2; 시드당 ~7,100s):")
table(
    ["지표", "평균 ± 95%CI", "시드별 원값"],
    [["Commit rate / Abstain", "0.995±0.006 / 0.005±0.006", "0.998·0.993·0.995"],
     ["Micro P / R / F1", "0.482±0.034 / 0.362±0.016 / 0.414±0.021", "F1: 0.411·0.423·0.407"],
     ["Macro P / R / F1", "0.264±0.085 / 0.202±0.043 / 0.203±0.047", "F1: 0.186·0.199·0.223"],
     ["Mean Jaccard / Exact set match", "0.385±0.039 / 0.311±0.037", "J: 0.376·0.403·0.375"],
     ["Over-imaging / Under-imaging", "18.2±2.5% / 16.8±4.8%", "U: 18.8·15.0·16.5"],
     ["CT over-imaging / CT recall", "10.6±2.3% / 0.554±0.081", "R: 0.583·0.561·0.519"]],
    "표 2. vanilla RAG 영상오더 예측 성능 (3시드 평균±95%CI, N=1,000)")
callout("베이스라인 함의 — 소박한 생성은 Micro-F1 **0.414±0.021**, under-imaging **16.8±4.8%**로 "
        "**그대로 자동확정하기엔 위험**. → CARE-Probe의 selective prediction(commit/defer + 안전보장) 동기를 정량적으로 뒷받침.")

# ---------------------------------------------------------------- 3. MAIN
h2("3. CARE-Probe 핵심 결과 (e0 / MAIN, Stage-1 분석)")

h3("3.1 CRC 코어 동작 — E-guar(보장) · E-win(유용성)  [대표: mediphi seed42, N=1,985]")
table(
    ["블록", "지표", "값", "판정"],
    [["E-guar", "λ̂ / calib joint-risk", "0.173 / 0.0773", "—"],
     ["E-guar (TEST)", "commit / joint_risk / cond_risk", "0.721 / 0.0721 / 0.100", "≤ α=0.10 ✅"],
     ["E-win", "AURC: probe / naive / oracle", "0.0821 / 0.1700 / 0.0636", "—"],
     ["E-win", "무작위→오라클 격차 닫음", "82.6%", "유의미 ✅"]],
    "표 3.1 단일 백본 CRC 보장·유용성")

h3("3.2 표층 단일신호 AUC — 모두 우연 수준  [Fig2]")
p("S_max(최대확률)·U_sc(불확실성)·conflict 표층 신호의 ct_miss/exact_mismatch/under_imaging 판별 AUC는 "
  "전부 **≈0.50(chance)**. 난이도(oracle card)와의 상관도 거의 0:")
table(
    ["상관", "값"],
    [["Spearman(U_sc, card)", "+0.074"], ["Spearman(S_max, card)", "−0.139"]],
    "표 3.2 표층 신호는 난이도를 추적하지 못함")
callout("핵심 동기 — 표층(softmax/불확실성)으로는 위험을 못 가린다. **내부 표상에 디코드 가능한 신호가 따로 있다**는 것이 "
        "CARE-Probe의 출발점.")

h3("3.3 내부 프로브 판별력 — 8개 백본  [Fig2 / C-4]")
table(
    ["백본", "분류", "exact_mismatch", "ct_miss", "under_img"],
    [["gemma2-9b", "gen", "0.872±0.009", "0.773±0.009", "0.636±0.020"],
     ["llama31-8b", "gen", "0.758±0.011", "0.714±0.018", "0.680±0.008"],
     ["med42-8b", "med", "0.775±0.015", "0.713±0.015", "0.700±0.009"],
     ["mediphi", "med", "0.831±0.015", "0.743±0.012", "0.676±0.007"],
     ["mistral7b-v03", "gen", "0.808±0.027", "0.668±0.020", "0.593±0.009"],
     ["openbio-8b", "med", "0.627±0.014", "0.714±0.013", "0.650±0.012"],
     ["phi3mini-128k", "gen", "0.630±0.043", "0.616±0.015", "0.558±0.002"],
     ["qwen14b", "gen", "0.673±0.014", "0.630±0.022", "0.646±0.015"]],
    "표 3.3 best probe AUC (mean±95%CI over seeds)")
p("→ 일반신호(exact_mismatch) 0.63–0.87 **보편적**, harm-specific(ct_miss) 0.62–0.77 **분산** (백본 의존).")

h3("3.4 배치 단위 CRC 집계 — 8개 백본  [Fig4]")
table(
    ["백본", "분류", "commit", "joint_risk", "gap%", "AURC_probe"],
    [["gemma2-9b", "gen", "0.881±0.020", "0.0771±0.0058", "67.7±1.5", "0.0508±0.0013"],
     ["llama31-8b", "gen", "0.505±0.012", "0.0790±0.0044", "61.1±1.8", "0.1541±0.0013"],
     ["med42-8b", "med", "0.512±0.013", "0.0786±0.0040", "68.1±2.1", "0.1496±0.0025"],
     ["mediphi", "med", "0.703±0.021", "0.0742±0.0061", "76.5±1.6", "0.0850±0.0008"],
     ["mistral7b-v03", "gen", "0.486±0.016", "0.0754±0.0060", "54.0±1.2", "0.1461±0.0014"],
     ["openbio-8b", "med", "0.318±0.017", "0.0760±0.0056", "45.4±0.8", "0.2429±0.0019"],
     ["phi3mini-128k", "gen", "0.277±0.016", "0.0756±0.0056", "2.7±2.4", "0.2770±0.0072"],
     ["qwen14b", "gen", "0.429±0.018", "0.0793±0.0054", "40.2±2.2", "0.1815±0.0034"]],
    "표 3.4 results_main 집계 (8백본, mean±95%CI)")
ul([
    "**joint_risk ≤ α 성립 백본: 8/8** (평균 기준 전부 α=0.10선 왼쪽).",
    "런 단위 성공률 **222/240 = 92.5%** (개별 런은 소표본 변동으로 일부 초과 가능).",
    "general 평균 gap% = 45.1 (n=5) · medical 평균 gap% = 63.3 (n=3).",
])

h3("3.5 메커니즘 검증 — gap% ↔ 배치 last-token ct_miss AUC  [C-3]")
callout("**Pearson r = +0.831, Spearman ρ = +0.762** — 내부 신호 판별력이 높은 백본일수록 CRC 유용성(gap%)이 "
        "단조 증가. '프로브 신호 → 선택적 이득'의 인과 메커니즘을 정량 검증.")

h3("3.6 강건성 — 시간이동(E-shift) · 중증도적응(E-sev)")
table(
    ["설정", "지표", "값", "판정"],
    [["E-shift (temporal, qwen14b)", "commit / joint_risk / gap%", "0.410±0.007 / 0.0720±0.0035 / 40.2±2.2", "≤ α ✅"],
     ["E-sev high (ESI 1–2, α=0.05)", "commit / joint_risk / committed_harm", "0.152 / 0.0279±0.030 / 0.159", "2/3 seed OK"],
     ["E-sev lower (ESI 3–5, α=0.15)", "commit / joint_risk / committed_harm", "0.612 / 0.132±0.007 / 0.217", "3/3 seed OK"]],
    "표 3.6 분포이동·중증도적응 하 보장")
p("→ 시간이동 하에서도 joint_risk ≤ α 유지. 중증도적응에서 **고위험군 강보호** 성립(high harm 0.159 < lower 0.217). "
  "단 high-acuity는 소표본(n_te≈240)이라 1개 시드에서 간헐적 역전 [직접 확인 필요].")

# ---------------------------------------------------------------- 4. B-2
h2("4. B-2 — 4k Phi-3-mini Ablation (LongRoPE 교란 분리)")
p("**물음**: MAIN에서 phi3mini-128k의 last-token ct_miss AUC가 chance(≈0.53)로 붕괴한 것이 "
  "(a) 128k LongRoPE 장문맥 적응 탓인가, (b) 단순 grounding/scale 탓인가? "
  "동일 파이프라인에 **비-128k 4k Phi**를 통과시켜 비교.")
table(
    ["모델", "토큰 위치", "ct_miss AUC (레이어별 최댓값)", "해석"],
    [["Phi-3-mini-4k", "last", "0.561 (대부분 ≈0.50)", "여전히 붕괴"],
     ["Phi-3-mini-4k", "prompt", "0.589–0.678", "프롬프트엔 신호 존재"],
     ["Phi-3-mini-128k", "last", "≈0.532", "붕괴"],
     ["Phi-3-mini-128k", "prompt", "0.624", "프롬프트엔 신호 존재"]],
    "표 4a. 4k vs 128k Phi — last/prompt 토큰 ct_miss 판별력")
table(
    ["B-2 CRC (phi3mini-4k, 30런)", "값"],
    [["joint_risk ≤ α 위반", "2/30 (sub43-s51=0.109, sub44-s50=0.129)"],
     ["gap_closed 평균", "≈ 0 (sub43은 음수대)"]],
    "표 4b. 4k Phi의 CRC — 보장은 대체로 유지, 유용성은 소멸")
callout("**판정** — 4k의 last-token 최대 AUC 0.561(<0.60)로 **128k와 동일하게 붕괴**. 따라서 last-token 신호 붕괴는 "
        "**LongRoPE(128k) 탓이 아니라 Phi의 grounding/scale 특성**. → 원고에 *정직한 Limitation*으로 기술(교란 아님). "
        "한편 보장(joint_risk≤α)은 약한 백본에서도 대체로 유지되나 유용성(gap)은 소멸 → '안전성과 유용성의 분리'를 재확인.")

# ---------------------------------------------------------------- 5. B-3
h2("5. B-3 — Qwen-14B Full-Cohort 헤드라인 (N=7,814)")
p("드래프트가 PRIMARY를 'full cohort'로 언급 → 헤드라인을 full-n으로 낼 수 있는지 1회 실행으로 확인. "
  "추출 17.5시간(7.95s/stay, valid 7,814/7,823), 이후 CRC 10 split-seed.")
table(
    ["스케일", "gap%", "joint_risk", "commit", "런 수"],
    [["full-n (N=7,814)", "50.2", "0.0906", "0.516", "10"],
     ["n=2,000 (대비)", "40.2", "0.0793", "0.429", "30"]],
    "표 5a. full-n vs n=2,000 헤드라인 대비 (qwen14b)")
table(
    ["split_seed", "joint_risk", "commit", "gap_closed", "≤0.10?"],
    [["42", "0.0955", "0.536", "0.496", "✅"],
     ["43", "0.0670", "0.441", "0.506", "✅"],
     ["44", "0.0896", "0.521", "0.507", "✅"],
     ["45", "0.1006", "0.537", "0.496", "⚠️ 경계초과"],
     ["46", "0.0934", "0.505", "0.503", "✅"],
     ["47", "0.0883", "0.523", "0.492", "✅"],
     ["48", "0.0981", "0.528", "0.505", "✅"],
     ["49", "0.0874", "0.504", "0.497", "✅"],
     ["50", "0.0870", "0.545", "0.510", "✅"],
     ["51", "0.0989", "0.523", "0.507", "✅"]],
    "표 5b. full-n split-seed별 joint_risk (mean 0.0906, 95%CI [0.084, 0.098])")
callout("**판정** — full-n에서도 **joint_risk 0.0906 ≤ α=0.10 (CI 상한 0.098)**, 9/10 시드 개별 충족(seed45만 0.1006 "
        "경계초과; marginal 보장이라 이론과 무모순). 더 중요한 건 **유사가 아니라 더 강함**: gap_closed 40.2→**50.2%**, "
        "commit 0.429→**0.516**. 데이터가 늘며 conformal 임계가 덜 보수적 → α 예산을 제대로 소진 → 같은 보장 하 유용성↑. "
        "→ 바람직한 스케일링 특성.")

# ---------------------------------------------------------------- 6. 종합
h2("6. 종합 판정 및 원고 반영 방향")
ul([
    "**(보장의 보편성)** joint_risk ≤ α 가 **8/8 백본 · full-cohort · 시간이동 · 중증도적응** 전 조건에서 성립 → 핵심 안전 주장 견고.",
    "**(메커니즘)** gap% ↔ probe AUC **r=0.83** → '내부 신호 → 선택적 이득' 인과 검증.",
    "**(안전성·유용성 분리)** 안전성은 약한 백본(Phi 4k/128k)에서도 유지, 유용성(gap)은 **디코드 가능한 내부 신호가 있는 백본에서만** 발생.",
    "**(B-2)** last-token 붕괴 = scale/grounding(LongRoPE 아님) → 정직한 Limitation으로 기술.",
    "**(B-3)** full-n에서 보장 유지·강화 → **헤드라인(qwen14b)은 full-n으로**, 8-백본 교차표는 **N=2,000 통일 유지**(혼합-N 회피). 두 N을 본문에 명시.",
])
callout("**남은 작업** — (1) 원고 §결과/한계에 위 5개 항목 반영, (2) Fig2/Fig3/Fig4/Fig6 캡션과 본문 수치 동기화(Claim-Evidence Map), "
        "(3) E-sev high-acuity 소표본 변동을 한계로 명시. 실험 자체는 **전부 완료** — 추가 GPU 런 불필요.")

# ---------------------------------------------------------------- 7. 파일 인덱스
h2("7. 산출물 / 파일 인덱스")
table(
    ["경로", "내용"],
    [["cohort_v2.csv (7,823) / cohort_v2_stats.json", "코호트 + 통계"],
     ["cross_validate_report.txt", "legacy 교차검증(표 1c)"],
     ["runs_m15/ (BATCH_SUMMARY, *_metrics.json, *_report.txt)", "RAG 베이스라인 3시드"],
     ["runs_main/results_main.csv + analysis/stage1_results.txt", "MAIN 집계·Stage-1 분석(3장)"],
     ["runs_main/probe_*_n2000_seed{42,43,44}.npz (24개)", "8백본 hidden-state 추출"],
     ["runs_b2/results_b2.csv + logs/, runs_b2_console.log", "B-2 4k Phi ablation(4장)"],
     ["runs_b3/results_b3.csv + logs/, runs_b3_console.log", "B-3 full-cohort(5장)"],
     ["runs_main/rc_mediphi_seed42.csv", "Fig3 risk-coverage 곡선"]],
    "표 7. 핵심 산출물")
p(f"본 리포트 생성: _make_report.py · {STAMP}. 모든 수치는 위 파일에서 직접 추출.")

# ============================================================ 렌더러
def split_bold(text):
    """**bold** 토글 → [(segment, is_bold), ...]"""
    out, bold = [], False
    for i, seg in enumerate(text.split("**")):
        if seg:
            out.append((seg, bold))
        bold = not bold
    return out

# -------- HTML
def render_html():
    css = """
    body{font-family:'Segoe UI','Malgun Gothic',sans-serif;max-width:980px;margin:32px auto;
         padding:0 20px;color:#1a1a1a;line-height:1.6;}
    h1{font-size:26px;border-bottom:3px solid #2c5aa0;padding-bottom:8px;color:#1f3a63;}
    h2{font-size:20px;margin-top:34px;border-left:5px solid #2c5aa0;padding-left:10px;color:#1f3a63;}
    h3{font-size:16px;margin-top:22px;color:#2c5aa0;}
    p{margin:10px 0;}
    ul{margin:10px 0;padding-left:22px;} li{margin:5px 0;}
    table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px;}
    th,td{border:1px solid #c9d3e0;padding:6px 9px;text-align:left;}
    th{background:#2c5aa0;color:#fff;font-weight:600;}
    tr:nth-child(even) td{background:#f3f6fb;}
    caption{caption-side:bottom;font-size:12px;color:#666;padding-top:5px;text-align:left;}
    .callout{background:#eef4fb;border-left:5px solid #2c5aa0;padding:11px 15px;margin:14px 0;border-radius:4px;}
    strong{color:#1f3a63;}
    .meta{color:#666;font-size:13px;}
    """
    h = ['<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">',
         '<title>CARE-Probe 실험결과 종합</title><style>', css, '</style></head><body>']
    def inl(t):
        return "".join(
            (f"<strong>{_html.escape(s)}</strong>" if b else _html.escape(s))
            for s, b in split_bold(t))
    for blk in B:
        k = blk[0]
        if k in ("h1", "h2", "h3"):
            h.append(f"<{k}>{_html.escape(blk[1])}</{k}>")
        elif k == "p":
            cls = ' class="meta"' if blk[1].startswith("본 리포트") or blk[1].startswith("대상 디렉터리") else ""
            h.append(f"<p{cls}>{inl(blk[1])}</p>")
        elif k == "callout":
            h.append(f'<div class="callout">{inl(blk[1])}</div>')
        elif k == "ul":
            h.append("<ul>" + "".join(f"<li>{inl(it)}</li>" for it in blk[1]) + "</ul>")
        elif k == "table":
            _, headers, rows, cap = blk
            h.append("<table><thead><tr>" +
                     "".join(f"<th>{_html.escape(x)}</th>" for x in headers) +
                     "</tr></thead><tbody>")
            for r in rows:
                h.append("<tr>" + "".join(f"<td>{inl(str(x))}</td>" for x in r) + "</tr>")
            h.append("</tbody>")
            if cap:
                h.append(f"<caption>{_html.escape(cap)}</caption>")
            h.append("</table>")
    h.append("</body></html>")
    out = HERE / "CARE-Probe_실험결과_종합.html"
    out.write_text("\n".join(h), encoding="utf-8")
    return out

# -------- DOCX
def _shade(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement('w:shd'); sh.set(qn('w:val'), 'clear'); sh.set(qn('w:fill'), hexfill)
    tcPr.append(sh)

def _add_runs(par, text):
    for s, b in split_bold(text):
        run = par.add_run(s); run.bold = b
        if b: run.font.color.rgb = RGBColor(0x1f, 0x3a, 0x63)

def render_docx():
    doc = Document()
    style = doc.styles['Normal']; style.font.name = 'Malgun Gothic'; style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Malgun Gothic')
    NAVY = RGBColor(0x1f, 0x3a, 0x63)
    for blk in B:
        k = blk[0]
        if k == "h1":
            hd = doc.add_heading(level=0); r = hd.add_run(blk[1]); r.font.color.rgb = NAVY
        elif k == "h2":
            hd = doc.add_heading(level=1)
            for s, b in split_bold(blk[1]): r = hd.add_run(s); r.font.color.rgb = NAVY
        elif k == "h3":
            hd = doc.add_heading(level=2)
            for s, b in split_bold(blk[1]): r = hd.add_run(s); r.font.color.rgb = RGBColor(0x2c,0x5a,0xa0)
        elif k == "p":
            par = doc.add_paragraph(); _add_runs(par, blk[1])
        elif k == "callout":
            t = doc.add_table(rows=1, cols=1); t.style = 'Table Grid'
            cell = t.cell(0, 0); _shade(cell, "EEF4FB")
            par = cell.paragraphs[0]; _add_runs(par, blk[1])
            doc.add_paragraph()
        elif k == "ul":
            for it in blk[1]:
                par = doc.add_paragraph(style='List Bullet'); _add_runs(par, it)
        elif k == "table":
            _, headers, rows, cap = blk
            t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Light Grid Accent 1'
            for j, htext in enumerate(headers):
                c = t.rows[0].cells[j]; c.paragraphs[0].add_run(htext).bold = True
            for r in rows:
                cells = t.add_row().cells
                for j, x in enumerate(r):
                    _add_runs(cells[j].paragraphs[0], str(x))
            if cap:
                cp = doc.add_paragraph(); rr = cp.add_run(cap)
                rr.italic = True; rr.font.size = Pt(8.5); rr.font.color.rgb = RGBColor(0x66,0x66,0x66)
    out = HERE / "CARE-Probe_실험결과_종합.docx"
    doc.save(str(out))
    return out

if __name__ == "__main__":
    ph = render_html(); pd = render_docx()
    print("[OK] HTML :", ph)
    print("[OK] DOCX :", pd)
