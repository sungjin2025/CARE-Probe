# CARE-Probe — Chart-Review Adjudication Rubric (frozen)

*Appendix documenting the appropriateness-standard anchoring of the blinded chart-review adjudication (main-text Section on the harm-proxy adjudication; Supplementary Section S23). Available to editors/reviewers on request. The rubric is journal-independent and was frozen after a ten-case pilot round.*

## Anchoring to a recognised appropriateness standard

The indication judgement (Q1) is anchored to the **ACR Appropriateness Criteria®** framework as the reference standard, applied by a board-certified emergency physician. Case-application conventions (boundary-case handling) were agreed and recorded in a **ten-case pilot round** and frozen before the main pass. The three-point scale is a deliberate collapse of the appropriateness judgement into *indicated / uncertain / not-indicated*; the adjudication does **not** assign per-case ACR variant numbers.

## Blinding (against incorporation bias)

The reader judged from **presentation-time information only** — structured triage record (demographics, vital signs, acuity) and the full history-of-present-illness narrative. Imaging actually ordered, radiology reports, model outputs, and commit/defer status were all **masked**. The adjudicated label therefore cannot be anchored on performed practice (the quantity it checks) or on the system under evaluation.

## Adjudication items

**Q1 — Indication** (each exam under review):

| Code | Definition |
|---|---|
| 1 = Indicated | Clearly justified by this clinical information alone (relevant clinical criteria / usual practice standards). |
| 2 = Uncertain | Discretionary — could reasonably be done or not / insufficient information. |
| 3 = Not indicated | Not justifiable from this information alone (includes suspected defensive/reflexive imaging). |

**Q2 — Harm if omitted** (only for exams judged Q1 = 1):

| Code | Definition | Example |
|---|---|---|
| H = Major | Real impact on morbidity/mortality risk from diagnostic delay or worsening. | Missed head CT in a head-trauma patient on anticoagulation. |
| M = Moderate | Possible care delay / re-visit. | |
| L = Minor | Negligible impact on clinical course. | |

**Q3 — Practice-missed indicated exam** (one item per case): "On the presented clinical information, is there an imaging study that was **not** performed but that you judge would have been indicated?"

| Response | Meaning |
|---|---|
| None | No indicated exam beyond those performed → "performed ⊇ indicated" holds. |
| Present + exam code | Practice missed an indicated exam (e.g. `CT_HEAD`) → under-imaging → record harm grade (H/M/L). |

**NONE-row rule** (`exam_under_review = NONE`, i.e. no imaging performed/read on the encounter): the object judged is the *no-imaging decision*, not a specific exam.

| Item | Entry for a NONE row |
|---|---|
| Q1 | Appropriateness of the no-imaging decision: 1 = no-imaging clearly appropriate (no indicated study) / 2 = borderline / 3 = no-imaging inappropriate (an indicated study existed). |
| Q2 | Always blank (no specific omitted exam to grade). |
| Q3 | If Q1 = 3, must be "present" — record the exam code judged necessary and its harm grade (H/M/L). If Q1 = 1/2, the usual rule applies. |

## Adjudication-sheet schema

| Column | Content | Filled by |
|---|---|---|
| `case_id` | Randomly re-assigned ID (stay_id mapping held by the study team) | pre-filled |
| `case_text` | Clinical text (chief complaint, triage, HPI, history, vitals, exam, age/sex) | pre-filled |
| `exam_under_review` | Exam under review (e.g. `CT_HEAD`); fixed 16-code list (15 exam types + NONE) | pre-filled |
| `q1_indication` | 1 / 2 / 3 | reader |
| `q2_harm_if_omitted` | H / M / L (only when Q1 = 1) | reader |
| `q3_missed_by_practice` | none / exam code(s) (semicolon-separated) | reader |
| `q3_practice_harm` | H / M / L (only when Q3 = present) | reader |
| `note` | Free-text one-line rationale (optional) | reader |
| `time_min` | Time taken, minutes (pilot + first ~20 cases only) | reader |

## Sample design (three arms; blinded arm membership, randomised order)

| Arm | Definition | n |
|---|---|---|
| A. Committed-error | System auto-committed but a performed CT was omitted from the recommendation | ~150 |
| B. Deferred | System routed the case to physician review (random sample) | 50 |
| C. Committed-normal | Auto-committed with no omitted performed CT (random sample) | 50 |

## Reader

Single primary adjudicator: a board-certified emergency physician, not involved in model development, cohort construction, or statistical analysis. An inter-rater double-read subset (Cohen's κ, consensus-resolved) is planned where an independent second reader is available.
