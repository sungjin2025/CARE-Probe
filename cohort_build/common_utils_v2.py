# -*- coding: utf-8 -*-
"""common_utils_v2.py — Multi-label utilities for CR-MSR Policy-A.

Generalizes common_utils.py to V2 multi-label label space + no_imaging mutex.
Backward compatible: mode='single_modality' preserves V1 behavior.

Functions:
  - read_csv_robust(path): encoding-robust CSV reader
  - build_case_text_ed(case_dict): assemble patient case text
  - build_prompt(case, evidence, ...): multi-label LLM prompt with JSON schema
  - build_repair_prompt(bad_text): JSON repair prompt
  - extract_first_json_object(text): parse JSON from LLM output
  - normalize_imaging_json(obj, ...): JSON → label bits dict (multi-label)
  - apply_none_gate(strength, tau_none): pre-gate (S(x) too low)
  - apply_label_gates(bits, strength, tau_dict, ...): per-label post-gate
  - enforce_no_imaging_mutex(bits, ...): mutex check
  - assert_policy_invariants_v2(df, ...): DataFrame-level invariants
"""
from typing import Dict, Any, List, Tuple, Optional
import re, json, io
import pandas as pd

# Reuse label spaces from policyA_decode_v2
try:
    from policyA_decode_v2 import LABEL_SPACE_V2, LABEL_SPACE_V1, NO_IMAGING_LABEL
except ImportError:
    LABEL_SPACE_V2 = [
        "CXR_CHEST", "CT_ABD_PELVIS", "CT_HEAD", "XR_EXTREMITY", "US_ABDOMEN",
        "CT_CSPINE", "CT_CHEST", "XR_ABDOMEN", "XR_PELVIS", "US_EXTREMITY",
        "CT_EXTREMITY", "US_PELVIS", "OTHER_OTHER", "CT_PELVIS", "MRI_CSPINE",
    ]
    NO_IMAGING_LABEL = "no_imaging"
    LABEL_SPACE_V1 = ["ct_head", "ct_abd", "cxr", "us"]


DEFAULT_INPUT_COLS = [
    "chiefcomplaint", "chief_complaint",
    "triage_note", "hpi", "history_present_illness",
    "pmh", "past_medical_history",
    "medications", "allergies",
    "vitals", "physical_exam",
    "assessment", "plan", "diagnosis",
]


# ---------- Small helpers ----------
def _clip01(x):
    try: x = float(x)
    except: return 0.0
    return max(0.0, min(1.0, x))


def _truthy(x):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return float(x) != 0.0
    if isinstance(x, str): return x.strip().lower() in ("1", "true", "yes", "y", "abstain", "defer")
    return False


def _append_error(prev, msg):
    prev = prev or ""
    msg = (msg or "").strip()
    if not msg: return prev
    return f"{prev} | {msg}" if prev else msg


def _json_dump_compact(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except:
        try: return json.dumps(str(obj), ensure_ascii=False)
        except: return "{}"


# ---------- CSV IO (robust) ----------
def read_csv_robust(path):
    """Encoding-robust CSV reader. Tries utf-8, utf-8-sig, cp1252, cp949, euc-kr, latin1."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        pass
    except Exception:
        pass
    for enc in ["utf-8", "utf-8-sig", "cp1252", "cp949", "euc-kr", "latin1"]:
        for eng in ["c", "python"]:
            try:
                return pd.read_csv(path, encoding=enc, engine=eng, low_memory=False)
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
    return pd.read_csv(path, encoding="utf-8", engine="python", encoding_errors="replace")


# ---------- Case text ----------
def build_case_text_ed(case: Dict[str, Any]) -> str:
    if not isinstance(case, dict) or not case: return ""
    parts = []
    for k, v in case.items():
        if v is None: continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none"): continue
        kk = str(k).strip()
        if not kk: continue
        parts.append(f"{kk}: {s}")
    return "\n".join(parts).strip()


# ---------- Prompt builders ----------
def build_prompt(case_text: str,
                 evidence_text: str = "",
                 label_space: Optional[List[str]] = None,
                 include_no_imaging: bool = True,
                 allow_abstain: bool = True,
                 mode: str = "multi_label") -> str:
    """Build LLM prompt with JSON schema for multi-label imaging decision.

    mode='multi_label' (default): schema includes LABEL_SPACE_V2 + no_imaging mutex rule.
    mode='single_modality':       schema includes LABEL_SPACE_V1 (legacy 4-bit).
    """
    if label_space is None:
        label_space = LABEL_SPACE_V2 if mode == "multi_label" else LABEL_SPACE_V1
    case_text = (case_text or "").strip()
    evidence_text = (evidence_text or "").strip()

    header = (
        "TASK: De-identified retrospective research labeling.\n"
        "This is NOT medical advice and NOT for real-time clinical use.\n"
        "Return ONLY one JSON object and nothing else.\n"
    )

    labels_in_schema = list(label_space)
    if include_no_imaging and mode == "multi_label":
        labels_in_schema = labels_in_schema + [NO_IMAGING_LABEL]
    json_keys = ", ".join(f'"{L}":0|1' for L in labels_in_schema)

    schema = (
        f"Return JSON with exactly these integer keys (0 or 1 only):\n"
        f"{{{json_keys}}}\n"
        f"Rules:\n"
        f"- Output ONLY one JSON object.\n"
        f"- Use only integer 0/1 values.\n"
    )
    if mode == "multi_label":
        schema += (
            f"- You MAY select multiple imaging labels if multiple modalities are indicated.\n"
            f"- If no imaging is needed, set 'no_imaging' to 1 and ALL imaging labels to 0.\n"
            f"- 'no_imaging' is MUTUALLY EXCLUSIVE with any imaging label.\n"
        )
    if allow_abstain:
        schema += (
            f"- If uncertain or insufficient information, you may abstain using:\n"
            f"  {{\"abstain\": true, \"reason\": \"insufficient_evidence\"}}\n"
        )

    blocks = [header, schema]
    if evidence_text:
        blocks.append(f"EVIDENCE:\n{evidence_text}")
    blocks.append(f"CASE:\n{case_text}")
    blocks.append("JSON:")
    return "\n\n".join(b for b in blocks if b).strip()


def build_repair_prompt(bad_text):
    return (
        "Fix the output to be a single valid JSON object.\n"
        "Return ONLY the JSON object and nothing else.\n\n"
        f"Bad output:\n{(bad_text or '').strip()}\n\nJSON:"
    )


# ---------- JSON extraction ----------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def strip_json_fence(s):
    s = (s or "").strip()
    if not s: return s
    m = _JSON_FENCE_RE.search(s)
    return (m.group(1) or "").strip() if m else s


def extract_first_json_object(text):
    """Extract first JSON object {...} from string. Returns dict or None."""
    if not isinstance(text, str): return None
    t = strip_json_fence(text.strip())
    if not t: return None
    try:
        obj = json.loads(t)
        if isinstance(obj, dict): return obj
    except Exception:
        pass
    start = t.find("{")
    if start < 0: return None
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start:i+1])
                    if isinstance(obj, dict): return obj
                except Exception:
                    return None
    return None


# ---------- JSON normalization (multi-label) ----------
_DEFAULT_V1_ALIASES = {
    "ct_head": "ct_head",
    "ct_abd": "ct_abd",
    "ct_abdomen": "ct_abd",
    "ct_abdominal": "ct_abd",
    "cxr": "cxr",
    "chest_xray": "cxr",
    "chestxray": "cxr",
    "xray": "cxr",
    "us": "us",
    "ultrasound": "us",
}


def normalize_imaging_json(obj: Any,
                           label_space: Optional[List[str]] = None,
                           include_no_imaging: bool = True,
                           mode: str = "multi_label",
                           key_aliases: Optional[Dict[str, str]] = None,
                          ) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Normalize model JSON → label bits dict.

    Returns: (pred_bits, meta)
      pred_bits: dict label → 0/1 (includes all label_space + optional no_imaging)
      meta: {is_abstain_json, abstain_reason, unknown_keys (optional)}
    """
    if label_space is None:
        label_space = LABEL_SPACE_V2 if mode == "multi_label" else LABEL_SPACE_V1
    if mode == "single_modality" and key_aliases is None:
        key_aliases = _DEFAULT_V1_ALIASES

    labels_to_emit = list(label_space)
    if include_no_imaging and mode == "multi_label":
        labels_to_emit = labels_to_emit + [NO_IMAGING_LABEL]

    pred = {L: 0 for L in labels_to_emit}
    meta: Dict[str, Any] = {"is_abstain_json": False, "abstain_reason": ""}

    if not isinstance(obj, dict) or not obj:
        return pred, meta

    # Detect abstain schema
    for k in ("abstain", "defer", "abstention"):
        if k in obj and _truthy(obj.get(k)):
            reason = (obj.get("reason", "") or obj.get("abstain_reason", "")
                      or obj.get("why", "") or "")
            try: reason = str(reason)[:120]
            except: reason = ""
            meta["is_abstain_json"] = True
            meta["abstain_reason"] = reason.strip() or "llm_abstain_json"
            return pred, meta

    # Case-insensitive canonical key map
    canon_upper = {L.upper(): L for L in labels_to_emit}
    alias_upper = {k.upper(): v for k, v in (key_aliases or {}).items()}

    unknown = []
    for k, v in obj.items():
        kk = str(k).strip().upper()
        if kk in ("ABSTAIN", "DEFER", "ABSTENTION", "REASON", "ABSTAIN_REASON", "WHY"):
            continue
        target = canon_upper.get(kk)
        if target is None:
            target = alias_upper.get(kk)
            if target and target not in pred:
                target = None
        if target:
            pred[target] = 1 if _truthy(v) else 0
        else:
            unknown.append(str(k))

    if unknown:
        meta["unknown_keys"] = unknown[:50]

    for k in list(pred.keys()):
        pred[k] = 1 if int(pred[k]) == 1 else 0

    return pred, meta


# ---------- Gating utilities ----------
def apply_none_gate(strength, tau_none) -> Tuple[bool, str]:
    """Pre-gate: if S(x) < tau_none, abstain without LLM call. Returns (gated?, reason)."""
    try:
        s = float(strength); t = float(tau_none)
    except Exception:
        return False, ""
    if s < t:
        return True, "none_gate"
    return False, ""


def apply_label_gates(pred_bits: Dict[str, int],
                      strength: float,
                      tau_dict: Optional[Dict[str, float]] = None,
                      label_space: Optional[List[str]] = None,
                     ) -> Tuple[Dict[str, int], List[str]]:
    """Post-gate per label: if strength < tau_dict[label], zero that label bit.

    Replaces V1's apply_modality_gates (which had hardcoded τ_CT/τ_CXR/τ_US).
    Per-label τ allows heterogeneous thresholds (e.g., CT_HEAD vs CT_CHEST).
    When CR-MSR's conformal layer runs, tau_dict will be CALIBRATED (not heuristic).

    Returns: (gated_bits, reasons_list)
    """
    if label_space is None:
        label_space = [L for L in (pred_bits or {}).keys() if L != NO_IMAGING_LABEL]
    out = dict(pred_bits or {})
    for L in label_space:
        out[L] = 1 if int(out.get(L, 0)) == 1 else 0
    reasons: List[str] = []
    if tau_dict is None or strength is None:
        return out, reasons
    try:
        s = float(strength)
    except Exception:
        return out, reasons
    for L in label_space:
        if L not in tau_dict: continue
        try:
            t = float(tau_dict[L])
        except Exception:
            continue
        if out[L] == 1 and s < t:
            out[L] = 0
            reasons.append(f"gate_{L}")
    return out, reasons


def enforce_no_imaging_mutex(pred_bits: Dict[str, int],
                             label_space: Optional[List[str]] = None,
                            ) -> Tuple[Dict[str, int], List[str]]:
    """Check no_imaging mutex. Returns (bits, reasons).
    If violated (no_imaging=1 AND any imaging=1), adds 'mutex_violation' reason.
    Does NOT auto-resolve (decoder/policy layer decides what to do)."""
    out = dict(pred_bits or {})
    no_img = 1 if int(out.get(NO_IMAGING_LABEL, 0)) == 1 else 0
    if label_space is None:
        label_space = [L for L in out.keys() if L != NO_IMAGING_LABEL]
    imaging_sum = sum(1 if int(out.get(L, 0)) == 1 else 0 for L in label_space)
    reasons: List[str] = []
    if no_img == 1 and imaging_sum >= 1:
        reasons.append("mutex_violation")
    return out, reasons


# ---------- Policy invariants (V2) ----------
def assert_policy_invariants_v2(df: pd.DataFrame,
                                label_space: Optional[List[str]] = None,
                               ) -> bool:
    """V2 multi-label policy invariants. Raises AssertionError on violation.
    Expects columns: stay_id, decision, is_commit_imaging, is_commit_no_imaging,
                     is_abstain, abstain_reason.
    Optional: pred_y__{L} for each L in label_space + pred_y__no_imaging
    """
    if df is None or len(df) == 0:
        raise AssertionError("empty df")
    if label_space is None:
        label_space = LABEL_SPACE_V2

    required = ["stay_id", "decision", "is_commit_imaging",
                "is_commit_no_imaging", "is_abstain", "abstain_reason"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"missing columns: {missing}")

    # decision ∈ {commit_imaging, commit_no_imaging, abstain}
    valid_decisions = {"commit_imaging", "commit_no_imaging", "abstain"}
    bad = ~df["decision"].astype(str).isin(valid_decisions)
    if bad.any():
        raise AssertionError(
            f"invalid decision values: {df.loc[bad, 'decision'].unique()}")

    # Exactly one of three booleans True per row
    flags = df[["is_commit_imaging", "is_commit_no_imaging", "is_abstain"]].fillna(0).astype(int).sum(axis=1)
    if (flags != 1).any():
        raise AssertionError(
            f"flags must sum to 1; violated {(flags != 1).sum()} rows")

    # abstain ⇒ reason non-empty
    abst = df["is_abstain"].fillna(0).astype(int) == 1
    if abst.any():
        ar = df.loc[abst, "abstain_reason"].fillna("").astype(str)
        if (ar.str.len() == 0).any():
            raise AssertionError(
                f"abstain without reason: {(ar.str.len() == 0).sum()} rows")

    # commit_imaging ⇒ at least one pred_y__{L} = 1 (if columns present)
    commit_img = df["is_commit_imaging"].fillna(0).astype(int) == 1
    label_cols = [f"pred_y__{L}" for L in label_space if f"pred_y__{L}" in df.columns]
    if commit_img.any() and label_cols:
        bits_sum = df.loc[commit_img, label_cols].fillna(0).astype(int).sum(axis=1)
        if (bits_sum < 1).any():
            raise AssertionError(
                f"commit_imaging with no label bits set: {(bits_sum < 1).sum()} rows")
        # mutex: commit_imaging ⇒ no_imaging=0
        if "pred_y__no_imaging" in df.columns:
            if (df.loc[commit_img, "pred_y__no_imaging"].fillna(0).astype(int) == 1).any():
                raise AssertionError("commit_imaging with no_imaging=1 (mutex violation)")

    # commit_no_imaging ⇒ no_imaging=1 + imaging bits=0
    commit_no = df["is_commit_no_imaging"].fillna(0).astype(int) == 1
    if commit_no.any() and "pred_y__no_imaging" in df.columns:
        if (df.loc[commit_no, "pred_y__no_imaging"].fillna(0).astype(int) != 1).any():
            raise AssertionError("commit_no_imaging without no_imaging=1")
        if label_cols:
            bits_sum = df.loc[commit_no, label_cols].fillna(0).astype(int).sum(axis=1)
            if (bits_sum > 0).any():
                raise AssertionError(
                    f"commit_no_imaging with imaging bits set: {(bits_sum > 0).sum()} rows")

    return True


# ---------- Tests ----------
def _test():
    passed = failed = 0
    def check(name, actual, expected):
        nonlocal passed, failed
        if actual == expected:
            passed += 1
        else:
            failed += 1
            print(f"FAIL: {name}")
            print(f"  expected: {expected!r}")
            print(f"  actual:   {actual!r}")

    # build_prompt
    p = build_prompt("Chief complaint: chest pain", "")
    check("prompt contains CXR_CHEST", "CXR_CHEST" in p, True)
    check("prompt contains no_imaging", "no_imaging" in p, True)
    check("prompt contains MUTUALLY EXCLUSIVE", "MUTUALLY EXCLUSIVE" in p, True)
    check("prompt contains abstain option", "abstain" in p.lower(), True)

    p_v1 = build_prompt("CC: cp", "", mode="single_modality")
    check("v1 prompt has ct_head", "ct_head" in p_v1, True)
    check("v1 prompt no CXR_CHEST", "CXR_CHEST" in p_v1, False)
    check("v1 prompt no MUTUALLY EXCLUSIVE", "MUTUALLY EXCLUSIVE" in p_v1, False)

    p_no_abstain = build_prompt("CC: x", "", allow_abstain=False)
    check("prompt without abstain option", "{\"abstain\"" in p_no_abstain, False)

    # extract_first_json_object
    obj = extract_first_json_object('{"CXR_CHEST": 1, "CT_HEAD": 0}')
    check("extract: dict returned", isinstance(obj, dict), True)
    check("extract: CXR_CHEST=1", obj.get("CXR_CHEST"), 1)

    obj = extract_first_json_object('```json\n{"CXR_CHEST": 1}\n```')
    check("extract: fenced json", obj.get("CXR_CHEST") if obj else None, 1)

    obj = extract_first_json_object("noise {\"CXR_CHEST\": 1} more")
    check("extract: embedded json", obj.get("CXR_CHEST") if obj else None, 1)

    check("extract: None for non-json", extract_first_json_object("no json here"), None)

    # normalize_imaging_json
    pred, meta = normalize_imaging_json({"CXR_CHEST": 1, "CT_HEAD": 1, "no_imaging": 0})
    check("normalize: CXR_CHEST=1", pred["CXR_CHEST"], 1)
    check("normalize: CT_HEAD=1", pred["CT_HEAD"], 1)
    check("normalize: no_imaging=0", pred["no_imaging"], 0)
    check("normalize: not abstain", meta["is_abstain_json"], False)

    pred, meta = normalize_imaging_json({"abstain": True, "reason": "insufficient"})
    check("normalize abstain detected", meta["is_abstain_json"], True)
    check("normalize abstain reason", meta["abstain_reason"], "insufficient")

    pred, meta = normalize_imaging_json({"cxr_chest": 1})  # case insensitive
    check("normalize case insensitive", pred["CXR_CHEST"], 1)

    pred, meta = normalize_imaging_json({"CXR_CHEST": 1, "WHATEVER": 1})
    check("normalize unknown keys tracked", "WHATEVER" in meta.get("unknown_keys", []), True)

    pred, meta = normalize_imaging_json({"cxr": 1, "ct_head": 0}, mode="single_modality")
    check("v1 normalize: cxr=1", pred["cxr"], 1)
    check("v1 normalize: no no_imaging key", "no_imaging" in pred, False)

    # V1 alias mapping
    pred, meta = normalize_imaging_json({"ct_abdomen": 1}, mode="single_modality")
    check("v1 alias ct_abdomen → ct_abd", pred["ct_abd"], 1)

    # apply_none_gate
    g, r = apply_none_gate(0.3, 0.5)
    check("none_gate triggered", (g, r), (True, "none_gate"))
    g, r = apply_none_gate(0.7, 0.5)
    check("none_gate not triggered", (g, r), (False, ""))
    g, r = apply_none_gate("bogus", 0.5)
    check("none_gate handles bad strength", (g, r), (False, ""))

    # apply_label_gates
    bits = {"CXR_CHEST": 1, "CT_HEAD": 1, "US_ABDOMEN": 1}
    tau_dict = {"CXR_CHEST": 0.3, "CT_HEAD": 0.7, "US_ABDOMEN": 0.5}
    gated, reasons = apply_label_gates(bits, 0.4, tau_dict)
    check("label_gates: CXR kept (0.4>=0.3)", gated["CXR_CHEST"], 1)
    check("label_gates: CT zeroed (0.4<0.7)", gated["CT_HEAD"], 0)
    check("label_gates: US zeroed (0.4<0.5)", gated["US_ABDOMEN"], 0)
    check("label_gates: gate_CT_HEAD reason", "gate_CT_HEAD" in reasons, True)
    check("label_gates: gate_US_ABDOMEN reason", "gate_US_ABDOMEN" in reasons, True)

    gated, reasons = apply_label_gates(bits, 0.4, None)
    check("label_gates: no tau = no gating", gated["CXR_CHEST"], 1)
    check("label_gates: no tau = empty reasons", reasons, [])

    # enforce_no_imaging_mutex
    out, reasons = enforce_no_imaging_mutex({"CXR_CHEST": 1, "no_imaging": 1},
                                            label_space=["CXR_CHEST"])
    check("mutex: violation detected", "mutex_violation" in reasons, True)

    out, reasons = enforce_no_imaging_mutex({"CXR_CHEST": 1, "no_imaging": 0},
                                            label_space=["CXR_CHEST"])
    check("mutex: no violation", reasons, [])

    out, reasons = enforce_no_imaging_mutex({"CXR_CHEST": 0, "no_imaging": 1},
                                            label_space=["CXR_CHEST"])
    check("mutex: no_imaging alone ok", reasons, [])

    # assert_policy_invariants_v2 — happy path
    df_ok = pd.DataFrame([
        {"stay_id": 1, "decision": "commit_imaging",
         "is_commit_imaging": 1, "is_commit_no_imaging": 0, "is_abstain": 0,
         "abstain_reason": "", "pred_y__CXR_CHEST": 1, "pred_y__no_imaging": 0},
        {"stay_id": 2, "decision": "commit_no_imaging",
         "is_commit_imaging": 0, "is_commit_no_imaging": 1, "is_abstain": 0,
         "abstain_reason": "", "pred_y__CXR_CHEST": 0, "pred_y__no_imaging": 1},
        {"stay_id": 3, "decision": "abstain",
         "is_commit_imaging": 0, "is_commit_no_imaging": 0, "is_abstain": 1,
         "abstain_reason": "ambiguous"},
    ])
    check("invariants happy path", assert_policy_invariants_v2(df_ok, label_space=["CXR_CHEST"]), True)

    # Violation: abstain without reason
    df_bad = df_ok.copy()
    df_bad.loc[df_bad["stay_id"] == 3, "abstain_reason"] = ""
    raised = False
    try: assert_policy_invariants_v2(df_bad, label_space=["CXR_CHEST"])
    except AssertionError: raised = True
    check("invariants: abstain without reason raises", raised, True)

    # Violation: commit_imaging without label bits
    df_bad2 = df_ok.copy()
    df_bad2.loc[df_bad2["stay_id"] == 1, "pred_y__CXR_CHEST"] = 0
    raised = False
    try: assert_policy_invariants_v2(df_bad2, label_space=["CXR_CHEST"])
    except AssertionError: raised = True
    check("invariants: commit_imaging w/o bits raises", raised, True)

    # Violation: invalid decision
    df_bad3 = df_ok.copy()
    df_bad3.loc[df_bad3["stay_id"] == 1, "decision"] = "weird"
    raised = False
    try: assert_policy_invariants_v2(df_bad3, label_space=["CXR_CHEST"])
    except AssertionError: raised = True
    check("invariants: invalid decision raises", raised, True)

    # Summary
    total = passed + failed
    print(f"\n=== Tests: {passed}/{total} passed ===")
    if failed > 0:
        print(f"[FAIL] {failed} tests failed")
        return False
    print("[OK] All tests passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if _test() else 1)
