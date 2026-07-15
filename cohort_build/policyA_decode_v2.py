# -*- coding: utf-8 -*-
"""policyA_decode_v2.py — Multi-label decoder for CR-MSR Policy-A.

Generalizes the original single-modality decoder (bits {ct_head, ct_abd, cxr, us}) to:
- Multi-label imaging set commit (>=1 imaging labels)
- Explicit no_imaging mutex (no_imaging=1 ↔ all imaging labels=0)
- 3-way decision: commit_imaging / commit_no_imaging / abstain

Backward compatible: mode='single_modality' preserves original V1 behavior.
"""
from typing import Dict, Any, List, Optional

# ---------- Label spaces ----------
# V2: CR-MSR multi-label = top-15 (modality, body_part) tuples from M1.2d cohort
LABEL_SPACE_V2: List[str] = [
    "CXR_CHEST", "CT_ABD_PELVIS", "CT_HEAD", "XR_EXTREMITY", "US_ABDOMEN",
    "CT_CSPINE", "CT_CHEST", "XR_ABDOMEN", "XR_PELVIS", "US_EXTREMITY",
    "CT_EXTREMITY", "US_PELVIS", "OTHER_OTHER", "CT_PELVIS", "MRI_CSPINE",
]
NO_IMAGING_LABEL = "no_imaging"

# V1: legacy single-modality (backward compat with original policyA_decode.py)
LABEL_SPACE_V1: List[str] = ["ct_head", "ct_abd", "cxr", "us"]

# ---------- Allowed abstain reasons ----------
_ALLOWED_REASONS = {
    "none_gate", "refusal", "llm_failed", "parse_fail", "repair_fail",
    "invalid_keys", "all_zero_after_gate", "multi_pred",
    "llm_abstain_json", "insufficient_evidence",
    "mutex_violation",   # V2 NEW: no_imaging=1 AND imaging label=1
    "ambiguous",         # V2 NEW: all bits zero (no imaging AND no no_imaging)
}


def _norm01(x: Any) -> int:
    try:
        return 1 if int(x) == 1 else 0
    except Exception:
        return 0


def _normalize_reason(reason: str) -> str:
    r = (reason or "").strip()
    if not r:
        return ""
    return r if r in _ALLOWED_REASONS else "parse_fail"


def decode_policyA_from_bits(
    pred_bits: Dict[str, Any],
    abstain_reason: str = "",
    label_space: Optional[List[str]] = None,
    mode: str = "multi_label",
) -> Dict[str, Any]:
    """
    Decode LLM bit predictions to Policy-A 3-way decision.

    Args:
      pred_bits: dict label -> 0/1
        - multi_label mode: keys in LABEL_SPACE_V2 + 'no_imaging'
        - single_modality mode: ct_head/ct_abd/cxr/us
      abstain_reason: if non-empty, force abstain (normalized to _ALLOWED_REASONS or 'parse_fail')
      label_space: imaging label keys (default LABEL_SPACE_V2 / V1 per mode)
      mode: 'multi_label' (default) or 'single_modality' (legacy)

    Returns dict (see _decode_v2 / _decode_v1 for fields).
    """
    if mode not in ("multi_label", "single_modality"):
        raise ValueError(f"mode must be 'multi_label' or 'single_modality', got {mode!r}")
    if label_space is None:
        label_space = LABEL_SPACE_V2 if mode == "multi_label" else LABEL_SPACE_V1
    if mode == "single_modality":
        return _decode_v1(pred_bits or {}, abstain_reason, label_space)
    return _decode_v2(pred_bits or {}, abstain_reason, label_space)


# ---------- V2 multi-label decoder ----------
def _decode_v2(pred_bits, abstain_reason, label_space):
    bits = {L: _norm01(pred_bits.get(L, 0)) for L in label_space}
    no_imaging = _norm01(pred_bits.get(NO_IMAGING_LABEL, 0))
    bits[NO_IMAGING_LABEL] = no_imaging

    ar = _normalize_reason(abstain_reason)
    if ar:
        return _abstain_v2(ar, bits)

    imaging_set = sorted([L for L in label_space if bits[L] == 1])
    imaging_sum = len(imaging_set)

    # Decision tree
    if no_imaging == 1 and imaging_sum >= 1:
        return _abstain_v2("mutex_violation", bits)
    if no_imaging == 0 and imaging_sum == 0:
        return _abstain_v2("ambiguous", bits)
    if no_imaging == 1:
        return {
            "pred_bits": bits,
            "pred_label_set": [],
            "pred_no_imaging": 1,
            "decision": "commit_no_imaging",
            "is_commit_imaging": False,
            "is_commit_no_imaging": True,
            "is_abstain": False,
            "abstain_reason": "",
            "pred_label": None,
            "pred_imaging_choice_7": None,
            "pred_imaging_choice_4": None,
            "is_multi_pred": False,
            "is_ambiguous_pred": False,
        }
    # commit_imaging (imaging_sum >= 1, no_imaging == 0)
    return {
        "pred_bits": bits,
        "pred_label_set": imaging_set,
        "pred_no_imaging": 0,
        "decision": "commit_imaging",
        "is_commit_imaging": True,
        "is_commit_no_imaging": False,
        "is_abstain": False,
        "abstain_reason": "",
        "pred_label": None,
        "pred_imaging_choice_7": None,
        "pred_imaging_choice_4": None,
        "is_multi_pred": (imaging_sum >= 2),  # informational; NOT an abstain trigger
        "is_ambiguous_pred": False,
    }


def _abstain_v2(reason, bits):
    return {
        "pred_bits": bits,
        "pred_label_set": [],
        "pred_no_imaging": 0,
        "decision": "abstain",
        "is_commit_imaging": False,
        "is_commit_no_imaging": False,
        "is_abstain": True,
        "abstain_reason": reason,
        "pred_label": None,
        "pred_imaging_choice_7": None,
        "pred_imaging_choice_4": None,
        "is_multi_pred": False,
        "is_ambiguous_pred": (reason == "ambiguous"),
    }


# ---------- V1 single-modality decoder (backward compat) ----------
def _decode_v1(pred_bits, abstain_reason, label_space):
    bits = {
        "ct_head": _norm01(pred_bits.get("ct_head", 0)),
        "ct_abd":  _norm01(pred_bits.get("ct_abd",  0)),
        "cxr":     _norm01(pred_bits.get("cxr",     0)),
        "us":      _norm01(pred_bits.get("us",      0)),
    }
    ar = _normalize_reason(abstain_reason)
    s = sum(bits.values())

    def _abstain_v1(reason, ambiguous=False, multi=False):
        return {
            "pred_bits": bits,
            "pred_label_set": [],
            "pred_no_imaging": 0,
            "decision": "abstain",
            "is_commit_imaging": False,
            "is_commit_no_imaging": False,
            "is_abstain": True,
            "abstain_reason": reason,
            "pred_label": "abstain",
            "pred_imaging_choice_7": None,
            "pred_imaging_choice_4": None,
            "is_multi_pred": multi,
            "is_ambiguous_pred": ambiguous,
        }

    if ar:
        return _abstain_v1(ar)
    if s == 0:
        return _abstain_v1("all_zero_after_gate")
    if s >= 2:
        return _abstain_v1("multi_pred", ambiguous=True, multi=True)

    # s == 1 — commit (legacy choice4 mapping: ct_head/ct_abd → ct)
    if bits["cxr"] == 1:
        choice7, choice4 = "cxr", "cxr"
    elif bits["us"] == 1:
        choice7, choice4 = "us", "us"
    elif bits["ct_head"] == 1:
        choice7, choice4 = "ct_head", "ct"
    else:
        choice7, choice4 = "ct_abd", "ct"
    return {
        "pred_bits": bits,
        "pred_label_set": [choice7],
        "pred_no_imaging": 0,
        "decision": "commit_imaging",
        "is_commit_imaging": True,
        "is_commit_no_imaging": False,
        "is_abstain": False,
        "abstain_reason": "",
        "pred_label": choice4,
        "pred_imaging_choice_7": choice7,
        "pred_imaging_choice_4": choice4,
        "is_multi_pred": False,
        "is_ambiguous_pred": False,
    }


# ---------- Unit tests ----------
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

    # ===== V2 multi-label tests =====
    r = decode_policyA_from_bits({"CXR_CHEST": 1})
    check("v2/single-label/decision", r["decision"], "commit_imaging")
    check("v2/single-label/set", r["pred_label_set"], ["CXR_CHEST"])
    check("v2/single-label/is_commit_imaging", r["is_commit_imaging"], True)
    check("v2/single-label/is_multi_pred", r["is_multi_pred"], False)

    r = decode_policyA_from_bits({"CXR_CHEST": 1, "CT_HEAD": 1, "US_ABDOMEN": 1})
    check("v2/multi-label/decision", r["decision"], "commit_imaging")
    check("v2/multi-label/set", set(r["pred_label_set"]),
          {"CXR_CHEST", "CT_HEAD", "US_ABDOMEN"})
    check("v2/multi-label/is_multi_pred", r["is_multi_pred"], True)
    check("v2/multi-label/no_imaging", r["pred_no_imaging"], 0)

    r = decode_policyA_from_bits({"no_imaging": 1})
    check("v2/no_imaging/decision", r["decision"], "commit_no_imaging")
    check("v2/no_imaging/is_commit_no_imaging", r["is_commit_no_imaging"], True)
    check("v2/no_imaging/set empty", r["pred_label_set"], [])
    check("v2/no_imaging/no_imaging bit", r["pred_no_imaging"], 1)

    r = decode_policyA_from_bits({"no_imaging": 1, "CXR_CHEST": 1})
    check("v2/mutex_violation/decision", r["decision"], "abstain")
    check("v2/mutex_violation/reason", r["abstain_reason"], "mutex_violation")

    r = decode_policyA_from_bits({})
    check("v2/all-zero/decision", r["decision"], "abstain")
    check("v2/all-zero/reason", r["abstain_reason"], "ambiguous")
    check("v2/all-zero/is_ambiguous_pred", r["is_ambiguous_pred"], True)

    r = decode_policyA_from_bits({"CXR_CHEST": 1}, abstain_reason="refusal")
    check("v2/forced-abstain/decision", r["decision"], "abstain")
    check("v2/forced-abstain/reason", r["abstain_reason"], "refusal")

    r = decode_policyA_from_bits({}, abstain_reason="bogus_reason")
    check("v2/invalid-reason normalized to parse_fail", r["abstain_reason"], "parse_fail")

    # ===== V1 backward-compat tests =====
    r = decode_policyA_from_bits({"cxr": 1}, mode="single_modality")
    check("v1/cxr/decision", r["decision"], "commit_imaging")
    check("v1/cxr/pred_label", r["pred_label"], "cxr")
    check("v1/cxr/pred_imaging_choice_4", r["pred_imaging_choice_4"], "cxr")
    check("v1/cxr/pred_imaging_choice_7", r["pred_imaging_choice_7"], "cxr")

    r = decode_policyA_from_bits({"ct_head": 1}, mode="single_modality")
    check("v1/ct_head/pred_label", r["pred_label"], "ct")
    check("v1/ct_head/pred_imaging_choice_7", r["pred_imaging_choice_7"], "ct_head")

    r = decode_policyA_from_bits({"ct_abd": 1}, mode="single_modality")
    check("v1/ct_abd/pred_label", r["pred_label"], "ct")

    r = decode_policyA_from_bits({"cxr": 1, "ct_head": 1}, mode="single_modality")
    check("v1/multi → abstain (legacy)", r["decision"], "abstain")
    check("v1/multi/reason", r["abstain_reason"], "multi_pred")
    check("v1/multi/is_multi_pred", r["is_multi_pred"], True)

    r = decode_policyA_from_bits({}, mode="single_modality")
    check("v1/all-zero/decision", r["decision"], "abstain")
    check("v1/all-zero/reason", r["abstain_reason"], "all_zero_after_gate")

    r = decode_policyA_from_bits({"cxr": 1}, abstain_reason="none_gate", mode="single_modality")
    check("v1/forced-abstain pre-gate", r["abstain_reason"], "none_gate")

    # ===== Type robustness =====
    r = decode_policyA_from_bits({"CXR_CHEST": "1"})
    check("v2/string '1' treated as 1", r["pred_label_set"], ["CXR_CHEST"])

    r = decode_policyA_from_bits({"CXR_CHEST": True, "CT_HEAD": False})
    check("v2/bool True treated as 1", r["pred_label_set"], ["CXR_CHEST"])

    r = decode_policyA_from_bits({"CXR_CHEST": "bogus"})
    check("v2/bogus value treated as 0 → ambiguous", r["decision"], "abstain")

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
