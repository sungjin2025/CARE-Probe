# -*- coding: utf-8 -*-
"""M1.2b: deterministic exam_name → (modality, body_part) mapping.
v2: VEINS/FETAL/NUCHAL/CAROTID SERIES → US; SKELETAL/STAND FEET/SCOLIOSIS → XR."""
import json, re, csv
from collections import Counter
import paths

# Label space (user-confirmed 2026-05-29):
#   modality:  CT, CXR, XR, US, MRI, MAMMO(excluded later), OTHER
#   body_part: HEAD, NECK, CHEST, ABDOMEN, PELVIS, ABD_PELVIS, CSPINE, TSPINE, LSPINE, EXTREMITY, BREAST, OTHER

def norm(s):
    return (s or "").upper().strip()

# --- MODALITY rules (priority order) ---
def get_modality(name):
    n = norm(name)
    if not n or n == "___":
        return "OTHER"
    if re.search(r"MAMMO|TOMO,CVIEW|^DIG SCREENING|H DIGITAL SCREEN", n):
        return "MAMMO"
    if re.search(r"\bMR\b|\bMRI\b|\bMRA\b|\bMRCP\b|\bMR\s*ANGIO", n):
        return "MRI"
    if re.search(r"\bCT\b|\bCTA\b|\bCT[- ]?ANGIO", n):
        return "CT"
    if re.search(r"\bUS\b|U\.S\.|\bULTRASOUND|\bDUPLEX|\bDOPPLER|\bDUP\b|\bVEN\b|"
                 r"\bVEINS?\b|\bECHO\b|\bSONO|\bFETAL\b|\bNUCHAL|"
                 r"CAROTID SERIES|FEMORAL VASCULAR|\bART EXT\b|FETAL BPP", n):
        return "US"
    if (n.startswith("CHEST") or "PORT CHEST" in n
        or ("TRAUMA" in n and "CHEST" in n)
        or ("PRE-OP" in n and "CHEST" in n)):
        return "CXR"
    if re.search(r"SKELETAL SURVEY|STAND FEET|STANDING FEET|SCOLIOSIS|"
                 r"BOTH KNEES|CERVICAL SINGLE VIEW", n):
        return "XR"
    if re.search(r"\bPARACENTESIS|\bBIOPSY|\bDRAINAGE|\bGUIDANCE|\bSWALLOW|\bVIDEO\b|"
                 r"\bARTHRO|\bNUCLEAR|\bMYELO|\bFLUORO|\bAORTOGRAM|\bANGIOGRAM\b|"
                 r"\bVENOGRAM|\bLP\b|\bTHORACENTESIS|\bDISCOGRAM|\bCYSTOGRAM|\bENEMA|"
                 r"\bUGI\b|\bBARIUM|\bDEXA\b|\bBONE DENSITOM|PICC|PORT[ ]A[ ]CATH|"
                 r"\bPLACEMENT\b|\bINJECTION\b|\bMODIFIED BARIUM|\bERCP\b|\bEMBO\b|"
                 r"\bTUBE\b|\bTUNNELED\b|\bMOD SEDATION|\bBX[ -]", n):
        return "OTHER"
    if re.search(r"\b(ANKLE|FOOT|FEET|KNEE|TIB|FIB|HIP|WRIST|HAND|SHOULDER|FINGER|ELBOW|"
                 r"PATELLA|CLAVICLE|CALCANEUS|FEMUR|HUMERUS|FOREARM|TIBIA|FIBULA|ULNA|"
                 r"RADIUS|TOE|SCAPULA|LEG|EXTREMITY|PELVIS|ABDOMEN|ABD|"
                 r"L-SPINE|LSPINE|L SPINE|T-SPINE|TSPINE|T SPINE|"
                 r"C-SPINE|CSPINE|C SPINE|LUMBO|LUMBAR|SACRAL|THORACIC|CERVICAL|"
                 r"RIB|SINUS|FACE|SKULL|NASAL|MANDIBLE|MAXIL|ORBIT|MASTOID|"
                 r"STERNUM|LUNG|CHEST|NECK|HEAD|SACRUM|COCCYX|ACROMIO)\b", n):
        return "XR"
    return "OTHER"

# --- BODY_PART rules (priority order) ---
def get_body_part(name):
    n = norm(name)
    if not n or n == "___":
        return "OTHER"
    if re.search(r"SKELETAL SURVEY", n):
        return "OTHER"
    if re.search(r"\bHEAD\b|\bBRAIN\b|\bSKULL\b|\bINTRACRAN|\bSINUS\b|\bMANDIBLE|"
                 r"\bMAXIL|\bFACE\b|\bFACIAL\b|\bNASAL\b|\bORBIT|\bMAXILLOFACIAL|"
                 r"\bTEMPORAL\b|\bMASTOID|\bPITUITARY|\bCIRCLE OF WILLIS", n):
        return "HEAD"
    if re.search(r"\bC[- ]?SPINE\b|\bCSPINE\b|\bCERVICAL[ -]?SPINE|CERVICAL SINGLE VIEW", n):
        return "CSPINE"
    if re.search(r"\bT[- ]?SPINE\b|\bTSPINE\b|\bTHORACIC[ -]?SPINE", n):
        return "TSPINE"
    if re.search(r"\bL[- ]?SPINE\b|\bLSPINE\b|\bLUMBAR|\bLUMBO|\bLUMBOSACRAL|"
                 r"\bSACRAL\b|\bSACRUM|\bCOCCYX|SCOLIOSIS", n):
        return "LSPINE"
    if re.search(r"\bNECK\b|\bTHYROID\b|\bCAROTID\b|\bPAROTID\b|\bSOFT TISSUE NECK", n):
        return "NECK"
    if re.search(r"\bABD.*PELV|\bABDOMEN.*PELV|\bABDOMINOPELVIC", n):
        return "ABD_PELVIS"
    if re.search(r"\bCHEST\b|\bTHORAX\b|\bRIB\b|\bSTERNUM\b|\bLUNG\b|\bACROMIO", n):
        return "CHEST"
    if re.search(r"\bABDOMEN\b|\bABD\b|\bLIVER\b|\bGALLBLADDER\b|\bKIDNEY\b|\bRENAL\b|"
                 r"\bBILIARY\b|\bSPLEEN\b|\bPANCREAS\b|\bMRCP\b|\bABDOMINAL\b|\bAORTA\b|"
                 r"\bRUQ\b|\bLUQ\b|\bMESENTERIC\b|\bENTEROGRAPHY|\bPARACENTESIS|"
                 r"\bERCP\b|\bBX[ -].*LIVER|\bBX[ -].*KIDNEY", n):
        return "ABDOMEN"
    if re.search(r"\bPELV|\bSCROTAL\b|\bSCROTUM\b|\bOB\b|\bOBSTETRIC|\bUTERUS\b|"
                 r"\bOVAR|\bTRANSVAGINAL\b|\bENDOVAGINAL\b|\bPROSTATE|\bRECTUM|"
                 r"\bFETAL\b|\bNUCHAL|FETAL BPP", n):
        return "PELVIS"
    if re.search(r"\bBREAST\b|\bMAMMO", n):
        return "BREAST"
    if re.search(r"\b(ANKLE|FOOT|FEET|KNEE|TIB|FIB|HIP|WRIST|HAND|SHOULDER|FINGER|ELBOW|"
                 r"PATELLA|CLAVICLE|CALCANEUS|FEMUR|HUMERUS|FOREARM|TIBIA|FIBULA|"
                 r"ULNA|RADIUS|TOE|SCAPULA|LEG|EXTREMITY|LIMB|EXT)\b|VEIN|"
                 r"FEMORAL VASCULAR|ART EXT", n):
        return "EXTREMITY"
    return "OTHER"


def main():
    with open(paths.PRIMARY_EXAM_NAMES_DIST_JSON, encoding="utf-8") as f:
        dist = json.load(f)

    rows = []
    mod_count, bp_count = Counter(), Counter()
    combo_count = Counter()
    other_mod_examples = []
    total = 0
    for name, cnt in dist:
        m = get_modality(name)
        b = get_body_part(name)
        rows.append((name, m, b, cnt))
        total += cnt
        mod_count[m] += cnt
        bp_count[b] += cnt
        combo_count[(m, b)] += cnt
        if m == "OTHER":
            other_mod_examples.append((cnt, name, b))

    rows.sort(key=lambda r: -r[3])
    with open(paths.EXAM_NAME_MAPPING_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exam_name", "modality", "body_part", "event_count"])
        for r in rows:
            w.writerow(r)

    L = []
    L.append(f"=== Mapping Coverage Report (v2, ENV={paths.ENV}) ===")
    L.append(f"Distinct exam_names mapped: {len(rows):,}")
    L.append(f"Total primary exam events:  {total:,}")
    L.append("")
    L.append("--- Modality distribution (events) ---")
    for m, c in mod_count.most_common():
        L.append(f"  {c:>8,}  ({100*c/total:5.1f}%)  {m}")
    L.append("")
    L.append("--- Body_part distribution (events) ---")
    for b, c in bp_count.most_common():
        L.append(f"  {c:>8,}  ({100*c/total:5.1f}%)  {b}")
    L.append("")
    L.append("--- (modality, body_part) top 40 combos ---")
    for (m, b), c in combo_count.most_common(40):
        L.append(f"  {c:>8,}  ({100*c/total:5.1f}%)  {m:>6s} x {b}")
    L.append("")
    L.append(f"--- Top 40 exam_names → OTHER modality (residual review) ---")
    other_mod_examples.sort(reverse=True)
    for c, n, b in other_mod_examples[:40]:
        L.append(f"  {c:>7,}  bp={b:<10s}  {n}")

    rpt = "\n".join(L)
    with open(paths.EXAM_NAME_MAPPING_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(rpt)
    print(rpt)
    print(f"\nSaved: {paths.EXAM_NAME_MAPPING_CSV}")
    print(f"Saved: {paths.EXAM_NAME_MAPPING_REPORT_TXT}")


if __name__ == "__main__":
    main()
