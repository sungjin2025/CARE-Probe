# -*- coding: utf-8 -*-
"""Centralized paths for the cohort-build and experiment scripts.

Switch environments via env var CRMSR_ENV ('dev' or 'server').

All locations below are PLACEHOLDERS driven by environment variables — set them
to your own local/server paths before running. No data ships with this
repository; regenerate the MIMIC-derived cohort from your own
PhysioNet-credentialed copy of MIMIC-IV / MIMIC-IV-ED.
"""
import os
from contextlib import contextmanager
from pathlib import Path

ENV = os.environ.get("CRMSR_ENV", "dev").lower()

# Directory that holds your credentialed MIMIC source CSVs (radiology.csv,
# edstays.csv, etc.). Set MIMIC_DATA_DIR to point at it.
_DATA = os.environ.get("MIMIC_DATA_DIR", "/path/to/mimic-data")

if ENV == "dev":
    # ----- Local workstation -----
    MIMIC_NOTE_ZIP         = os.environ.get("MIMIC_NOTE_ZIP", "/path/to/mimic-iv-note-2.2.zip")
    RAD_DETAIL_GZ_ENTRY    = "note/radiology_detail.csv.gz"   # entry path inside the note zip
    RAD_DETAIL_GZ_PATH     = None                             # not extracted on dev
    RADIOLOGY_CSV          = _DATA + "/radiology.csv"
    EDSTAYS_CSV            = _DATA + "/edstays.csv"
    MIMIC_ED_INPUT_HPI_CSV = _DATA + "/mimic_ed_input_hpi.csv"
    LEGACY_COHORT_CSV      = _DATA + "/cohort_imaging_labels_4class.csv"  # for cross-validation (optional)
    WORK_DIR               = os.environ.get("WORK_DIR", "/path/to/cohort_build")

elif ENV == "server":
    # ----- GPU server -----
    # Only WORK_DIR + cohort_v2.csv are needed to run the experiments; the five
    # rebuild inputs below are required only when regenerating the cohort.
    MIMIC_NOTE_ZIP         = None
    RAD_DETAIL_GZ_ENTRY    = None
    RAD_DETAIL_GZ_PATH     = os.environ.get("RAD_DETAIL_GZ_PATH")     # set when rebuilding
    RADIOLOGY_CSV          = os.environ.get("RADIOLOGY_CSV")
    EDSTAYS_CSV            = os.environ.get("EDSTAYS_CSV")
    MIMIC_ED_INPUT_HPI_CSV = os.environ.get("MIMIC_ED_INPUT_HPI_CSV")
    LEGACY_COHORT_CSV      = os.environ.get("LEGACY_COHORT_CSV")
    WORK_DIR               = os.environ.get("WORK_DIR", "/path/to/cohort_build")

else:
    raise ValueError(f"Unknown CRMSR_ENV={ENV!r}. Use 'dev' or 'server'.")

# Ensure work dir exists
Path(WORK_DIR).mkdir(parents=True, exist_ok=True)


# ----- Derived output paths (always under WORK_DIR) -----
def _wd(name: str) -> str:
    return str(Path(WORK_DIR) / name)

PRIMARY_EXAM_NAMES_DIST_JSON = _wd("primary_exam_names_dist.json")
PRIMARY_EXAM_CODES_DIST_JSON = _wd("primary_exam_codes_dist.json")
EXAM_NAME_MAPPING_CSV        = _wd("exam_name_mapping.csv")
EXAM_NAME_MAPPING_REPORT_TXT = _wd("exam_name_mapping_report.txt")
IMAGING_EVENTS_V2_CSV        = _wd("imaging_events_v2.csv")
COHORT_V2_CSV                = _wd("cohort_v2.csv")
COHORT_V2_STATS_JSON         = _wd("cohort_v2_stats.json")
CROSS_VALIDATE_REPORT_TXT    = _wd("cross_validate_report.txt")
CROSS_VALIDATE_DISAGREE_CSV  = _wd("cross_validate_disagreements.csv")


# ----- Helper: open radiology_detail gzip as binary stream (env-agnostic) -----
@contextmanager
def open_radiology_detail_gz():
    """Yields a binary file-like object reading the gzipped radiology_detail CSV.
    Caller wraps with gzip.GzipFile(fileobj=...) for decompression."""
    if ENV == "dev":
        import zipfile
        with zipfile.ZipFile(MIMIC_NOTE_ZIP) as zf:
            with zf.open(RAD_DETAIL_GZ_ENTRY) as f:
                yield f
    else:
        with open(RAD_DETAIL_GZ_PATH, "rb") as f:
            yield f


if __name__ == "__main__":
    print(f"CRMSR_ENV={ENV}")
    print(f"WORK_DIR={WORK_DIR}")
    print(f"MIMIC_NOTE_ZIP={MIMIC_NOTE_ZIP}")
    print(f"RADIOLOGY_CSV={RADIOLOGY_CSV}")
    print(f"EDSTAYS_CSV={EDSTAYS_CSV}")
    print(f"EXAM_NAME_MAPPING_CSV={EXAM_NAME_MAPPING_CSV}")
    print(f"IMAGING_EVENTS_V2_CSV={IMAGING_EVENTS_V2_CSV}")
