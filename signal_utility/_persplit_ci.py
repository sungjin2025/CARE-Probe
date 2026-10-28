# -*- coding: utf-8 -*-
"""_persplit_ci.py -- paired per-split Delta-coverage CIs for Table (signal utility).

Reads work_b_signal_crc_persplit-deleak.csv (server re-run of work_b_signal_crc.py,
pass-1b version with the per-split dump; PROBE_TAG=-deleak, SUBS=42) and computes,
per backbone, the paired per-split (n=10 splits, seeds 42-51) contrasts:

  D1 = commit(combined) - commit(input-only)   [RQ2 "six of eight" claim]
  D2 = commit(probe)    - commit(ptrue)        [P(True) "seven of eight" claim]

For each contrast: mean, SD, mean +/- 1.96*SE, paired-bootstrap percentile CI
(resampling the 10 splits with replacement, B=10,000, seed=42), and the count of
splits with D>0.  CAVEAT (printed + to be kept in the manuscript): the ten splits
share one cohort subsample, so these intervals quantify partition variability
only, not cohort-level uncertainty.

Also runs an ANCHOR check: per-split commit means must reproduce the aggregate
work_b_signal_crc_results-deleak.csv probe/input/combined/ptrue commit values
(tolerance 1e-3 after rounding).

Usage (local, after recovering the persplit CSV into this folder):
  python _persplit_ci.py
Outputs: persplit_ci_summary.csv + console table.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PERSPLIT = os.path.join(HERE, "work_b_signal_crc_persplit-deleak.csv")
AGG = os.path.join(HERE, "work_b_signal_crc_results-deleak.csv")

SEED = 42          # reproducibility (fixed)
B = 10_000         # bootstrap replicates
CONTRASTS = [("combined", "input-only", "D_comb_minus_input"),
             ("probe", "ptrue", "D_probe_minus_ptrue")]
ORDER = ["qwen14b", "llama31-8b", "gemma2-9b", "mistral7b-v03", "phi3mini-128k",
         "med42-8b", "openbio-8b", "mediphi"]


def boot_ci(d, rng, b=B):
    """Paired bootstrap percentile CI over splits."""
    n = len(d)
    idx = rng.integers(0, n, size=(b, n))
    means = d[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    if not os.path.exists(PERSPLIT):
        raise SystemExit(f"[missing] {PERSPLIT} -- generate it by running "
                         "work_b_signal_crc.py with PROBE_TAG=-deleak.")
    ps = pd.read_csv(PERSPLIT)
    # SUBS=42 expected; guard against accidental MAIN (multi-sub) file
    subs = sorted(ps["sub"].unique())
    if subs != [42]:
        print(f"[warn] sub seeds in file: {subs} (expected [42]); using sub=42 only")
        ps = ps[ps["sub"] == 42]

    # ---- anchor: per-split means must reproduce the aggregate CSV ----
    if os.path.exists(AGG):
        agg = pd.read_csv(AGG).set_index(["backbone", "signal"])["commit"].astype(float)
        worst = 0.0
        for (bb, sig), v in agg.items():
            m = ps[(ps.backbone == bb) & (ps.signal == sig)]["commit"].mean()
            if not np.isnan(m):
                worst = max(worst, abs(m - v))
        status = "OK" if worst <= 1e-3 + 5e-4 else "MISMATCH -- STOP, investigate"
        print(f"[anchor] max |per-split mean - aggregate commit| = {worst:.4f} -> {status}")
        if "MISMATCH" in status:
            raise SystemExit(1)
    else:
        print("[anchor] aggregate CSV not found -- skipped")

    rng = np.random.default_rng(SEED)
    rows = []
    for bb in ORDER:
        sub = ps[ps.backbone == bb]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="split", columns="signal", values="commit")
        for a, c, name in CONTRASTS:
            if a not in piv.columns or c not in piv.columns:
                continue
            d = (piv[a] - piv[c]).dropna().to_numpy()
            n = len(d)
            mean, sd = d.mean(), d.std(ddof=1)
            se = sd / np.sqrt(n)
            lo_b, hi_b = boot_ci(d, rng)
            rows.append(dict(backbone=bb, contrast=name, n_splits=n,
                             mean=round(mean, 4), sd=round(sd, 4),
                             lo_1p96se=round(mean - 1.96 * se, 4),
                             hi_1p96se=round(mean + 1.96 * se, 4),
                             boot_lo=round(lo_b, 4), boot_hi=round(hi_b, 4),
                             n_pos=int((d > 0).sum()),
                             excl_zero_boot=bool(lo_b > 0 or hi_b < 0)))
    out = pd.DataFrame(rows)
    outf = os.path.join(HERE, "persplit_ci_summary.csv")
    out.to_csv(outf, index=False)
    print("[saved]", outf)
    for name in [c[2] for c in CONTRASTS]:
        subt = out[out.contrast == name]
        k = int(subt["excl_zero_boot"].sum())
        print(f"\n== {name}: bootstrap CI excludes zero on {k} of {len(subt)} backbones ==")
        print(subt.drop(columns="contrast").to_string(index=False))
    print("\n[caveat] the ten splits share one cohort subsample; intervals quantify "
          "partition variability only (state this wherever the CI is reported).")


if __name__ == "__main__":
    main()
