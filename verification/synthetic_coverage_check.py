#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-contained synthetic coverage audit of the CARE-Probe certification procedure
Reproduces the Supplement algorithm EXACTLY on
synthetic (score, loss) data with a KNOWN ground-truth joint risk, so the nominal
guarantee Pr(R_joint(lambda_hat) <= alpha) >= 1 - delta is checked directly,
without the LLM/server.

Procedure under test (= Supplement S1):
  candidate thresholds = sorted DISTINCT observed calibration scores (data-dependent grid);
  for each lambda ascending (safest first): Rhat(lambda) = mean_i loss_i * 1{s_i <= lambda};
  Hoeffding-Bentkus p-value for H0: R(lambda) >= alpha,
     p = min( exp(-n KL(Rhat || alpha)),  e * P[Bin(n, alpha) <= ceil(n Rhat)] ),  (p=1 if Rhat>=alpha)
  fixed-sequence: accept while p <= delta, STOP at first rejection; lambda_hat = last accepted;
  if none accepted -> defer-all (commit nothing, R_joint = 0, trivially valid).

Checks performed:
  (a) a PRE-FIXED grid variant is also run as cross-check;
  (c) tie / empty-commit / boundary handling exercised (a heavy-ties scenario + defer-all tracking);
  (d) many independent replicates; empirical failure rate reported against nominal delta.

Vectorized: certification is O(n log n) per replicate.
"""
import numpy as np
from scipy.stats import binom

ALPHA = 0.10
DELTA = 0.10
E = np.e


def kl_bern_vec(a, b):
    """Elementwise Bernoulli KL(a||b) for array a, scalar b; safe at a in {0,1}."""
    a = np.clip(a, 0.0, 1.0)
    out = np.empty_like(a, dtype=float)
    mid = (a > 0.0) & (a < 1.0)
    out[~mid & (a <= 0.0)] = -np.log(1.0 - b)
    out[~mid & (a >= 1.0)] = -np.log(b)
    am = a[mid]
    out[mid] = am * np.log(am / b) + (1.0 - am) * np.log((1.0 - am) / (1.0 - b))
    return out


def hb_pvalue_vec(rhat, n, alpha=ALPHA):
    """Hoeffding-Bentkus p-values for an array of Rhat (H0: R >= alpha)."""
    rhat = np.asarray(rhat, dtype=float)
    hoeff = np.exp(-n * kl_bern_vec(rhat, alpha))
    bentkus = E * binom.cdf(np.ceil(n * rhat), n, alpha)
    p = np.minimum(1.0, np.minimum(hoeff, bentkus))
    p[rhat >= alpha] = 1.0
    return p


def certify_fast(scores, losses, alpha=ALPHA, delta=DELTA, fixed_grid=None):
    """Return lambda_hat (np.nan = defer-all). Candidate grid = unique scores (S1),
    or a pre-fixed grid if given. Fixed-sequence over ascending candidates."""
    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = losses[order]
    cum_loss = np.cumsum(y_sorted)                # committed-loss count if we commit s<=s_sorted[i]
    if fixed_grid is None:
        cand = np.unique(s_sorted)                               # data-dependent grid (S1)
        counts = np.searchsorted(s_sorted, cand, side="right")   # # cases with s <= cand
        rhat = cum_loss[counts - 1] / n
    else:
        cand = np.asarray(fixed_grid, dtype=float)
        counts = np.searchsorted(s_sorted, cand, side="right")
        rhat = np.where(counts > 0, cum_loss[np.clip(counts - 1, 0, n - 1)] / n, 0.0)
    p = hb_pvalue_vec(rhat, n, alpha)
    accept = p <= delta                            # ascending candidates
    if not accept[0]:
        # even the safest candidate rejected -> but a smaller-than-any-score threshold
        # commits nothing (R=0) and always certifies; S1 defers every case only if no
        # candidate certifies AND committing zero is the fallback. Zero-commit is valid.
        return np.nan if not accept.any() else None
    # fixed-sequence: last accepted before the first rejection
    first_rej = np.argmax(~accept) if (~accept).any() else len(accept)
    if first_rej == 0:
        return np.nan
    return cand[first_rej - 1]


SCEN = {
    "linear P(err|s)=s":   (lambda s: s,                              lambda L: L**2 / 2),
    "steep P(err|s)=s^2":  (lambda s: np.clip(s**2, 0, 1),            lambda L: L**3 / 3),
    "shallow 0.2+0.6s":    (lambda s: 0.2 + 0.6 * s,                  lambda L: 0.2*L + 0.3*L**2),
    "high-base 0.1+0.8s":  (lambda s: np.clip(0.1 + 0.8*s, 0, 1),     lambda L: 0.1*L + 0.4*L**2),
}


def run(name, perr, integ, n_cal, M, rng, fixed=False, tie_round=None):
    fail = 0
    commit_sum = 0.0
    deferall = 0
    for _ in range(M):
        s = rng.uniform(0.0, 1.0, n_cal)
        if tie_round is not None:
            s = np.round(s * tie_round) / tie_round
        y = (rng.uniform(0.0, 1.0, n_cal) < perr(s)).astype(float)
        grid = np.linspace(0.0, 1.0, 201)[1:] if fixed else None
        lam = certify_fast(s, y, fixed_grid=grid)
        if lam is None or np.isnan(lam):
            deferall += 1
            continue
        commit_sum += float(np.mean(s <= lam))
        if integ(lam) > ALPHA:
            fail += 1
    kind = "pre-fixed" if fixed else "data-dependent"
    nonz = M - deferall
    mc = commit_sum / nonz if nonz else 0.0
    print(f"  [{name:22s}] grid={kind:14s} n={n_cal:4d} M={M} "
          f"| FAIL={fail/M:.4f} (<= delta={DELTA}?) "
          f"| mean commit={mc:.3f} | defer-all={deferall/M:.3f}")
    return fail / M


def main():
    rng = np.random.default_rng(20261027)
    print("Synthetic coverage audit of the CARE-Probe certification (Supplement S1).")
    print(f"Target: empirical failure rate <= nominal delta = {DELTA}.\n")
    print("S1 data-dependent candidate grid (= observed calibration scores):")
    worst = 0.0
    for name, (perr, integ) in SCEN.items():
        for nc in (245, 600, 1000):
            worst = max(worst, run(name, perr, integ, nc, 10000, rng))
    print("\nTie / boundary stress (scores rounded to 20 bins -> heavy ties):")
    pe, ig = SCEN["linear P(err|s)=s"]
    worst = max(worst, run("linear + heavy ties", pe, ig, 600, 10000, rng, tie_round=20))
    print("\nCross-check: PRE-FIXED data-independent grid:")
    for key in ("linear P(err|s)=s", "shallow 0.2+0.6s"):
        pe, ig = SCEN[key]
        worst = max(worst, run(key, pe, ig, 600, 10000, rng, fixed=True))
    print(f"\n==> worst-case empirical failure across all scenarios = {worst:.4f} "
          f"(nominal delta = {DELTA}); PASS if <= {DELTA}.")


if __name__ == "__main__":
    main()
