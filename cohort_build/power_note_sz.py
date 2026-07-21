# -*- coding: utf-8 -*-
"""power_note_sz.py — Supplementary SZ: high-acuity small-group power note.

Question (R1-Q2 / revision guide §7.3): how large must the high-acuity
calibration set be so that Hoeffding–Bentkus (HB) certification at
(alpha_high, delta) = (0.05, 0.10) succeeds with probability >= 0.90,
given a candidate threshold whose true joint risk is R_true < alpha?

Model (matches manuscript Eq. (4)–(5), single candidate threshold):
  K ~ Binomial(n, R_true) harmful commits among n calibration cases;
  HB p-value  p(k) = min( exp(-n * kl(k/n || alpha)),  e * P[Binom(n, alpha) <= k] )
  (first term only valid when k/n < alpha; p = 1 if k/n >= alpha).
  Certify iff p(k) <= delta.  Power(n) = P[K <= k_max(n)].

Deterministic closed-form computation — no RNG, no seed needed.
Output: minimal n with Power >= 0.90 for a grid of R_true, plus the power at
the current high-acuity calibration size (n ~ 245, cf. §5.5 n_test ~ 245 with
a 30/30 calibration/test split).
"""
import math

ALPHA = 0.05
DELTA = 0.10
POWER_TARGET = 0.90
N_CURRENT = 245
# R_true grid: bracketing the observed seed-averaged high-acuity joint risks
# (C3 E-sev: e.g. Qwen 0.017; exceedance runs reached <= 0.057)
R_TRUE_GRID = [0.010, 0.017, 0.020, 0.025, 0.030, 0.040]


def kl_bern(a: float, b: float) -> float:
    """Binary KL divergence kl(a || b), natural log."""
    if a <= 0.0:
        return math.log(1.0 / (1.0 - b))
    if a >= 1.0:
        return math.log(1.0 / b)
    return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))


def log_binom_pmf(k: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1.0 - p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P[Binom(n, p) <= k], exact summation in linear space (n <= ~1e5 ok)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.exp(log_binom_pmf(i, n, p)) for i in range(k + 1))


def hb_pvalue(k: int, n: int, alpha: float) -> float:
    """Hoeffding–Bentkus p-value for H0: R >= alpha, observed k/n."""
    r_hat = k / n
    if r_hat >= alpha:
        return 1.0
    hoeffding = math.exp(-n * kl_bern(r_hat, alpha))
    bentkus = math.e * binom_cdf(k, n, alpha)
    return min(hoeffding, bentkus, 1.0)


def k_max_certifiable(n: int, alpha: float, delta: float) -> int:
    """Largest observed count k with p(k) <= delta (p is increasing in k)."""
    k = -1
    for kk in range(0, n + 1):
        if hb_pvalue(kk, n, alpha) <= delta:
            k = kk
        else:
            break
    return k


def power(n: int, r_true: float, alpha: float, delta: float) -> float:
    kmax = k_max_certifiable(n, alpha, delta)
    return binom_cdf(kmax, n, r_true) if kmax >= 0 else 0.0


def min_n_for_power(r_true: float, alpha: float, delta: float,
                    target: float, n_lo: int = 20, n_hi: int = 20000) -> int:
    """Smallest n with power >= target (power is not strictly monotone in n
    because k_max jumps, so scan upward from a bisection bracket)."""
    lo, hi = n_lo, n_hi
    while lo < hi:
        mid = (lo + hi) // 2
        if power(mid, r_true, alpha, delta) >= target:
            hi = mid
        else:
            lo = mid + 1
    # walk back over local jitter to the first n from which power stays >= target
    n = lo
    while n > n_lo and all(power(m, r_true, alpha, delta) >= target
                           for m in range(n - 1, min(n + 20, n_hi))):
        n -= 1
    return lo


def main() -> None:
    print(f"HB certification power note  (alpha={ALPHA}, delta={DELTA}, "
          f"target power={POWER_TARGET})")
    print(f"{'R_true':>8} | {'power @ n=245':>13} | {'min n (power>=0.90)':>20}")
    print("-" * 50)
    for r in R_TRUE_GRID:
        p245 = power(N_CURRENT, r, ALPHA, DELTA)
        nmin = min_n_for_power(r, ALPHA, DELTA, POWER_TARGET)
        print(f"{r:>8.3f} | {p245:>13.3f} | {nmin:>20d}")
    # reference: k_max at current n
    kmax = k_max_certifiable(N_CURRENT, ALPHA, DELTA)
    print(f"\nAt n={N_CURRENT}: certification requires observing "
          f"k <= {kmax} harmful commits ({kmax}/{N_CURRENT} = {kmax/N_CURRENT:.3f}).")


if __name__ == "__main__":
    main()
