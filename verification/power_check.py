#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte-Carlo computation of the calibration-set power analysis reported in the
Supplement.

Power of the single-threshold Hoeffding-Bentkus certification test at level
delta, as a function of calibration size n and the true joint risk R (< alpha):
  draw n ~ Bernoulli(R); Rhat = mean; p = HB p-value(Rhat, n, alpha);
  certify (reject H0: R >= alpha) iff p <= delta.  power = P(certify).
"""
import numpy as np
from scipy.stats import binom

ALPHA = 0.10
DELTA = 0.10
E = np.e


def kl_bern(a, b):
    if a <= 0.0:
        return -np.log(1.0 - b)
    if a >= 1.0:
        return -np.log(b)
    return a * np.log(a / b) + (1.0 - a) * np.log((1.0 - a) / (1.0 - b))


def hb_p(rhat, n, alpha=ALPHA):
    if rhat >= alpha:
        return 1.0
    hoeff = np.exp(-n * kl_bern(rhat, alpha))
    bentkus = E * binom.cdf(np.ceil(n * rhat), n, alpha)
    return min(1.0, hoeff, bentkus)


def power(n, true_R, M, rng, alpha=ALPHA, delta=DELTA):
    k = rng.binomial(n, true_R, size=M)        # number of committed errors
    rh = k / n
    # vectorized HB
    hoeff = np.exp(-n * np.array([kl_bern(x, alpha) for x in rh]))
    bentkus = E * binom.cdf(np.ceil(n * rh), n, alpha)
    p = np.minimum(1.0, np.minimum(hoeff, bentkus))
    p[rh >= alpha] = 1.0
    return float(np.mean(p <= delta))


def n90(true_R, rng, M=20000, lo=50, hi=8000):
    """Smallest n with power >= 0.90 at true risk true_R (binary search on a grid)."""
    best = None
    n = lo
    while n <= hi:
        if power(n, true_R, M, rng) >= 0.90:
            best = n
            break
        n = int(np.ceil(n * 1.15))
    return best


def main():
    rng = np.random.default_rng(42)
    print(f"Hoeffding-Bentkus single-test power; alpha=delta={ALPHA}. M=20000.\n")
    print(f"{'true R':>8} {'gap=a-R':>8} | " + " ".join(f"n={n:<6}" for n in (245, 600, 1000)))
    for R in (0.083, 0.06, 0.04, 0.02):
        row = [f"{power(n, R, 20000, rng):.3f}" for n in (245, 600, 1000)]
        print(f"{R:>8.3f} {ALPHA-R:>8.3f} | " + " ".join(f"{v:<8}" for v in row))
    print("\nn for >=90% power (grid search):")
    for R in (0.083, 0.06, 0.04, 0.02):
        print(f"  true R={R:.3f} (gap {ALPHA-R:.3f}): n90 ~ {n90(R, rng)}")


if __name__ == "__main__":
    main()
