"""Statistics used in the analysis (NumPy only; no SciPy required).

* bootstrap_ci        percentile interval of the mean over resampled units
* mcnemar_exact       exact two-sided McNemar test on discordant pairs (binomial, p=1/2)
* wilson              Wilson score interval of a proportion (manual audits)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], *, n: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3  # type: ignore[return-value]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    """b = pairs where only A is correct, c = only B is correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
