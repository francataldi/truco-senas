from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054


def wilson_interval(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_interval(k1: float, n1: int, k2: float, n2: int, z: float = Z95) -> tuple[float, float]:
    if n1 <= 0 or n2 <= 0:
        return (-1.0, 1.0)
    p1 = k1 / n1
    p2 = k2 / n2
    l1, u1 = wilson_interval(k1, n1, z)
    l2, u2 = wilson_interval(k2, n2, z)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def proportion_summary(successes: float, n: int) -> dict:
    lo, hi = wilson_interval(successes, n)
    return {"n": int(n), "k": float(successes), "p": round(successes / n, 4) if n else 0.0, "ci95": [round(lo, 4), round(hi, 4)]}


def difference_summary(k1: float, n1: int, k2: float, n2: int) -> dict:
    lo, hi = newcombe_interval(k1, n1, k2, n2)
    d = (k1 / n1 if n1 else 0.0) - (k2 / n2 if n2 else 0.0)
    return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def bootstrap_mean_interval(values: np.ndarray, samples: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(1)
    alpha = (1 - level) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))
