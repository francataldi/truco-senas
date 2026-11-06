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



def _paired_draws(base: list[float], other: list[float], samples: int, seed: int, block: int = 1) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(base, dtype=np.float64)
    o = np.asarray(other, dtype=np.float64)
    if b.shape != o.shape or b.size == 0:
        raise ValueError("las dos condiciones tienen que tener los mismos partidos")
    if block < 1 or b.size % block:
        block = 1
    starts = np.random.default_rng(seed).integers(0, b.size // block, size=(samples, b.size // block)) * block
    idx = (starts[..., None] + np.arange(block)).reshape(samples, b.size)
    return b[idx], o[idx]


def paired_difference(base: list[float], other: list[float], samples: int = 2000, seed: int = 0, block: int = 1) -> dict:
    bb, oo = _paired_draws(base, other, samples, seed, block)
    lo, hi = np.quantile(bb.mean(1) - oo.mean(1), [0.025, 0.975])
    d = float(np.mean(base) - np.mean(other))
    return {"diff": round(d, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def _adjusted(b: np.ndarray, o: np.ndarray, expected: float) -> tuple[np.ndarray, np.ndarray]:
    bm, om = b.mean(-1), o.mean(-1)
    var = b.var(-1)
    cov = ((b - bm[..., None]) * (o - om[..., None])).mean(-1)
    beta = np.divide(cov, var, out=np.zeros_like(cov), where=var > 0)
    return (expected - om) + beta * (bm - expected), beta


def control_adjusted_difference(
    base: list[float], other: list[float], expected: float = 0.5, samples: int = 2000, seed: int = 0, block: int = 1
) -> dict:
    bb, oo = _paired_draws(base, other, samples, seed, block)
    value, beta = _adjusted(np.asarray(base, dtype=np.float64)[None], np.asarray(other, dtype=np.float64)[None], expected)
    lo, hi = np.quantile(_adjusted(bb, oo, expected)[0], [0.025, 0.975])
    return {"diff": round(float(value[0]), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "beta": round(float(beta[0]), 4)}
