from __future__ import annotations

import math

import numpy as np

ELO_PER_LOGIT = 400.0 / math.log(10.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def log_likelihood(wins: np.ndarray, ratings: np.ndarray) -> float:
    wins = np.asarray(wins, dtype=np.float64)
    diff = ratings[:, None] - ratings[None, :]
    p = np.clip(_sigmoid(diff), 1e-12, 1.0)
    return float((wins * np.log(p)).sum())


def fit_bradley_terry(wins: np.ndarray, prior_sd: float | None = 4.0, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    wins = np.asarray(wins, dtype=np.float64)
    k = wins.shape[0]
    if wins.shape != (k, k):
        raise ValueError("la matriz de victorias tiene que ser cuadrada")
    games = wins + wins.T
    lam = 0.0 if prior_sd is None or not math.isfinite(prior_sd) else 1.0 / (prior_sd * prior_sd)
    ratings = np.zeros(k)
    for _ in range(max_iter):
        p = _sigmoid(ratings[:, None] - ratings[None, :])
        grad = (wins - games * p).sum(1) - lam * ratings
        w = games * p * (1.0 - p)
        hess = w - np.diag(w.sum(1)) - lam * np.eye(k)
        if lam > 0:
            step = np.linalg.solve(hess, grad)
        else:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        ratings = ratings - step
        if np.abs(step).max() < tol:
            break
    return ratings - ratings.mean()


def expected_win_matrix(ratings: np.ndarray) -> np.ndarray:
    p = _sigmoid(ratings[:, None] - ratings[None, :])
    np.fill_diagonal(p, np.nan)
    return p


def bootstrap_ratings(
    wins: np.ndarray,
    samples: int = 1000,
    seed: int = 0,
    prior_sd: float | None = 4.0,
    anchor: int | None = None,
) -> np.ndarray:
    wins = np.asarray(wins, dtype=np.float64)
    k = wins.shape[0]
    games = np.rint(wins + wins.T).astype(np.int64)
    upper = np.triu_indices(k, 1)
    n = games[upper]
    p = np.divide(wins[upper], n, out=np.full(len(n), 0.5), where=n > 0)
    rng = np.random.default_rng(seed)
    out = np.empty((samples, k))
    for b in range(samples):
        drawn = rng.binomial(n, p)
        resampled = np.zeros((k, k))
        resampled[upper] = drawn
        resampled[upper[1], upper[0]] = n - drawn
        ratings = fit_bradley_terry(resampled, prior_sd=prior_sd)
        if anchor is not None:
            ratings = ratings - ratings[anchor]
        out[b] = ratings
    return out


def rating_table(
    names: list[str],
    wins: np.ndarray,
    samples: int = 1000,
    seed: int = 0,
    prior_sd: float | None = 4.0,
    anchor: str | None = None,
    level: float = 0.95,
) -> list[dict]:
    wins = np.asarray(wins, dtype=np.float64)
    anchor_index = names.index(anchor) if anchor is not None and anchor in names else None
    ratings = fit_bradley_terry(wins, prior_sd=prior_sd)
    if anchor_index is not None:
        ratings = ratings - ratings[anchor_index]
    boot = bootstrap_ratings(wins, samples=samples, seed=seed, prior_sd=prior_sd, anchor=anchor_index)
    alpha = (1.0 - level) / 2.0
    lo = np.quantile(boot, alpha, axis=0)
    hi = np.quantile(boot, 1.0 - alpha, axis=0)
    best = np.bincount(boot.argmax(1), minlength=len(names)) / max(1, samples)
    games = (wins + wins.T).sum(1)
    total_wins = wins.sum(1)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "nombre": name,
                "rating": round(float(ratings[i]), 4),
                "elo": round(float(ratings[i] * ELO_PER_LOGIT), 1),
                "elo_lo": round(float(lo[i] * ELO_PER_LOGIT), 1),
                "elo_hi": round(float(hi[i] * ELO_PER_LOGIT), 1),
                "p_mejor": round(float(best[i]), 4),
                "partidos": int(games[i]),
                "victorias": int(round(total_wins[i])),
                "tasa": round(float(total_wins[i] / games[i]), 4) if games[i] else 0.0,
            }
        )
    rows.sort(key=lambda r: -r["rating"])
    return rows
