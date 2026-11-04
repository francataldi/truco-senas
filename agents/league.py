from __future__ import annotations

import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from truco.rules import Rules

from .arena import head_to_head
from .ratings import ELO_PER_LOGIT, expected_win_matrix, fit_bradley_terry, rating_table
from .registry import make_agent, spec_label, spec_rules

_ITER = re.compile(r"iter_(\d+)\.pt$")


@dataclass(frozen=True)
class Entrant:
    name: str
    spec: str


def checkpoint_iteration(path: str | Path) -> int | None:
    match = _ITER.search(Path(path).name)
    return int(match.group(1)) if match else None


def run_checkpoints(run_dir: str | Path, per_run: int = 3, iterations: list[int] | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    found = sorted((checkpoint_iteration(p), p) for p in run_dir.glob("iter_*.pt") if checkpoint_iteration(p) is not None)
    if not found:
        last = run_dir / "last.pt"
        return [last] if last.exists() else []
    if iterations:
        chosen = []
        for target in iterations:
            it, path = min(found, key=lambda item: (abs(item[0] - target), -item[0]))
            if path not in chosen:
                chosen.append(path)
        return chosen
    if per_run <= 1:
        return [found[-1][1]]
    picks = np.unique(np.linspace(0, len(found) - 1, min(per_run, len(found))).round().astype(int))
    return [found[i][1] for i in picks]


def parse_entrant(text: str) -> Entrant:
    name, sep, spec = text.partition("=")
    if sep and ":" not in name and name:
        return Entrant(name, spec)
    return Entrant(spec_label(text), text)


def compatible(spec: str, rules: Rules) -> tuple[bool, str]:
    kind = spec.partition(":")[0]
    own = spec_rules(spec)
    if own is None:
        return True, ""
    if own.num_players != rules.num_players:
        return False, f"entrenado para {own.num_players} jugadores"
    if kind.startswith("ppo") and own.signals and rules.signals and own.signal_vocab != rules.signal_vocab:
        return False, f"vocabulario de {own.signal_vocab} señas"
    return True, ""


def _set_threads() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _play_pair(task: dict) -> dict:
    _set_threads()
    rules = Rules(**task["rules"])
    a = make_agent(task["spec_a"], seed=task["seed"])
    b = make_agent(task["spec_b"], seed=task["seed"] + 1)
    stats = head_to_head(rules, [a], [b], task["matches"], seed=task["seed"])
    return {"i": task["i"], "j": task["j"], "wins": stats.wins_a, **stats.to_dict()}


def pair_seed(seed: int, i: int, j: int) -> int:
    return int(np.random.SeedSequence([seed, i, j]).generate_state(1)[0] % (2**31))


def round_robin(rules: Rules, entrants: list[Entrant], matches: int, seed: int = 0, workers: int = 1, progress=None) -> list[dict]:
    tasks = [
        {
            "rules": rules.to_dict(),
            "spec_a": entrants[i].spec,
            "spec_b": entrants[j].spec,
            "i": i,
            "j": j,
            "matches": matches,
            "seed": pair_seed(seed, i, j),
        }
        for i in range(len(entrants))
        for j in range(i + 1, len(entrants))
    ]
    results = []
    if workers > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            for result in pool.imap_unordered(_play_pair, tasks):
                results.append(result)
                if progress is not None:
                    progress(len(results), len(tasks), result)
    else:
        for task in tasks:
            results.append(_play_pair(task))
            if progress is not None:
                progress(len(results), len(tasks), results[-1])
    results.sort(key=lambda r: (r["i"], r["j"]))
    return results


def win_matrix(n: int, results: list[dict]) -> np.ndarray:
    wins = np.zeros((n, n))
    for r in results:
        wins[r["i"], r["j"]] += r["wins"]
        wins[r["j"], r["i"]] += r["matches"] - r["wins"]
    return wins


def summarize_league(
    entrants: list[Entrant],
    results: list[dict],
    samples: int = 1000,
    seed: int = 0,
    prior_sd: float | None = 4.0,
    anchor: str | None = None,
) -> dict:
    names = [e.name for e in entrants]
    wins = win_matrix(len(entrants), results)
    if anchor is None and "heuristico" in names:
        anchor = "heuristico"
    table = rating_table(names, wins, samples=samples, seed=seed, prior_sd=prior_sd, anchor=anchor)
    specs = {e.name: e.spec for e in entrants}
    for row in table:
        row["spec"] = specs[row["nombre"]]
    ratings = fit_bradley_terry(wins, prior_sd=prior_sd)
    expected = expected_win_matrix(ratings)
    games = wins + wins.T
    observed = np.divide(wins, games, out=np.full_like(wins, np.nan), where=games > 0)
    residual = np.nanmax(np.abs(observed - expected)) if len(names) > 1 else 0.0
    pairs = []
    for r in results:
        pairs.append(
            {
                "a": names[r["i"]],
                "b": names[r["j"]],
                "partidos": r["matches"],
                "victorias_a": r["wins"],
                "tasa_a": r["win_rate"],
                "ci95_lo": r["ci95"][0],
                "ci95_hi": r["ci95"][1],
                "esperado_bt": round(float(expected[r["i"], r["j"]]), 4),
                "dif_puntos_por_mano": r["point_diff_per_hand"],
                "manos": r["hands"],
            }
        )
    return {
        "nombres": names,
        "ancla": anchor,
        "prior_sd": prior_sd,
        "bootstrap": samples,
        "elo_por_logit": ELO_PER_LOGIT,
        "ratings": table,
        "partidos": pairs,
        "tasa_observada": np.where(np.isnan(observed), None, np.round(observed, 4)).tolist(),
        "max_residuo_bt": round(float(residual), 4),
    }
