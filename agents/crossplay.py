from __future__ import annotations

import itertools
import multiprocessing as mp
import random

import numpy as np

from truco import actions as A
from truco.engine import TrucoHand
from truco.rules import Rules

from .ablations import token_marginal
from .arena import SignalNoise, head_to_head
from .base import limit_threads
from .league import pair_seed
from .registry import make_agent
from .stats import difference_summary, proportion_summary


def first_signal_distribution(agent, rules: Rules, hands: int, seed: int = 0, zero_score_prob: float = 0.3) -> np.ndarray:
    rng = random.Random(seed)
    batched = hasattr(agent, "probabilities_batch")
    obs, masks = [], []
    probs = np.zeros((hands, rules.signal_vocab))
    for h in range(hands):
        scores = (0, 0) if rng.random() < zero_score_prob else (rng.randrange(rules.target), rng.randrange(rules.target))
        hand = TrucoHand(rules, rng=random.Random(rng.random()), scores=scores, mano=rng.randrange(rules.num_players))
        if batched:
            o, m = agent.observe(hand, hand.current_player)
            obs.append(o)
            masks.append(m)
        else:
            probs[h, A.signal_token(agent.act(hand, hand.current_player))] = 1.0
    if batched:
        probs = agent.probabilities_batch(np.stack(obs), np.stack(masks))[:, A.SIGNAL_BASE : A.SIGNAL_BASE + rules.signal_vocab].astype(np.float64)
    return probs / np.maximum(probs.sum(1, keepdims=True), 1e-12)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def best_permutation(joint: np.ndarray) -> tuple[float, list[int]]:
    k = joint.shape[0]
    if k <= 9:
        perms = np.array(list(itertools.permutations(range(k))), dtype=np.int64)
        scores = joint[np.arange(k)[None, :], perms].sum(1)
        best = int(np.argmax(scores))
        return float(scores[best]), perms[best].tolist()
    remaining = set(range(k))
    mapping = [0] * k
    for a in np.argsort(-joint.max(1)):
        b = max(remaining, key=lambda j: joint[a, j])
        mapping[int(a)] = b
        remaining.discard(b)
    return float(joint[np.arange(k), mapping].sum()), mapping


def dictionary_alignment(p: np.ndarray, q: np.ndarray) -> dict:
    joint = p.T @ q / len(p)
    mp_, mq = joint.sum(1), joint.sum(0)
    permuted, mapping = best_permutation(joint)
    nz = joint > 0
    mi = float((joint[nz] * np.log2(joint[nz] / np.outer(mp_, mq)[nz])).sum())
    hp, hq = _entropy(mp_), _entropy(mq)
    return {
        "acuerdo_directo": round(float(np.trace(joint)), 4),
        "acuerdo_azar": round(float(mp_ @ mq), 4),
        "acuerdo_permutado": round(permuted, 4),
        "permutacion": mapping,
        "mi_bits": round(mi, 4),
        "mi_normalizada": round(mi / max(min(hp, hq), 1e-12), 4),
        "entropia_bits": [round(hp, 4), round(hq, 4)],
    }


def _team_task(task: dict) -> dict:
    rules = Rules(**task["rules"])
    first = make_agent(task["a"], seed=task["seed"])
    second = make_agent(task["b"], seed=task["seed"] + 1)
    opponent = make_agent(task["opponent"], seed=task["seed"] + 2)
    limit_threads()
    noise = None
    if task["scrambled"]:
        marginal = token_marginal(make_agent(task["a"], seed=task["seed"] + 3), rules, task["marginal_hands"], seed=task["seed"] + 4)
        noise = SignalNoise(partner_a=True, dist_a=tuple(float(x) for x in marginal))
    stats = head_to_head(rules, [first, second], [opponent], task["matches"], seed=task["seed"], noise=noise)
    return {"i": task["i"], "j": task["j"], "scrambled": task["scrambled"], "wins": stats.wins_a, "outcomes": stats.outcomes, **stats.to_dict()}


def _pooled(cells: list[dict]) -> tuple[int, int]:
    return sum(c["wins"] for c in cells), sum(c["matches"] for c in cells)


def _bootstrap_gap(sp: list[dict], xp: list[dict], scr: list[dict], samples: int, seed: int) -> tuple[float, float] | None:
    if not sp or not xp or not scr:
        return None
    rng = np.random.default_rng(seed)

    def draw(cells: list[dict]) -> float:
        wins = total = 0
        for c in cells:
            outcomes = np.asarray(c["outcomes"])
            wins += outcomes[rng.integers(0, len(outcomes), len(outcomes))].sum()
            total += len(outcomes)
        return wins / max(1, total)

    ratios = []
    for _ in range(samples):
        s, x, r = draw(sp), draw(xp), draw(scr)
        if abs(s - r) > 1e-9:
            ratios.append((s - x) / (s - r))
    if len(ratios) < samples // 2:
        return None
    return (float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975)))


def run_crossplay(
    rules: Rules,
    names: list[str],
    specs: list[str],
    opponent: str = "heuristic",
    matches: int = 200,
    seed: int = 0,
    workers: int = 1,
    marginal_hands: int = 1000,
    signal_hands: int = 2000,
    bootstrap: int = 2000,
) -> dict:
    k = len(specs)
    tasks = []
    for i in range(k):
        for j in range(i, k):
            tasks.append({"i": i, "j": j, "scrambled": False})
        if rules.signals:
            tasks.append({"i": i, "j": i, "scrambled": True})
    for t in tasks:
        t.update(
            {
                "rules": rules.to_dict(),
                "a": specs[t["i"]],
                "b": specs[t["j"]],
                "opponent": opponent,
                "matches": matches,
                "seed": pair_seed(seed, t["i"], t["j"]),
                "marginal_hands": marginal_hands,
            }
        )
    if workers > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            results = pool.map(_team_task, tasks)
    else:
        results = [_team_task(t) for t in tasks]
    matrix = np.full((k, k), np.nan)
    cells = []
    for r in results:
        if not r["scrambled"]:
            matrix[r["i"], r["j"]] = matrix[r["j"], r["i"]] = r["win_rate"]
        cells.append(
            {
                "a": names[r["i"]],
                "b": names[r["j"]],
                "tipo": "mezclado" if r["scrambled"] else ("self-play" if r["i"] == r["j"] else "cross-play"),
                "partidos": r["matches"],
                "victorias": r["wins"],
                "tasa": r["win_rate"],
                "ci95_lo": r["ci95"][0],
                "ci95_hi": r["ci95"][1],
                "dif_puntos_por_mano": r["point_diff_per_hand"],
            }
        )
    sp = [r for r in results if not r["scrambled"] and r["i"] == r["j"]]
    xp = [r for r in results if not r["scrambled"] and r["i"] != r["j"]]
    scr = [r for r in results if r["scrambled"]]
    summary = {
        "nombres": names,
        "specs": specs,
        "rival": opponent,
        "rules": rules.to_dict(),
        "partidos_por_celda": matches,
        "matriz": np.where(np.isnan(matrix), None, np.round(matrix, 4)).tolist(),
        "celdas": cells,
    }
    pooled = {}
    for label, group in (("self_play", sp), ("cross_play", xp), ("self_play_mezclado", scr)):
        if group:
            pooled[label] = proportion_summary(*_pooled(group))
    summary["agregado"] = pooled
    diffs = {}
    for label, (first, second) in {
        "self_menos_cross": (sp, xp),
        "self_menos_mezclado": (sp, scr),
        "cross_menos_mezclado": (xp, scr),
    }.items():
        if first and second:
            diffs[label] = difference_summary(*_pooled(first), *_pooled(second))
    summary["diferencias"] = diffs
    if sp and xp and scr:
        s, x, r = (_pooled(g)[0] / _pooled(g)[1] for g in (sp, xp, scr))
        gap = {"valor": round((s - x) / (s - r), 4) if abs(s - r) > 1e-9 else None}
        interval = _bootstrap_gap(sp, xp, scr, bootstrap, seed)
        if interval is not None:
            gap["ci95"] = [round(interval[0], 4), round(interval[1], 4)]
        summary["fraccion_del_valor_de_las_señas_perdida"] = gap
    if rules.signals and signal_hands > 0 and k > 1:
        agents = [make_agent(s, seed=seed) for s in specs]
        limit_threads()
        dists = [first_signal_distribution(agent, rules, signal_hands, seed=seed + 99) for agent in agents]
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append({"a": names[i], "b": names[j], **dictionary_alignment(dists[i], dists[j])})
        summary["diccionarios"] = pairs
        for key in ("acuerdo_directo", "acuerdo_azar", "acuerdo_permutado", "mi_normalizada"):
            summary.setdefault("diccionarios_media", {})[key] = round(float(np.mean([p[key] for p in pairs])), 4)
    return summary
