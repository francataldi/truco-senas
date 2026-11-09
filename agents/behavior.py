from __future__ import annotations

import multiprocessing as mp
import random

import numpy as np

from truco import actions as A
from truco.engine import TrucoHand, TrucoMatch
from truco.rules import Rules

from .arena import SignalNoise, seat_agents
from .base import limit_threads
from .registry import make_agent
from .signal_analysis import top_card_bucket
from .stats import wilson_interval

SOURCES = ("envido", "cartas", "truco_rechazado", "mazo")
COUNTERS = (
    "envido_canta",
    "envido_canta_flojo",
    "envido_canta_suma",
    "envido_enfrenta",
    "envido_acepta",
    "envido_rechaza",
    "envido_sube",
    "envido_jugado",
    "envido_gana",
    "truco_canta",
    "truco_canta_sin_cartas",
    "truco_enfrenta",
    "truco_acepta",
    "truco_rechaza",
    "truco_sube",
    "mazo",
)
WEAK_ENVIDO = 25
WEAK_TOP_BUCKET = 4


def _source(reason: str | None) -> str:
    reason = (reason or "").removeprefix("target_")
    if reason == "no_quiero":
        return "truco_rechazado"
    if reason == "mazo":
        return "mazo"
    if reason == "envido":
        return "envido"
    return "cartas"


def _weak_hand(hand: TrucoHand, seat: int) -> bool:
    cards = hand.remaining_cards(seat)
    return bool(cards) and top_card_bucket(cards) >= WEAK_TOP_BUCKET


def _record(hand: TrucoHand, seat: int, action: int, counts: list[dict]) -> None:
    pending = hand.pending
    team = hand.team(seat)
    mine = counts[team]
    if action in A.ENVIDO_ACTIONS:
        if pending is None or pending.kind == "truco":
            mine["envido_canta"] += 1
            mine["envido_canta_suma"] += hand.envido_of[seat]
            mine["envido_canta_flojo"] += int(hand.envido_of[seat] < WEAK_ENVIDO)
        else:
            mine["envido_enfrenta"] += 1
            mine["envido_sube"] += 1
    elif action == A.TRUCO:
        if pending is None:
            if hand.truco_level == 0:
                mine["truco_canta"] += 1
                mine["truco_canta_sin_cartas"] += int(_weak_hand(hand, seat))
        else:
            mine["truco_enfrenta"] += 1
            mine["truco_sube"] += 1
    elif action in (A.QUIERO, A.NO_QUIERO) and pending is not None:
        kind = "envido" if pending.kind == "envido" else "truco"
        mine[f"{kind}_enfrenta"] += 1
        mine[f"{kind}_acepta" if action == A.QUIERO else f"{kind}_rechaza"] += 1
    elif action == A.MAZO:
        mine["mazo"] += 1


def _play_match(match: TrucoMatch, seats: list, options: dict, a_team: int, totals: dict) -> tuple[int, int, np.ndarray]:
    diff = np.zeros(len(SOURCES))
    length = 0
    while not match.done:
        hand = match.new_hand(**options)
        counts = [dict.fromkeys(COUNTERS, 0) for _ in range(2)]
        while not hand.done:
            seat = hand.current_player
            action = seats[seat].act(hand, seat)
            envido_quiero = action == A.QUIERO and hand.pending is not None and hand.pending.kind == "envido"
            _record(hand, seat, action, counts)
            hand.step(action)
            if envido_quiero and hand.envido_winner is not None:
                for team in (0, 1):
                    counts[team]["envido_jugado"] += 1
                counts[hand.envido_winner]["envido_gana"] += 1
        points = np.zeros((2, len(SOURCES)))
        for team in (0, 1):
            envido = hand.envido_value if hand.envido_winner == team else 0
            points[team, 0] += envido
            points[team, SOURCES.index(_source(hand.end_reason))] += hand.points[team] - envido
        for side, team in (("a", a_team), ("b", 1 - a_team)):
            for key, value in counts[team].items():
                totals[side][key] += value
            totals[side]["puntos"] += points[team]
            totals[side]["manos_ganadas"] += int(hand.winner_team == team)
        diff += points[a_team] - points[1 - a_team]
        length += 1
        match.close_hand()
    return int(match.winner == a_team), length, diff


def _empty_side() -> dict:
    return {**dict.fromkeys(COUNTERS, 0), "puntos": np.zeros(len(SOURCES)), "manos_ganadas": 0}


def _empty_totals() -> dict:
    return {"a": _empty_side(), "b": _empty_side()}


def _run_chunk(task: dict) -> dict:
    rules = Rules(**task["rules"])
    agent_a = make_agent(task["a"], seed=task["seed"] + 1)
    agent_b = make_agent(task["b"], seed=task["seed"] + 2)
    limit_threads()
    noise = SignalNoise(**task["noise"]) if task["noise"] else SignalNoise()
    rng = random.Random(task["seed"])
    totals = _empty_totals()
    outcomes, lengths, diffs = [], [], []
    for m in range(task["matches"]):
        a_team = m % 2
        seats = seat_agents(rules, [agent_a], [agent_b], a_team)
        if not task["duplicate"] or a_team == 0:
            match_seed = rng.random()
            first_mano = rng.randrange(rules.num_players)
        match = TrucoMatch(rules, rng=random.Random(match_seed), first_mano=first_mano)
        for agent in seats:
            agent.reset()
        won, length, diff = _play_match(match, seats, noise.for_match(a_team), a_team, totals)
        outcomes.append(won)
        lengths.append(length)
        diffs.append(diff)
    return {"totals": totals, "outcomes": outcomes, "lengths": lengths, "diffs": diffs}


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _side_summary(t: dict, hands: int) -> dict:
    points = t["puntos"]
    return {
        "puntos_por_mano": round(float(points.sum()) / hands, 4),
        "puntos_por_mano_por_fuente": {s: round(float(points[i]) / hands, 4) for i, s in enumerate(SOURCES)},
        "manos_ganadas": _ratio(t["manos_ganadas"], hands),
        "envido": {
            "canta_primero": _ratio(t["envido_canta"], hands),
            "envido_medio_al_cantar": _ratio(t["envido_canta_suma"], t["envido_canta"]),
            "canta_con_menos_de_25": _ratio(t["envido_canta_flojo"], t["envido_canta"]),
            "acepta": _ratio(t["envido_acepta"], t["envido_enfrenta"]),
            "rechaza": _ratio(t["envido_rechaza"], t["envido_enfrenta"]),
            "sube": _ratio(t["envido_sube"], t["envido_enfrenta"]),
            "gana_los_que_se_juegan": _ratio(t["envido_gana"], t["envido_jugado"]),
        },
        "truco": {
            "canta": _ratio(t["truco_canta"], hands),
            "canta_sin_cartas_altas": _ratio(t["truco_canta_sin_cartas"], t["truco_canta"]),
            "acepta": _ratio(t["truco_acepta"], t["truco_enfrenta"]),
            "rechaza": _ratio(t["truco_rechaza"], t["truco_enfrenta"]),
            "sube": _ratio(t["truco_sube"], t["truco_enfrenta"]),
        },
        "se_va_al_mazo": _ratio(t["mazo"], hands),
    }


def compare(
    rules: Rules,
    spec_a: str,
    spec_b: str,
    matches: int,
    seed: int = 0,
    duplicate: bool = True,
    workers: int = 1,
    noise: dict | None = None,
    samples: int = 1000,
) -> dict:
    chunks = max(1, min(workers, matches // 2))
    sizes = [matches // chunks + (1 if i < matches % chunks else 0) for i in range(chunks)]
    if duplicate:
        sizes = [s + s % 2 for s in sizes]
    tasks = [
        {"rules": rules.to_dict(), "a": spec_a, "b": spec_b, "matches": n, "seed": seed + 7919 * i, "duplicate": duplicate, "noise": noise}
        for i, n in enumerate(sizes)
    ]
    if len(tasks) > 1:
        with mp.get_context("spawn").Pool(len(tasks)) as pool:
            results = pool.map(_run_chunk, tasks)
    else:
        results = [_run_chunk(t) for t in tasks]
    totals = _empty_totals()
    outcomes, lengths, diffs = [], [], []
    for r in results:
        for side in ("a", "b"):
            for key, value in r["totals"][side].items():
                totals[side][key] += value
        outcomes += r["outcomes"]
        lengths += r["lengths"]
        diffs += r["diffs"]
    n = len(outcomes)
    hands = int(sum(lengths))
    wins = int(sum(outcomes))
    lo, hi = wilson_interval(wins, n)
    diffs_arr = np.asarray(diffs)
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    block = 2 if duplicate else 1
    groups = n // block
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, groups, size=(samples, groups))
    idx = (idx[..., None] * block + np.arange(block)).reshape(samples, -1)
    boot = diffs_arr[idx].sum(1) / lengths_arr[idx].sum(1)[:, None]
    by_source = {
        s: {
            "diff": round(float(diffs_arr[:, i].sum() / hands), 4),
            "ci95": [round(float(np.quantile(boot[:, i], 0.025)), 4), round(float(np.quantile(boot[:, i], 0.975)), 4)],
        }
        for i, s in enumerate(SOURCES)
    }
    total = boot.sum(1)
    return {
        "a": spec_a,
        "b": spec_b,
        "rules": rules.to_dict(),
        "duplicado": duplicate,
        "ruido": noise,
        "partidos": n,
        "manos": hands,
        "gana_a": round(wins / n, 4),
        "gana_a_ci95": [round(lo, 4), round(hi, 4)],
        "diferencia_por_mano": {
            "diff": round(float(diffs_arr.sum() / hands), 4),
            "ci95": [round(float(np.quantile(total, 0.025)), 4), round(float(np.quantile(total, 0.975)), 4)],
        },
        "diferencia_por_fuente": by_source,
        "equipo_a": _side_summary(totals["a"], hands),
        "equipo_b": _side_summary(totals["b"], hands),
    }
