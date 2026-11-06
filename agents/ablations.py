from __future__ import annotations

import multiprocessing as mp

import numpy as np

from truco import actions as A
from truco.rules import Rules

from .arena import SignalNoise, head_to_head
from .base import Agent, limit_threads
from .registry import make_agent
from .signal_analysis import collect_signals
from .stats import control_adjusted_difference, paired_difference, wilson_interval

CONDITIONS = (
    "normal",
    "compañero_mezclado",
    "rivales_mezclados",
    "ambos_mezclados",
    "rivales_ocultos",
    "seña_constante",
)

VALUES = {
    "valor_escuchar_compañero": "compañero_mezclado",
    "valor_espiar_rivales": "rivales_ocultos",
    "costo_rivales_mezclados": "rivales_mezclados",
    "valor_de_todas_las_señas": "ambos_mezclados",
    "valor_neto_de_señalar": "seña_constante",
}


class ConstantSignalAgent(Agent):
    def __init__(self, agent: Agent, token: int) -> None:
        self.agent = agent
        self.token = int(token)
        self.name = f"{getattr(agent, 'name', 'agent')}_constante"

    def reset(self) -> None:
        self.agent.reset()

    def act(self, hand, seat: int) -> int:
        if hand.signal_phase:
            return A.signal_action(self.token)
        return self.agent.act(hand, seat)


def token_marginal(agent: Agent, rules: Rules, hands: int = 2000, seed: int = 0, zero_score_prob: float = 1.0) -> np.ndarray:
    data = collect_signals(rules, [agent] * rules.num_players, hands, seed=seed, zero_score_prob=zero_score_prob)
    counts = np.bincount(data["tokens"], minlength=rules.signal_vocab).astype(np.float64)
    if counts.sum() == 0:
        return np.full(rules.signal_vocab, 1.0 / rules.signal_vocab)
    return counts / counts.sum()


def condition_setup(condition: str, marginal_a: np.ndarray, marginal_b: np.ndarray) -> tuple[SignalNoise | None, int | None]:
    dist_a = tuple(float(x) for x in marginal_a)
    dist_b = tuple(float(x) for x in marginal_b)
    if condition == "normal":
        return None, None
    if condition == "compañero_mezclado":
        return SignalNoise(partner_a=True, dist_a=dist_a), None
    if condition == "rivales_mezclados":
        return SignalNoise(rivals_a=True, dist_b=dist_b), None
    if condition == "ambos_mezclados":
        return SignalNoise(partner_a=True, rivals_a=True, dist_a=dist_a, dist_b=dist_b), None
    if condition == "rivales_ocultos":
        return SignalNoise(blind_a=True), None
    if condition == "seña_constante":
        return None, int(np.argmax(marginal_a))
    raise ValueError(f"condición desconocida: {condition}")



def _run_condition(task: dict) -> dict:
    rules = Rules(**task["rules"])
    agent_a = make_agent(task["a"], seed=task["seed"])
    agent_b = make_agent(task["b"], seed=task["seed"] + 1)
    limit_threads()
    noise, constant = condition_setup(task["condition"], np.asarray(task["marginal_a"]), np.asarray(task["marginal_b"]))
    if constant is not None:
        agent_a = ConstantSignalAgent(agent_a, constant)
    stats = head_to_head(rules, [agent_a], [agent_b], task["matches"], seed=task["match_seed"], noise=noise, duplicate=task.get("duplicate", False))
    return {"condition": task["condition"], "stats": stats.to_dict(), "wins": stats.wins_a, "outcomes": stats.outcomes, "constant_token": constant}


def run_ablations(
    rules: Rules,
    spec_a: str,
    spec_b: str,
    matches: int,
    seed: int = 0,
    workers: int = 1,
    conditions: tuple[str, ...] = CONDITIONS,
    marginal_hands: int = 2000,
    duplicate: bool = False,
    bootstrap: int = 2000,
    zero_score_prob: float = 1.0,
) -> dict:
    if not rules.signals:
        raise ValueError("las ablaciones de señas necesitan reglas con señas")
    first = make_agent(spec_a, seed=seed + 7)
    second = None if spec_b == spec_a else make_agent(spec_b, seed=seed + 8)
    limit_threads()
    marginal_a = token_marginal(first, rules, marginal_hands, seed=seed + 11, zero_score_prob=zero_score_prob)
    marginal_b = marginal_a if second is None else token_marginal(second, rules, marginal_hands, seed=seed + 12, zero_score_prob=zero_score_prob)
    tasks = [
        {
            "rules": rules.to_dict(),
            "a": spec_a,
            "b": spec_b,
            "condition": condition,
            "matches": matches,
            "seed": seed,
            "match_seed": seed + 1000,
            "marginal_a": marginal_a.tolist(),
            "marginal_b": marginal_b.tolist(),
            "duplicate": duplicate,
        }
        for condition in conditions
    ]
    if workers > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            results = pool.map(_run_condition, tasks)
    else:
        results = [_run_condition(t) for t in tasks]
    by_condition = {r["condition"]: r for r in results}
    summary = {
        "a": spec_a,
        "b": spec_b,
        "rules": rules.to_dict(),
        "matches": matches,
        "seed": seed,
        "duplicado": duplicate,
        "prob_marcador_0_0": zero_score_prob,
        "marginal_a": [round(float(x), 4) for x in marginal_a],
        "marginal_b": [round(float(x), 4) for x in marginal_b],
        "condiciones": {},
        "valores": {},
    }
    for condition in conditions:
        r = by_condition[condition]
        entry = dict(r["stats"])
        if r["constant_token"] is not None:
            entry["seña"] = r["constant_token"]
        summary["condiciones"][condition] = entry
    mirror = spec_b == spec_a
    block = 2 if duplicate else 1
    summary["referencia_valores"] = "simetria_0.5" if mirror else "normal_medido"
    if "normal" in by_condition:
        normal = by_condition["normal"]
        lo, hi = wilson_interval(normal["wins"], matches)
        summary["control_simetria"] = {"diff": round(normal["wins"] / matches - 0.5, 4), "ci95": [round(lo - 0.5, 4), round(hi - 0.5, 4)]} if mirror else None
    for name, condition in VALUES.items():
        if condition not in by_condition:
            continue
        r = by_condition[condition]
        normal = by_condition.get("normal")
        if mirror:
            lo, hi = wilson_interval(r["wins"], matches)
            value = {"diff": round(0.5 - r["wins"] / matches, 4), "ci95": [round(0.5 - hi, 4), round(0.5 - lo, 4)]}
            if normal is not None:
                value["ajustado_por_control"] = control_adjusted_difference(normal["outcomes"], r["outcomes"], samples=bootstrap, seed=seed, block=block)
            summary["valores"][name] = value
        elif normal is not None:
            summary["valores"][name] = paired_difference(normal["outcomes"], r["outcomes"], samples=bootstrap, seed=seed, block=block)
    return summary
