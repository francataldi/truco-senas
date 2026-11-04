from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.base import Agent, RandomAgent
from agents.cfr import CFRAgent, infoset_key
from agents.heuristic import HeuristicAgent
from agents.ppo.agent import PPOAgent
from truco import actions as A
from truco.engine import TrucoHand
from truco.features import FEATURE_NAMES

from .dictionary import FEATURE_LABELS, base_rates


@dataclass
class Thought:
    kind: str
    action: int
    probs: dict[int, float] | None = None
    value: float | None = None
    raw_value: float | None = None
    beliefs: list[float] | None = None


def decide(agent: Agent, hand: TrucoHand, seat: int, value_scale: float = 4.0) -> Thought:
    legal = hand.legal_actions()
    if isinstance(agent, PPOAgent):
        if agent.is_mute(hand):
            action = agent.act(hand, seat)
            return Thought("mute", action, {action: 1.0})
        probs, value, beliefs = agent.evaluate(hand, seat)
        action = agent.sample(probs)
        return Thought("policy", action, _over(legal, probs), value * value_scale, value, beliefs.tolist())
    if isinstance(agent, CFRAgent):
        strategy = agent.table.get(infoset_key(hand, seat))
        action = agent.act(hand, seat)
        if strategy is not None and len(strategy) == len(legal):
            weights = strategy.astype(np.float64)
            weights = weights / weights.sum()
            return Thought("table", action, {a: float(w) for a, w in zip(legal, weights)})
        return Thought("rules", action)
    if isinstance(agent, RandomAgent):
        action = agent.act(hand, seat)
        return Thought("random", action, {a: 1.0 / len(legal) for a in legal})
    if isinstance(agent, HeuristicAgent):
        return Thought("rules", agent.act(hand, seat))
    return Thought("rules", agent.act(hand, seat))


def advise(agent: Agent, hand: TrucoHand, seat: int, value_scale: float = 4.0) -> Thought:
    legal = hand.legal_actions()
    if isinstance(agent, PPOAgent) and not agent.is_mute(hand):
        probs, value, beliefs = agent.evaluate(hand, seat)
        options = _over(legal, probs)
        best = max(options, key=options.get)
        return Thought("policy", best, options, value * value_scale, value, beliefs.tolist())
    return Thought("rules", HeuristicAgent(seed=len(hand.history)).act(hand, seat))


def _over(legal: list[int], probs: np.ndarray) -> dict[int, float]:
    total = float(sum(probs[a] for a in legal)) or 1.0
    return {a: float(probs[a]) / total for a in legal}


def beliefs_payload(beliefs: list[float] | None) -> list[dict] | None:
    if beliefs is None:
        return None
    base = base_rates()[0]
    return [
        {"name": name, "label": FEATURE_LABELS[name], "p": round(float(p), 4), "base": round(float(base[i]), 4)}
        for i, (name, p) in enumerate(zip(FEATURE_NAMES, beliefs))
    ]


def context(hand: TrucoHand, you: bool = False) -> str:
    if hand.signal_phase:
        return "Elegís la seña para tu compañero" if you else "Elige la seña para su compañero"
    pending = hand.pending
    verb = "Respondés" if you else "Responde"
    if pending is not None:
        if pending.kind == "truco":
            return f"{verb} al {A.TRUCO_NAMES[pending.level - 1]}"
        return f"{verb} al envido"
    rounds = ("la primera", "la segunda", "la tercera")[min(hand.round, 2)]
    return f"{'Jugás' if you else 'Juega'} {rounds}"
