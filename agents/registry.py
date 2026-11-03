from __future__ import annotations

from .base import Agent, RandomAgent
from .heuristic import HeuristicAgent


def make_agent(spec: str, seed: int | None = None) -> Agent:
    kind, _, path = spec.partition(":")
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if kind == "heuristic-mudo":
        return HeuristicAgent(seed=seed, use_signals=False)
    if kind in ("ppo", "ppo-greedy"):
        from .ppo.agent import PPOAgent

        return PPOAgent.from_checkpoint(path, greedy=kind == "ppo-greedy", seed=seed)
    if kind in ("cfr", "cfr-greedy"):
        from .cfr import CFRAgent

        return CFRAgent.from_file(path, greedy=kind == "cfr-greedy", seed=seed)
    raise ValueError(f"agente desconocido: {spec}")
