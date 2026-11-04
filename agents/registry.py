from __future__ import annotations

import json
import re
from pathlib import Path

from truco.rules import Rules

from .base import Agent, RandomAgent
from .heuristic import HeuristicAgent

_ITER = re.compile(r"iter_(\d+)\.pt$")
_BASE_LABELS = {"random": "random", "heuristic": "heuristico", "heuristic-mudo": "heuristico_mudo"}


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


def spec_label(spec: str) -> str:
    kind, _, path = spec.partition(":")
    if kind in _BASE_LABELS:
        return _BASE_LABELS[kind]
    suffix = "_greedy" if kind.endswith("-greedy") else ""
    p = Path(path)
    if kind.startswith("ppo"):
        match = _ITER.search(p.name)
        step = str(int(match.group(1))) if match else ("final" if p.stem == "last" else p.stem)
        return f"{p.parent.name or p.stem}@{step}{suffix}"
    if kind.startswith("cfr"):
        return f"cfr_{p.stem}{suffix}"
    return spec


def spec_rules(spec: str) -> Rules | None:
    kind, _, path = spec.partition(":")
    if kind.startswith("ppo"):
        from .ppo.net import read_checkpoint_rules

        return read_checkpoint_rules(path)
    if kind.startswith("cfr"):
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            return Rules(**json.loads(str(data["meta"]))["rules"])
    if kind in _BASE_LABELS:
        return None
    raise ValueError(f"agente desconocido: {spec}")
