from __future__ import annotations

import gzip
import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import numpy as np

from truco import actions as A
from truco.cards import STRENGTH, envido_points
from truco.engine import TrucoHand
from truco.rules import Rules

from .base import Agent
from .heuristic import HeuristicAgent

ENVIDO_EDGES = (21, 26, 28, 30, 32)
STRENGTH_BUCKET = {14: "A", 13: "A", 12: "S", 11: "S", 10: "3", 9: "2", 8: "F", 7: "F", 6: "M", 5: "M", 4: "L", 3: "L", 2: "L", 1: "L"}


def card_bucket(card: int) -> str:
    return STRENGTH_BUCKET[STRENGTH[card]]


def envido_bucket(value: int) -> int:
    return sum(value >= edge for edge in ENVIDO_EDGES)


def infoset_key(hand: TrucoHand, seat: int) -> str:
    own = "".join(
        "x" if hand.played_slot[seat][i] else card_bucket(card) for i, card in enumerate(hand.hands[seat])
    )
    moves = []
    for s, a in hand.history:
        who = "m" if s == seat else "o"
        if a in A.PLAY_ACTIONS:
            moves.append(f"{who}p{card_bucket(hand.hands[s][a])}")
        else:
            moves.append(f"{who}{a}")
    rival = (seat + 1) % hand.n
    revealed = hand.revealed_envido[rival]
    mano = "M" if hand.mano == seat else "P"
    shown = "-" if revealed is None else envido_bucket(revealed)
    return f"{mano}|{own}|{envido_bucket(envido_points(hand.hands[seat]))}|{''.join(moves)}|{shown}"


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    return np.full(len(regrets), 1.0 / len(regrets))


class MCCFR:
    def __init__(self, rules: Rules, seed: int = 0, plus: bool = True) -> None:
        if rules.num_players != 2:
            raise ValueError("MCCFR baseline is for heads-up truco")
        self.rules = rules
        self.rng = random.Random(seed)
        self.plus = plus
        self.nodes: dict[str, list[np.ndarray]] = {}
        self.iterations = 0

    def _node(self, key: str, size: int) -> list[np.ndarray]:
        node = self.nodes.get(key)
        if node is None:
            node = [np.zeros(size), np.zeros(size)]
            self.nodes[key] = node
        return node

    def traverse(self, hand: TrucoHand, traverser: int, weight: float) -> float:
        if hand.done:
            return float(hand.team_reward(traverser))
        seat = hand.current_player
        legal = hand.legal_actions()
        node = self._node(infoset_key(hand, seat), len(legal))
        sigma = regret_matching(node[0])
        if seat == traverser:
            utils = np.empty(len(legal))
            for i, action in enumerate(legal):
                child = hand.clone()
                child.step(action)
                utils[i] = self.traverse(child, traverser, weight)
            value = float(sigma @ utils)
            node[0] += utils - value
            if self.plus:
                np.maximum(node[0], 0.0, out=node[0])
            return value
        node[1] += weight * sigma
        choice = self.rng.choices(range(len(legal)), weights=sigma)[0]
        hand.step(legal[choice])
        return self.traverse(hand, traverser, weight)

    def iterate(self) -> None:
        self.iterations += 1
        weight = float(self.iterations) if self.plus else 1.0
        hand = TrucoHand(self.rules, rng=random.Random(self.rng.random()), mano=self.rng.randrange(2))
        for traverser in (0, 1):
            self.traverse(hand.clone(), traverser, weight)

    def average_strategy(self) -> dict[str, np.ndarray]:
        table = {}
        for key, (_, strategy_sum) in self.nodes.items():
            total = strategy_sum.sum()
            if total > 0:
                table[key] = (strategy_sum / total).astype(np.float16)
        return table


def _worker(args: tuple) -> dict[str, np.ndarray]:
    rules_dict, iterations, seed, report_every = args
    solver = MCCFR(Rules(**rules_dict), seed=seed)
    start = time.time()
    for i in range(1, iterations + 1):
        solver.iterate()
        if report_every and i % report_every == 0 and seed % 1000 == 0:
            rate = i / (time.time() - start)
            print(f"[worker {seed}] {i}/{iterations} iteraciones, {rate:.0f} it/s, {len(solver.nodes)} infosets", flush=True)
    return {k: v[1] for k, v in solver.nodes.items()}


def train_parallel(rules: Rules, iterations: int, workers: int, seed: int = 0, report_every: int = 1000) -> dict[str, np.ndarray]:
    per_worker = max(1, iterations // workers)
    args = [(rules.to_dict(), per_worker, seed * 1000 + w * 1000, report_every) for w in range(workers)]
    if workers == 1:
        results = [_worker(args[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_worker, args)
    merged: dict[str, np.ndarray] = {}
    for result in results:
        for key, strategy_sum in result.items():
            if key in merged and merged[key].shape == strategy_sum.shape:
                merged[key] = merged[key] + strategy_sum
            else:
                merged.setdefault(key, strategy_sum.copy())
    table = {}
    for key, strategy_sum in merged.items():
        total = strategy_sum.sum()
        if total > 0:
            table[key] = (strategy_sum / total).astype(np.float16)
    return table


def save_table(table: dict[str, np.ndarray], path: str | Path, rules: Rules) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump({"rules": rules.to_dict(), "table": table}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_table(path: str | Path) -> tuple[Rules, dict[str, np.ndarray]]:
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    return Rules(**data["rules"]), data["table"]


class CFRAgent(Agent):
    name = "cfr"

    def __init__(self, table: dict[str, np.ndarray], seed: int | None = None, greedy: bool = False) -> None:
        self.table = table
        self.rng = random.Random(seed)
        self.greedy = greedy
        self.fallback = HeuristicAgent(seed=seed)
        self.hits = 0
        self.misses = 0

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "CFRAgent":
        _, table = load_table(path)
        return cls(table, **kwargs)

    def act(self, hand: TrucoHand, seat: int) -> int:
        legal = hand.legal_actions()
        strategy = self.table.get(infoset_key(hand, seat))
        if strategy is None or len(strategy) != len(legal):
            self.misses += 1
            return self.fallback.act(hand, seat)
        self.hits += 1
        probs = strategy.astype(np.float64)
        probs = probs / probs.sum()
        if self.greedy:
            return legal[int(np.argmax(probs))]
        return legal[self.rng.choices(range(len(legal)), weights=probs)[0]]
