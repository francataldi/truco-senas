from __future__ import annotations

import random

import numpy as np

from truco import actions as A
from truco.encoding import action_mask, encode, layout
from truco.engine import TrucoHand
from truco.rules import Rules

from ..base import Agent


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class PolicyAgent(Agent):
    name = "policy"

    def __init__(self, net, greedy: bool = False, seed: int | None = None, temperature: float = 1.0, rules: Rules | None = None) -> None:
        self.net = net
        self.greedy = greedy
        self.temperature = temperature
        self.rules = rules
        self.rng = random.Random(seed)

    def _signals(self, hand: TrucoHand) -> bool:
        return hand.rules.signals if self.rules is None else self.rules.signals

    def is_mute(self, hand: TrucoHand) -> bool:
        return hand.signal_phase and not self._signals(hand)

    def compatible(self, rules: Rules) -> bool:
        if self.rules is not None and self.rules.num_players != rules.num_players:
            return False
        signals = rules.signals if self.rules is None else self.rules.signals
        if signals and self.net.n_actions != A.num_actions(rules.signal_vocab):
            return False
        return layout(rules.num_players, rules.signal_vocab, signals).size == self.net.obs_dim

    def observe(self, hand: TrucoHand, seat: int) -> tuple[np.ndarray, np.ndarray]:
        return encode(hand, seat, signals=self._signals(hand)), action_mask(hand, self.net.n_actions)

    def forward(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def evaluate(self, hand: TrucoHand, seat: int) -> tuple[np.ndarray, float, np.ndarray]:
        obs, mask = self.observe(hand, seat)
        logits, value, aux = self.forward(obs[None], mask[None])
        return softmax(logits, self.temperature)[0], float(value[0]), sigmoid(aux[0])

    def probabilities(self, hand: TrucoHand, seat: int) -> np.ndarray:
        obs, mask = self.observe(hand, seat)
        return self.probabilities_batch(obs[None], mask[None])[0]

    def probabilities_batch(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits, _, _ = self.forward(np.ascontiguousarray(obs, dtype=np.float32), np.ascontiguousarray(mask))
        return softmax(logits, self.temperature)

    def sample(self, probs: np.ndarray) -> int:
        if self.greedy:
            return int(np.argmax(probs))
        return self.rng.choices(range(len(probs)), weights=[float(p) for p in probs])[0]

    def act(self, hand: TrucoHand, seat: int) -> int:
        if self.is_mute(hand):
            return A.signal_action(0)
        return self.sample(self.probabilities(hand, seat))
