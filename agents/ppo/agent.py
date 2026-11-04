from __future__ import annotations

import random

import numpy as np
import torch

from truco import actions as A
from truco.encoding import action_mask, encode, layout
from truco.engine import TrucoHand
from truco.rules import Rules

from ..base import Agent
from .net import TrucoNet, load_checkpoint


class PPOAgent(Agent):
    name = "ppo"

    def __init__(
        self,
        net: TrucoNet,
        greedy: bool = False,
        seed: int | None = None,
        temperature: float = 1.0,
        rules: Rules | None = None,
    ) -> None:
        self.net = net.eval()
        self.greedy = greedy
        self.temperature = temperature
        self.rules = rules
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(self.rng.randrange(2**31))

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> PPOAgent:
        net, rules, _ = load_checkpoint(path)
        return cls(net, rules=rules, **kwargs)

    def _signals(self, hand: TrucoHand) -> bool:
        return hand.rules.signals if self.rules is None else self.rules.signals

    def compatible(self, rules: Rules) -> bool:
        if self.rules is not None and self.rules.num_players != rules.num_players:
            return False
        signals = rules.signals if self.rules is None else self.rules.signals
        if signals and self.net.n_actions != A.num_actions(rules.signal_vocab):
            return False
        return layout(rules.num_players, rules.signal_vocab, signals).size == self.net.obs_dim

    def observe(self, hand: TrucoHand, seat: int) -> tuple[np.ndarray, np.ndarray]:
        return encode(hand, seat, signals=self._signals(hand)), action_mask(hand, self.net.n_actions)

    def probabilities(self, hand: TrucoHand, seat: int) -> np.ndarray:
        obs, mask = self.observe(hand, seat)
        return self.probabilities_batch(obs[None], mask[None])[0]

    def probabilities_batch(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits, _, _ = self.net(torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(mask)))
            probs = torch.softmax(logits / self.temperature, dim=-1)
        return probs.numpy()

    def act(self, hand: TrucoHand, seat: int) -> int:
        if hand.signal_phase and not self._signals(hand):
            return A.signal_action(0)
        probs = self.probabilities(hand, seat)
        if self.greedy:
            return int(np.argmax(probs))
        return int(torch.multinomial(torch.from_numpy(probs), 1, generator=self.generator).item())
