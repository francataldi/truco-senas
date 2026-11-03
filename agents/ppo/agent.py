from __future__ import annotations

import random

import numpy as np
import torch

from truco.encoding import action_mask, encode
from truco.engine import TrucoHand

from ..base import Agent
from .net import TrucoNet, load_checkpoint


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, net: TrucoNet, greedy: bool = False, seed: int | None = None, temperature: float = 1.0) -> None:
        self.net = net.eval()
        self.greedy = greedy
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.generator = torch.Generator().manual_seed(self.rng.randrange(2**31))

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "PPOAgent":
        net, _, _ = load_checkpoint(path)
        return cls(net, **kwargs)

    def probabilities(self, hand: TrucoHand, seat: int) -> np.ndarray:
        obs = torch.from_numpy(encode(hand, seat)).unsqueeze(0)
        mask = torch.from_numpy(action_mask(hand, self.net.n_actions)).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = self.net(obs, mask)
            probs = torch.softmax(logits / self.temperature, dim=-1)[0]
        return probs.numpy()

    def act(self, hand: TrucoHand, seat: int) -> int:
        probs = self.probabilities(hand, seat)
        if self.greedy:
            return int(np.argmax(probs))
        return int(torch.multinomial(torch.from_numpy(probs), 1, generator=self.generator).item())
