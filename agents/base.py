from __future__ import annotations

import random

from truco.engine import TrucoHand


class Agent:
    name = "agent"

    def act(self, hand: TrucoHand, seat: int) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, hand: TrucoHand, seat: int) -> int:
        return self.rng.choice(hand.legal_actions())
