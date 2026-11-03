from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Rules:
    num_players: int = 4
    target: int = 30
    flor: bool = False
    signals: bool = True
    signal_vocab: int = 8
    spy_prob: float = 0.2
    envido: bool = True

    def __post_init__(self) -> None:
        if self.num_players not in (2, 4):
            raise ValueError("num_players must be 2 or 4")
        if self.signals and self.num_players != 4:
            object.__setattr__(self, "signals", False)
        if self.signal_vocab < 2:
            raise ValueError("signal_vocab must be at least 2")
        if not 0.0 <= self.spy_prob <= 1.0:
            raise ValueError("spy_prob must be in [0, 1]")

    @property
    def num_teams(self) -> int:
        return 2

    def team_of(self, seat: int) -> int:
        return seat % 2

    def to_dict(self) -> dict:
        return asdict(self)


HEADS_UP = Rules(num_players=2, target=30, flor=False, signals=False)
PAREJAS = Rules(num_players=4, target=30, flor=False, signals=True)
PAREJAS_CON_FLOR = Rules(num_players=4, target=30, flor=True, signals=True)
