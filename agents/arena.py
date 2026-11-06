from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

import numpy as np

from truco.engine import TrucoHand, TrucoMatch
from truco.rules import Rules

from .base import Agent
from .stats import wilson_interval


@dataclass
class MatchStats:
    matches: int
    wins_a: int
    points_a: int
    points_b: int
    hands: int
    outcomes: list[int] = field(default_factory=list, repr=False)
    diffs: list[int] = field(default_factory=list, repr=False)
    lengths: list[int] = field(default_factory=list, repr=False)

    @property
    def win_rate(self) -> float:
        return self.wins_a / self.matches if self.matches else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        if self.matches == 0:
            return (0.0, 0.0)
        return wilson_interval(self.wins_a, self.matches)

    @property
    def point_diff_per_hand(self) -> float:
        return (self.points_a - self.points_b) / self.hands if self.hands else 0.0

    def point_diff_ci95(self, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
        if len(self.diffs) < 2:
            return (self.point_diff_per_hand, self.point_diff_per_hand)
        diffs = np.asarray(self.diffs, dtype=np.float64)
        lengths = np.asarray(self.lengths, dtype=np.float64)
        idx = np.random.default_rng(seed).integers(0, len(diffs), size=(samples, len(diffs)))
        ratios = diffs[idx].sum(1) / np.maximum(lengths[idx].sum(1), 1.0)
        return (float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975)))

    def to_dict(self) -> dict:
        lo, hi = self.ci95
        out = {
            "matches": self.matches,
            "win_rate": round(self.win_rate, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "point_diff_per_hand": round(self.point_diff_per_hand, 4),
            "hands": self.hands,
        }
        if self.diffs:
            plo, phi = self.point_diff_ci95()
            out["point_diff_ci95"] = [round(plo, 4), round(phi, 4)]
        return out


@dataclass(frozen=True)
class SignalNoise:
    partner_a: bool = False
    partner_b: bool = False
    rivals_a: bool = False
    rivals_b: bool = False
    blind_a: bool = False
    blind_b: bool = False
    dist_a: tuple[float, ...] | None = None
    dist_b: tuple[float, ...] | None = None

    def for_match(self, a_team: int) -> dict:
        b_team = 1 - a_team

        def teams(flag_a: bool, flag_b: bool) -> tuple[int, ...]:
            return tuple(t for t, flag in ((a_team, flag_a), (b_team, flag_b)) if flag)

        noise = [None, None]
        noise[a_team] = self.dist_a
        noise[b_team] = self.dist_b
        return {
            "scramble_teams": teams(self.partner_a, self.partner_b),
            "scramble_rival_teams": teams(self.rivals_a, self.rivals_b),
            "blind_teams": teams(self.blind_a, self.blind_b),
            "noise": None if noise == [None, None] else (noise[0], noise[1]),
        }


def play_hand(hand: TrucoHand, seats: list[Agent]) -> TrucoHand:
    while not hand.done:
        seat = hand.current_player
        hand.step(seats[seat].act(hand, seat))
    return hand


def seat_agents(rules: Rules, team_a: list[Agent], team_b: list[Agent], a_is_team: int) -> list[Agent]:
    seats: list[Agent] = []
    ia = ib = 0
    for seat in range(rules.num_players):
        if seat % 2 == a_is_team:
            seats.append(team_a[ia % len(team_a)])
            ia += 1
        else:
            seats.append(team_b[ib % len(team_b)])
            ib += 1
    return seats


def head_to_head(
    rules: Rules,
    team_a: list[Agent],
    team_b: list[Agent],
    matches: int,
    seed: int = 0,
    scramble_a: bool = False,
    scramble_b: bool = False,
    noise: SignalNoise | None = None,
    duplicate: bool = False,
) -> MatchStats:
    if noise is None:
        noise = SignalNoise(partner_a=scramble_a, partner_b=scramble_b)
    elif scramble_a or scramble_b:
        noise = replace(noise, partner_a=noise.partner_a or scramble_a, partner_b=noise.partner_b or scramble_b)
    rng = random.Random(seed)
    stats = MatchStats(matches, 0, 0, 0, 0)
    for m in range(matches):
        a_team = m % 2
        seats = seat_agents(rules, team_a, team_b, a_team)
        options = noise.for_match(a_team)
        if not duplicate or a_team == 0:
            match_seed = rng.random()
            first_mano = rng.randrange(rules.num_players)
        match = TrucoMatch(rules, rng=random.Random(match_seed), first_mano=first_mano)
        for agent in seats:
            agent.reset()
        diff = 0
        length = 0
        while not match.done:
            hand = match.new_hand(**options)
            play_hand(hand, seats)
            stats.points_a += hand.points[a_team]
            stats.points_b += hand.points[1 - a_team]
            diff += hand.points[a_team] - hand.points[1 - a_team]
            length += 1
            match.close_hand()
        won = int(match.winner == a_team)
        stats.wins_a += won
        stats.hands += length
        stats.outcomes.append(won)
        stats.diffs.append(diff)
        stats.lengths.append(length)
    return stats
