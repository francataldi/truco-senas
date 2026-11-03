from __future__ import annotations

import math
import random
from dataclasses import dataclass

from truco.engine import TrucoHand, TrucoMatch
from truco.rules import Rules

from .base import Agent


@dataclass
class MatchStats:
    matches: int
    wins_a: int
    points_a: int
    points_b: int
    hands: int

    @property
    def win_rate(self) -> float:
        return self.wins_a / self.matches if self.matches else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        n = self.matches
        if n == 0:
            return (0.0, 0.0)
        p = self.win_rate
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (center - half, center + half)

    @property
    def point_diff_per_hand(self) -> float:
        return (self.points_a - self.points_b) / self.hands if self.hands else 0.0

    def to_dict(self) -> dict:
        lo, hi = self.ci95
        return {
            "matches": self.matches,
            "win_rate": round(self.win_rate, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "point_diff_per_hand": round(self.point_diff_per_hand, 4),
            "hands": self.hands,
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
) -> MatchStats:
    rng = random.Random(seed)
    wins = points_a = points_b = hands = 0
    for m in range(matches):
        a_team = m % 2
        seats = seat_agents(rules, team_a, team_b, a_team)
        scramble = tuple(t for t, flag in ((a_team, scramble_a), (1 - a_team, scramble_b)) if flag)
        match = TrucoMatch(rules, rng=random.Random(rng.random()), first_mano=rng.randrange(rules.num_players))
        for agent in seats:
            agent.reset()
        while not match.done:
            hand = match.new_hand(scramble_teams=scramble)
            play_hand(hand, seats)
            points_a += hand.points[a_team]
            points_b += hand.points[1 - a_team]
            hands += 1
            match.close_hand()
        wins += match.winner == a_team
    return MatchStats(matches, wins, points_a, points_b, hands)
