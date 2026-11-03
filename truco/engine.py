from __future__ import annotations

import random
from dataclasses import dataclass

from . import actions as A
from .cards import N_CARDS, STRENGTH, envido_points, flor_points, has_flor, sort_by_strength
from .rules import Rules

TRUCO_VALUES = (1, 2, 3, 4)
ENVIDO_CALL_VALUES = {A.ENVIDO: 2, A.REAL_ENVIDO: 3}
PARDA = -1


@dataclass(frozen=True)
class Pending:
    kind: str
    caller: int
    responder: int
    level: int = 0


class IllegalAction(ValueError):
    pass


class TrucoHand:
    def __init__(
        self,
        rules: Rules,
        rng: random.Random | None = None,
        scores: tuple[int, int] = (0, 0),
        mano: int = 0,
        hands: list[list[int]] | None = None,
        scramble_teams: tuple[int, ...] = (),
    ) -> None:
        self.rules = rules
        self.rng = rng if rng is not None else random.Random()
        self.n = rules.num_players
        self.scores = (int(scores[0]), int(scores[1]))
        self.mano = mano % self.n
        if hands is None:
            deck = list(range(N_CARDS))
            self.rng.shuffle(deck)
            hands = [deck[3 * i : 3 * i + 3] for i in range(self.n)]
        if len(hands) != self.n or any(len(h) != 3 for h in hands):
            raise ValueError("each player needs exactly three cards")
        self.hands = [sort_by_strength(list(h)) for h in hands]
        self.envido_of = [envido_points(h) for h in self.hands]
        self.flor_of = [has_flor(h) for h in self.hands]
        self._legal: list[int] | None = None
        self.scramble_teams = tuple(scramble_teams)
        self.played_slot = [[False, False, False] for _ in range(self.n)]
        self.table: list[list[int | None]] = [[None] * self.n for _ in range(3)]
        self.round = 0
        self.round_leader = self.mano
        self.round_order: list[int] = []
        self.round_winners: list[int] = []
        self.round_winner_seat: list[int] = []
        self.turn = self.mano
        self.truco_level = 0
        self.truco_holder: int | None = None
        self.envido_calls: list[int] = []
        self.envido_done = not rules.envido
        self.envido_value = 0
        self.envido_winner: int | None = None
        self.revealed_envido: list[int | None] = [None] * self.n
        self.flor_done = not rules.flor
        self.flor_winner: int | None = None
        self.pending: Pending | None = None
        self.suspended: Pending | None = None
        self.points = [0, 0]
        self.done = False
        self.winner_team: int | None = None
        self.end_reason: str | None = None
        self.history: list[tuple[int, int]] = []
        self.signal_phase = rules.signals
        self.signals_sent: list[int | None] = [None] * self.n
        self.signal_view: list[list[int | None]] = [[None] * self.n for _ in range(self.n)]
        self.signals_count = 0

    def team(self, seat: int) -> int:
        return seat % 2

    def next_opponent(self, seat: int) -> int:
        s = (seat + 1) % self.n
        while self.team(s) == self.team(seat):
            s = (s + 1) % self.n
        return s

    @property
    def current_player(self) -> int:
        if self.done:
            return -1
        if self.pending is not None:
            return self.pending.responder
        return self.turn

    def has_played_in_round(self, seat: int, rnd: int | None = None) -> bool:
        rnd = self.round if rnd is None else rnd
        return self.table[rnd][seat] is not None

    def remaining_cards(self, seat: int) -> list[int]:
        return [c for c, used in zip(self.hands[seat], self.played_slot[seat]) if not used]

    def falta_value(self) -> int:
        lead = max(self.scores[0] + self.points[0], self.scores[1] + self.points[1])
        return max(1, self.rules.target - lead)

    def envido_options(self) -> list[int]:
        calls = self.envido_calls
        if A.FALTA_ENVIDO in calls:
            return []
        options = []
        if calls.count(A.ENVIDO) < 2 and A.REAL_ENVIDO not in calls:
            options.append(A.ENVIDO)
        if A.REAL_ENVIDO not in calls:
            options.append(A.REAL_ENVIDO)
        options.append(A.FALTA_ENVIDO)
        return options

    def can_start_envido(self, seat: int) -> bool:
        if self.envido_done or self.round != 0 or self.truco_level != 0:
            return False
        if self.flor_winner is not None:
            return False
        if self.has_played_in_round(seat, 0):
            return False
        return self.pending is None or self.pending.kind == "truco"

    def can_declare_flor(self, seat: int) -> bool:
        if self.flor_done or self.round != 0 or self.has_played_in_round(seat, 0):
            return False
        return self.flor_of[seat]

    def can_call_truco(self, seat: int) -> bool:
        if self.pending is not None or self.truco_level >= 3:
            return False
        return self.truco_holder is None or self.truco_holder == self.team(seat)

    def legal_actions(self) -> list[int]:
        if self._legal is None:
            self._legal = self._compute_legal()
        return self._legal

    def _compute_legal(self) -> list[int]:
        if self.done:
            return []
        seat = self.current_player
        if self.signal_phase:
            return [A.signal_action(k) for k in range(self.rules.signal_vocab)]
        pending = self.pending
        if pending is not None:
            legal = [A.QUIERO, A.NO_QUIERO]
            if pending.kind == "truco":
                if pending.level < 3:
                    legal.append(A.TRUCO)
                if self.can_start_envido(seat):
                    legal.extend(self.envido_options())
            else:
                legal.extend(self.envido_options())
            if self.can_declare_flor(seat):
                legal.append(A.FLOR)
            return sorted(legal)
        legal = [slot for slot in range(3) if not self.played_slot[seat][slot]]
        if self.can_call_truco(seat):
            legal.append(A.TRUCO)
        if self.can_start_envido(seat):
            legal.extend(self.envido_options())
        if self.can_declare_flor(seat):
            legal.append(A.FLOR)
        legal.append(A.MAZO)
        return sorted(legal)

    def step(self, action: int) -> None:
        if self.done:
            raise IllegalAction("hand already finished")
        if action not in self.legal_actions():
            raise IllegalAction(f"illegal action {action}")
        self._legal = None
        seat = self.current_player
        self.history.append((seat, action))
        if self.signal_phase:
            self._signal(seat, A.signal_token(action))
            return
        pending = self.pending
        if action in A.PLAY_ACTIONS:
            self._play(seat, action)
        elif action == A.TRUCO:
            self._truco(seat, pending)
        elif action in A.ENVIDO_ACTIONS:
            self._envido(seat, action, pending)
        elif action == A.QUIERO:
            self._quiero(seat, pending)
        elif action == A.NO_QUIERO:
            self._no_quiero(pending)
        elif action == A.MAZO:
            self._finish(1 - self.team(seat), TRUCO_VALUES[self.truco_level], "mazo")
        elif action == A.FLOR:
            self._flor(seat)

    def _signal(self, seat: int, token: int) -> None:
        self.signals_sent[seat] = token
        for observer in range(self.n):
            if observer == seat:
                self.signal_view[observer][seat] = token
            elif self.team(observer) == self.team(seat):
                if self.team(seat) in self.scramble_teams:
                    self.signal_view[observer][seat] = self.rng.randrange(self.rules.signal_vocab)
                else:
                    self.signal_view[observer][seat] = token
            elif self.rng.random() < self.rules.spy_prob:
                self.signal_view[observer][seat] = token
        self.signals_count += 1
        if self.signals_count >= self.n:
            self.signal_phase = False
            self.turn = self.mano
        else:
            self.turn = (seat + 1) % self.n

    def _play(self, seat: int, slot: int) -> None:
        card = self.hands[seat][slot]
        self.played_slot[seat][slot] = True
        self.table[self.round][seat] = card
        self.round_order.append(seat)
        if len(self.round_order) < self.n:
            self.turn = (seat + 1) % self.n
            return
        self._close_round()

    def _close_round(self) -> None:
        played = [(STRENGTH[self.table[self.round][s]], s) for s in self.round_order]
        top = max(strength for strength, _ in played)
        top_seats = [s for strength, s in played if strength == top]
        teams = {self.team(s) for s in top_seats}
        winner = self.team(top_seats[0]) if len(teams) == 1 else PARDA
        self.round_winners.append(winner)
        self.round_winner_seat.append(top_seats[0])
        result = self._hand_result()
        if result is not None:
            self._finish(result, TRUCO_VALUES[self.truco_level], "cartas")
            return
        self.round += 1
        self.round_leader = top_seats[0]
        self.round_order = []
        self.turn = top_seats[0]

    def _hand_result(self) -> int | None:
        rw = self.round_winners
        if len(rw) < 2:
            return None
        if len(rw) == 2:
            first, second = rw
            if first != PARDA and first == second:
                return first
            if first == PARDA and second != PARDA:
                return second
            if first != PARDA and second == PARDA:
                return first
            return None
        first, second, third = rw
        if third != PARDA:
            return third
        if first != PARDA:
            return first
        if second != PARDA:
            return second
        return self.team(self.mano)

    def _truco(self, seat: int, pending: Pending | None) -> None:
        if pending is None:
            self.pending = Pending("truco", seat, self.next_opponent(seat), self.truco_level + 1)
            return
        self.truco_level = pending.level
        self.truco_holder = self.team(seat)
        self.pending = Pending("truco", seat, pending.caller, pending.level + 1)

    def _envido(self, seat: int, action: int, pending: Pending | None) -> None:
        if pending is None or pending.kind == "truco":
            self.suspended = pending
            self.envido_calls = [action]
            self.pending = Pending("envido", seat, self.next_opponent(seat))
            return
        self.envido_calls.append(action)
        self.pending = Pending("envido", seat, pending.caller)

    def _quiero(self, seat: int, pending: Pending) -> None:
        if pending.kind == "truco":
            self.truco_level = pending.level
            self.truco_holder = self.team(seat)
            self.pending = None
            return
        if A.FALTA_ENVIDO in self.envido_calls:
            value = self.falta_value()
        else:
            value = sum(ENVIDO_CALL_VALUES[c] for c in self.envido_calls)
        winner_seat = self.best_envido_seat()
        self.revealed_envido[winner_seat] = self.envido_of[winner_seat]
        self._close_envido(self.team(winner_seat), value)

    def _no_quiero(self, pending: Pending) -> None:
        if pending.kind == "truco":
            self._finish(self.team(pending.caller), TRUCO_VALUES[self.truco_level], "no_quiero")
            return
        previous = self.envido_calls[:-1]
        value = sum(ENVIDO_CALL_VALUES[c] for c in previous) if previous else 1
        self._close_envido(self.team(pending.caller), value)

    def _close_envido(self, team: int, value: int) -> None:
        self.envido_done = True
        self.envido_value = value
        self.envido_winner = team
        self.pending = self.suspended
        self.suspended = None
        self._award(team, value, "envido")

    def best_envido_seat(self) -> int:
        order = [(self.mano + i) % self.n for i in range(self.n)]
        best_seat = order[0]
        best_value = self.envido_of[best_seat]
        for s in order[1:]:
            value = self.envido_of[s]
            if value > best_value:
                best_seat, best_value = s, value
        return best_seat

    def best_flor_seat(self) -> tuple[int | None, int | None]:
        order = [(self.mano + i) % self.n for i in range(self.n)]
        best: list[int | None] = [None, None]
        best_value = [-1, -1]
        for s in order:
            if has_flor(self.hands[s]):
                t = self.team(s)
                value = flor_points(self.hands[s])
                if value > best_value[t]:
                    best[t], best_value[t] = s, value
        return best[0], best[1]

    def _flor(self, seat: int) -> None:
        team = self.team(seat)
        best = self.best_flor_seat()
        rival = best[1 - team]
        self.flor_done = True
        self.envido_done = True
        if self.pending is not None and self.pending.kind == "envido":
            self.pending = self.suspended
            self.suspended = None
            self.envido_calls = []
        if rival is None:
            self.flor_winner = team
            self._award(team, 3, "flor")
            return
        mine = best[team]
        order = [(self.mano + i) % self.n for i in range(self.n)]
        mine_key = (flor_points(self.hands[mine]), -order.index(mine))
        rival_key = (flor_points(self.hands[rival]), -order.index(rival))
        winner = team if mine_key > rival_key else 1 - team
        self.flor_winner = winner
        self._award(winner, 6, "flor")

    def _award(self, team: int, value: int, reason: str) -> None:
        if self.done:
            return
        self.points[team] += value
        if self.scores[team] + self.points[team] >= self.rules.target:
            self.done = True
            self.winner_team = team
            self.end_reason = f"target_{reason}"
            self.pending = None

    def _finish(self, team: int, value: int, reason: str) -> None:
        self._award(team, value, reason)
        if not self.done:
            self.done = True
            self.winner_team = team
            self.end_reason = reason
        self.pending = None

    def team_reward(self, team: int) -> int:
        return self.points[team] - self.points[1 - team]

    def clone(self) -> "TrucoHand":
        other = TrucoHand.__new__(TrucoHand)
        other.__dict__.update(self.__dict__)
        other.rng = random.Random(self.rng.random())
        other.hands = [list(h) for h in self.hands]
        other.envido_of = list(self.envido_of)
        other.flor_of = list(self.flor_of)
        other._legal = None
        other.played_slot = [list(p) for p in self.played_slot]
        other.table = [list(r) for r in self.table]
        other.round_order = list(self.round_order)
        other.round_winners = list(self.round_winners)
        other.round_winner_seat = list(self.round_winner_seat)
        other.envido_calls = list(self.envido_calls)
        other.revealed_envido = list(self.revealed_envido)
        other.points = list(self.points)
        other.history = list(self.history)
        other.signals_sent = list(self.signals_sent)
        other.signal_view = [list(v) for v in self.signal_view]
        return other


class TrucoMatch:
    def __init__(self, rules: Rules, rng: random.Random | None = None, first_mano: int = 0) -> None:
        self.rules = rules
        self.rng = rng if rng is not None else random.Random()
        self.scores = [0, 0]
        self.mano = first_mano % rules.num_players
        self.hands_played = 0
        self.hand: TrucoHand | None = None

    @property
    def done(self) -> bool:
        return max(self.scores) >= self.rules.target

    @property
    def winner(self) -> int | None:
        if not self.done:
            return None
        return 0 if self.scores[0] >= self.rules.target else 1

    def new_hand(self, scramble_teams: tuple[int, ...] = ()) -> TrucoHand:
        self.hand = TrucoHand(
            self.rules,
            rng=random.Random(self.rng.random()),
            scores=(self.scores[0], self.scores[1]),
            mano=self.mano,
            scramble_teams=scramble_teams,
        )
        return self.hand

    def close_hand(self) -> None:
        if self.hand is None or not self.hand.done:
            raise RuntimeError("current hand is not finished")
        for t in (0, 1):
            self.scores[t] += self.hand.points[t]
        self.hands_played += 1
        self.mano = (self.mano + 1) % self.rules.num_players
        self.hand = None
