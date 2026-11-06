from __future__ import annotations

import random

from truco import actions as A
from truco.cards import STRENGTH, Card, envido_points
from truco.engine import PARDA, TrucoHand

from .base import Agent

ANCHO_ESPADA = Card(1, "espada").index
ANCHO_BASTO = Card(1, "basto").index
SIETE_ESPADA = Card(7, "espada").index
SIETE_ORO = Card(7, "oro").index

CLASSIC_SIGNALS = {
    ANCHO_ESPADA: 0,
    ANCHO_BASTO: 1,
    SIETE_ESPADA: 2,
    SIETE_ORO: 3,
}
SIGNAL_TRES = 4
SIGNAL_ENVIDO = 5
SIGNAL_NADA = 6
SIGNAL_DOS = 7


def classic_signal(cards: list[int], vocab: int) -> int:
    best = max(cards, key=lambda x: STRENGTH[x])
    if best in CLASSIC_SIGNALS:
        token = CLASSIC_SIGNALS[best]
    elif envido_points(cards) >= 30:
        token = SIGNAL_ENVIDO
    elif STRENGTH[best] == 10:
        token = SIGNAL_TRES
    elif STRENGTH[best] == 9:
        token = SIGNAL_DOS
    else:
        token = SIGNAL_NADA
    return token % vocab


def mute_signal(vocab: int) -> int:
    return SIGNAL_NADA % vocab


def _card_value(strength: int) -> float:
    return max(0.0, (strength - 6) / 8.0)


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, seed: int | None = None, bluff: float = 0.05, use_signals: bool = True) -> None:
        self.rng = random.Random(seed)
        self.bluff = bluff
        self.use_signals = use_signals

    def act(self, hand: TrucoHand, seat: int) -> int:
        legal = hand.legal_actions()
        if hand.signal_phase:
            token = classic_signal(hand.hands[seat], hand.rules.signal_vocab) if self.use_signals else mute_signal(hand.rules.signal_vocab)
            return A.signal_action(token)
        if A.FLOR in legal:
            return A.FLOR
        pending = hand.pending
        envido = envido_points(hand.hands[seat])
        if pending is not None and pending.kind == "envido":
            return self._answer_envido(hand, legal, envido)
        if pending is not None and pending.kind == "truco":
            if A.ENVIDO in legal and envido >= 29:
                return A.REAL_ENVIDO if envido >= 31 and A.REAL_ENVIDO in legal else A.ENVIDO
            return self._answer_truco(hand, seat, legal)
        if A.ENVIDO in legal or A.REAL_ENVIDO in legal:
            call = self._start_envido(legal, envido)
            if call is not None:
                return call
        if A.TRUCO in legal and self._should_call_truco(hand, seat):
            return A.TRUCO
        return self._play(hand, seat, legal)

    def _start_envido(self, legal: list[int], envido: int) -> int | None:
        if envido >= 31 and A.REAL_ENVIDO in legal:
            return A.REAL_ENVIDO
        if envido >= 27 and A.ENVIDO in legal and self.rng.random() < 0.85:
            return A.ENVIDO
        if envido < 20 and A.ENVIDO in legal and self.rng.random() < self.bluff:
            return A.ENVIDO
        return None

    def _answer_envido(self, hand: TrucoHand, legal: list[int], envido: int) -> int:
        falta = A.FALTA_ENVIDO in hand.envido_calls
        if envido >= 33 and A.FALTA_ENVIDO in legal:
            return A.FALTA_ENVIDO
        if envido >= 31 and A.REAL_ENVIDO in legal:
            return A.REAL_ENVIDO
        threshold = 30 if falta else 27
        if envido >= threshold:
            return A.QUIERO
        return A.NO_QUIERO

    def _strength(self, hand: TrucoHand, seat: int) -> float:
        me = hand.team(seat)
        score = sum(_card_value(STRENGTH[c]) for c in hand.remaining_cards(seat))
        for w in hand.round_winners:
            if w == me:
                score += 1.0
            elif w != PARDA:
                score -= 1.0
        if self.use_signals and hand.n == 4:
            partner = (seat + 2) % 4
            token = hand.signal_view[seat][partner]
            if token is not None and token in (0, 1, 2, 3):
                score += 0.7
            elif token == SIGNAL_TRES:
                score += 0.4
        return score

    def _should_call_truco(self, hand: TrucoHand, seat: int) -> bool:
        s = self._strength(hand, seat)
        if s >= 1.6:
            return True
        return self.rng.random() < self.bluff

    def _answer_truco(self, hand: TrucoHand, seat: int, legal: list[int]) -> int:
        s = self._strength(hand, seat)
        level = hand.pending.level if hand.pending is not None else 1
        if s >= 1.4 + 0.4 * level and A.TRUCO in legal:
            return A.TRUCO
        if s >= 0.6 + 0.25 * level:
            return A.QUIERO
        if self.rng.random() < self.bluff:
            return A.QUIERO
        return A.NO_QUIERO

    def _play(self, hand: TrucoHand, seat: int, legal: list[int]) -> int:
        slots = [a for a in legal if a in A.PLAY_ACTIONS]
        if not slots:
            return legal[0]
        by_strength = sorted(slots, key=lambda slot: STRENGTH[hand.hands[seat][slot]])
        lowest, highest = by_strength[0], by_strength[-1]
        played = [(STRENGTH[hand.table[hand.round][s]], s) for s in hand.round_order]
        if not played:
            if hand.round == 0 and len(by_strength) >= 2:
                return by_strength[-2] if hand.n == 4 else highest
            return highest
        best_strength = max(st for st, _ in played)
        best_seats = [s for st, s in played if st == best_strength]
        mine_winning = all(hand.team(s) == hand.team(seat) for s in best_seats)
        if mine_winning:
            return lowest
        for slot in by_strength:
            if STRENGTH[hand.hands[seat][slot]] > best_strength:
                return slot
        return lowest
