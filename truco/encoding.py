from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from . import actions as A
from .cards import MAX_ENVIDO, MAX_STRENGTH, N_CARDS, STRENGTH
from .engine import PARDA, TrucoHand
from .rules import Rules


@dataclass(frozen=True)
class Layout:
    n: int
    k: int
    signals: bool
    slots: int
    strengths: int
    own: int
    table: int
    rounds: int
    round_idx: int
    mano: int
    leader: int
    truco: int
    holder: int
    pending_kind: int
    pending_level: int
    responder: int
    caller: int
    envido: int
    revealed: int
    scores: int
    signal_phase: int
    signal_view: int
    size: int


@cache
def layout(n: int, k: int, signals: bool) -> Layout:
    sizes = [
        ("slots", 3 * N_CARDS),
        ("strengths", 3),
        ("own", 2),
        ("table", 3 * n * N_CARDS),
        ("rounds", 9),
        ("round_idx", 3),
        ("mano", n),
        ("leader", n),
        ("truco", 4),
        ("holder", 3),
        ("pending_kind", 3),
        ("pending_level", 4),
        ("responder", 1),
        ("caller", n),
        ("envido", 5),
        ("revealed", 2 * n),
        ("scores", 5),
        ("signal_phase", 1 if signals else 0),
        ("signal_view", n * (k + 1) if signals else 0),
    ]
    offsets = {}
    pos = 0
    for name, size in sizes:
        offsets[name] = pos
        pos += size
    return Layout(n=n, k=k, signals=signals, size=pos, **offsets)


def layout_for(rules: Rules) -> Layout:
    return layout(rules.num_players, rules.signal_vocab, rules.signals)


def encode(hand: TrucoHand, seat: int, out: np.ndarray | None = None, signals: bool | None = None) -> np.ndarray:
    rules = hand.rules
    L = layout(hand.n, rules.signal_vocab, rules.signals if signals is None else signals)
    n = hand.n
    if out is None:
        v = np.zeros(L.size, dtype=np.float32)
    else:
        v = out
        v.fill(0.0)
    me = seat % 2
    target = float(rules.target)

    cards = hand.hands[seat]
    played = hand.played_slot[seat]
    for i in range(3):
        if not played[i]:
            c = cards[i]
            v[L.slots + i * N_CARDS + c] = 1.0
            v[L.strengths + i] = STRENGTH[c] / MAX_STRENGTH
    v[L.own] = hand.envido_of[seat] / MAX_ENVIDO
    v[L.own + 1] = float(hand.flor_of[seat])

    for r in range(hand.round + 1 if hand.round < 3 else 3):
        row = hand.table[r]
        for s in range(n):
            c = row[s]
            if c is not None:
                v[L.table + (r * n + (s - seat) % n) * N_CARDS + c] = 1.0

    for r, w in enumerate(hand.round_winners):
        v[L.rounds + 3 * r + (2 if w == PARDA else (0 if w == me else 1))] = 1.0

    v[L.round_idx + min(hand.round, 2)] = 1.0
    v[L.mano + (hand.mano - seat) % n] = 1.0
    v[L.leader + (hand.round_leader - seat) % n] = 1.0
    v[L.truco + hand.truco_level] = 1.0
    holder = hand.truco_holder
    v[L.holder + (0 if holder is None else (1 if holder == me else 2))] = 1.0

    pending = hand.pending
    if pending is None:
        v[L.pending_kind] = 1.0
    else:
        if pending.kind == "truco":
            v[L.pending_kind + 1] = 1.0
            if 0 <= pending.level < 4:
                v[L.pending_level + pending.level] = 1.0
        else:
            v[L.pending_kind + 2] = 1.0
        v[L.responder] = float(pending.responder == seat)
        v[L.caller + (pending.caller - seat) % n] = 1.0

    calls = hand.envido_calls
    if calls:
        v[L.envido] = calls.count(A.ENVIDO) / 2.0
        v[L.envido + 1] = float(A.REAL_ENVIDO in calls)
        v[L.envido + 2] = float(A.FALTA_ENVIDO in calls)
    v[L.envido + 3] = float(hand.envido_done)
    v[L.envido + 4] = float(hand.flor_done)

    for s in range(n):
        value = hand.revealed_envido[s]
        if value is not None:
            base = L.revealed + 2 * ((s - seat) % n)
            v[base] = value / MAX_ENVIDO
            v[base + 1] = 1.0

    v[L.scores] = hand.scores[me] / target
    v[L.scores + 1] = hand.scores[1 - me] / target
    v[L.scores + 2] = hand.falta_value() / target
    v[L.scores + 3] = hand.points[me] / target
    v[L.scores + 4] = hand.points[1 - me] / target

    if L.signals:
        k = L.k
        v[L.signal_phase] = float(hand.signal_phase)
        view = hand.signal_view[seat]
        for s in range(n):
            token = view[s]
            v[L.signal_view + ((s - seat) % n) * (k + 1) + (k if token is None else token)] = 1.0

    return v


def obs_size(rules: Rules) -> int:
    return layout_for(rules).size


def action_mask(hand: TrucoHand, n_actions: int, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        mask = np.zeros(n_actions, dtype=bool)
    else:
        mask = out
        mask.fill(False)
    mask[hand.legal_actions()] = True
    return mask
