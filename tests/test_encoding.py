import random

import numpy as np

from truco import actions as A
from truco.cards import MAX_ENVIDO, MAX_STRENGTH, N_CARDS, STRENGTH, envido_points, has_flor
from truco.encoding import encode, layout_for, obs_size
from truco.engine import PARDA, TrucoHand
from truco.rules import Rules


def _one_hot(index, size):
    v = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size:
        v[index] = 1.0
    return v


def reference_encode(hand, seat):
    rules = hand.rules
    n = hand.n
    me = hand.team(seat)
    target = float(rules.target)
    parts = []
    slots = np.zeros((3, N_CARDS), dtype=np.float32)
    strengths = np.zeros(3, dtype=np.float32)
    for i, card in enumerate(hand.hands[seat]):
        if not hand.played_slot[seat][i]:
            slots[i, card] = 1.0
            strengths[i] = STRENGTH[card] / MAX_STRENGTH
    parts += [slots.ravel(), strengths]
    parts.append(np.array([envido_points(hand.hands[seat]) / MAX_ENVIDO, float(has_flor(hand.hands[seat]))], dtype=np.float32))
    table = np.zeros((3, n, N_CARDS), dtype=np.float32)
    for r in range(3):
        for s in range(n):
            card = hand.table[r][s]
            if card is not None:
                table[r, (s - seat) % n, card] = 1.0
    parts.append(table.ravel())
    rounds = np.zeros((3, 3), dtype=np.float32)
    for r, w in enumerate(hand.round_winners):
        rounds[r, 2 if w == PARDA else (0 if w == me else 1)] = 1.0
    parts.append(rounds.ravel())
    parts.append(_one_hot(min(hand.round, 2), 3))
    parts.append(_one_hot((hand.mano - seat) % n, n))
    parts.append(_one_hot((hand.round_leader - seat) % n, n))
    parts.append(_one_hot(hand.truco_level, 4))
    holder = 0 if hand.truco_holder is None else (1 if hand.truco_holder == me else 2)
    parts.append(_one_hot(holder, 3))
    pending = hand.pending
    kind = 0 if pending is None else (1 if pending.kind == "truco" else 2)
    parts.append(_one_hot(kind, 3))
    parts.append(_one_hot(pending.level if pending is not None and pending.kind == "truco" else None, 4))
    parts.append(np.array([float(pending is not None and pending.responder == seat)], dtype=np.float32))
    parts.append(_one_hot((pending.caller - seat) % n if pending is not None else None, n))
    calls = hand.envido_calls
    parts.append(
        np.array(
            [calls.count(A.ENVIDO) / 2.0, float(A.REAL_ENVIDO in calls), float(A.FALTA_ENVIDO in calls), float(hand.envido_done), float(hand.flor_done)],
            dtype=np.float32,
        )
    )
    revealed = np.zeros((n, 2), dtype=np.float32)
    for s in range(n):
        value = hand.revealed_envido[s]
        if value is not None:
            revealed[(s - seat) % n] = (value / MAX_ENVIDO, 1.0)
    parts.append(revealed.ravel())
    parts.append(
        np.array(
            [hand.scores[me] / target, hand.scores[1 - me] / target, hand.falta_value() / target, hand.points[me] / target, hand.points[1 - me] / target],
            dtype=np.float32,
        )
    )
    if rules.signals:
        k = rules.signal_vocab
        parts.append(np.array([float(hand.signal_phase)], dtype=np.float32))
        sig = np.zeros((n, k + 1), dtype=np.float32)
        for s in range(n):
            token = hand.signal_view[seat][s]
            sig[(s - seat) % n, k if token is None else token] = 1.0
        parts.append(sig.ravel())
    return np.concatenate(parts)


def test_fast_encoding_matches_reference():
    rng = random.Random(0)
    for rules in (Rules(num_players=4, signals=True), Rules(num_players=2, signals=False), Rules(num_players=4, flor=True, signals=True)):
        assert obs_size(rules) == reference_encode(TrucoHand(rules), 0).shape[0]
        for _ in range(60):
            hand = TrucoHand(rules, rng=random.Random(rng.random()), scores=(rng.randrange(30), rng.randrange(30)), mano=rng.randrange(rules.num_players))
            while not hand.done:
                for seat in range(rules.num_players):
                    assert np.array_equal(encode(hand, seat), reference_encode(hand, seat))
                hand.step(rng.choice(hand.legal_actions()))


def test_encode_into_buffer():
    rules = Rules()
    hand = TrucoHand(rules, rng=random.Random(1))
    buf = np.ones(layout_for(rules).size, dtype=np.float32)
    out = encode(hand, 0, out=buf)
    assert out is buf
    assert np.array_equal(buf, reference_encode(hand, 0))
