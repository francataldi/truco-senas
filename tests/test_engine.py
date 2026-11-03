import random

import pytest

from truco import actions as A
from truco.cards import STRENGTH, Card, envido_points, flor_points, has_flor
from truco.engine import PARDA, IllegalAction, TrucoHand, TrucoMatch
from truco.rules import Rules

HU = Rules(num_players=2, target=30, flor=False, signals=False)
PAREJAS = Rules(num_players=4, target=30, flor=False, signals=True, spy_prob=0.0)


def c(rank, suit):
    return Card(rank, suit).index


def deal_hu(mano_cards, other_cards, mano=0, scores=(0, 0), rules=HU):
    hands = [None, None]
    hands[mano] = mano_cards
    hands[1 - mano] = other_cards
    return TrucoHand(rules, rng=random.Random(0), scores=scores, mano=mano, hands=hands)


def test_strength_order():
    order = [
        c(1, "espada"),
        c(1, "basto"),
        c(7, "espada"),
        c(7, "oro"),
        c(3, "copa"),
        c(2, "oro"),
        c(1, "copa"),
        c(12, "basto"),
        c(11, "oro"),
        c(10, "copa"),
        c(7, "copa"),
        c(6, "espada"),
        c(5, "oro"),
        c(4, "basto"),
    ]
    values = [STRENGTH[x] for x in order]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values) == 14


def test_envido_points():
    assert envido_points([c(7, "oro"), c(6, "oro"), c(12, "copa")]) == 33
    assert envido_points([c(12, "oro"), c(11, "oro"), c(3, "copa")]) == 20
    assert envido_points([c(7, "espada"), c(5, "basto"), c(4, "copa")]) == 7
    assert envido_points([c(10, "espada"), c(11, "basto"), c(12, "copa")]) == 0
    assert envido_points([c(7, "oro"), c(6, "oro"), c(5, "oro")]) == 33


def test_flor():
    cards = [c(7, "oro"), c(6, "oro"), c(5, "oro")]
    assert has_flor(cards)
    assert flor_points(cards) == 38
    assert not has_flor([c(7, "oro"), c(6, "oro"), c(5, "copa")])


def test_hands_are_sorted_by_strength():
    hand = deal_hu([c(4, "copa"), c(1, "espada"), c(3, "oro")], [c(5, "oro"), c(6, "oro"), c(7, "copa")])
    assert hand.hands[0] == [c(1, "espada"), c(3, "oro"), c(4, "copa")]


def test_two_rounds_win_the_hand():
    hand = deal_hu([c(1, "espada"), c(1, "basto"), c(4, "copa")], [c(3, "oro"), c(2, "oro"), c(5, "copa")])
    hand.step(A.PLAY_0)
    hand.step(A.PLAY_0)
    assert hand.round_winners == [0]
    assert hand.current_player == 0
    hand.step(A.PLAY_1)
    hand.step(A.PLAY_1)
    assert hand.done
    assert hand.winner_team == 0
    assert hand.points == [1, 0]


def test_parda_first_round_then_second_decides():
    hand = deal_hu([c(3, "oro"), c(4, "copa"), c(5, "copa")], [c(3, "copa"), c(6, "oro"), c(4, "oro")])
    hand.step(A.PLAY_0)
    hand.step(A.PLAY_0)
    assert hand.round_winners == [PARDA]
    hand.step(A.PLAY_1)
    hand.step(A.PLAY_1)
    assert hand.done and hand.winner_team == 1


def test_win_first_parda_second():
    hand = deal_hu([c(1, "espada"), c(4, "copa"), c(5, "copa")], [c(3, "copa"), c(4, "oro"), c(6, "oro")])
    hand.step(A.PLAY_0)
    hand.step(A.PLAY_0)
    hand.step(A.PLAY_2)
    hand.step(A.PLAY_2)
    assert hand.round_winners == [0, PARDA]
    assert hand.done and hand.winner_team == 0


def test_split_rounds_and_parda_third_goes_to_first_winner():
    hand = deal_hu([c(1, "espada"), c(4, "copa"), c(5, "copa")], [c(3, "copa"), c(2, "oro"), c(4, "oro")])
    hand.step(A.PLAY_0)
    hand.step(A.PLAY_0)
    hand.step(A.PLAY_1)
    hand.step(A.PLAY_1)
    assert hand.round_winners == [0, 1]
    assert not hand.done
    assert hand.current_player == 1
    hand.step(A.PLAY_2)
    hand.step(A.PLAY_2)
    assert hand.round_winners == [0, 1, PARDA]
    assert hand.winner_team == 0


def test_all_pardas_mano_wins():
    hand = deal_hu(
        [c(3, "oro"), c(2, "oro"), c(4, "oro")],
        [c(3, "copa"), c(2, "copa"), c(4, "copa")],
        mano=1,
    )
    for slot in (A.PLAY_0, A.PLAY_1, A.PLAY_2):
        hand.step(slot)
        hand.step(slot)
    assert hand.round_winners == [PARDA, PARDA, PARDA]
    assert hand.winner_team == 1


def test_truco_rejected_gives_one_point():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "copa")], [c(1, "espada"), c(2, "oro"), c(3, "oro")])
    hand.step(A.TRUCO)
    assert hand.current_player == 1
    hand.step(A.NO_QUIERO)
    assert hand.done and hand.points == [1, 0]


def test_retruco_rejected_gives_two_points():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "copa")], [c(1, "espada"), c(2, "oro"), c(3, "oro")])
    hand.step(A.TRUCO)
    hand.step(A.TRUCO)
    assert hand.truco_level == 1
    assert hand.current_player == 0
    hand.step(A.NO_QUIERO)
    assert hand.points == [0, 2]


def test_truco_accepted_raises_hand_value():
    hand = deal_hu([c(1, "espada"), c(1, "basto"), c(4, "copa")], [c(3, "oro"), c(2, "oro"), c(5, "copa")])
    hand.step(A.TRUCO)
    hand.step(A.QUIERO)
    assert hand.truco_level == 1 and hand.truco_holder == 1
    assert A.TRUCO not in hand.legal_actions()
    hand.step(A.PLAY_0)
    assert A.TRUCO in hand.legal_actions()
    hand.step(A.PLAY_0)
    hand.step(A.PLAY_1)
    hand.step(A.PLAY_1)
    assert hand.points == [2, 0]


def test_vale_cuatro_is_the_cap():
    hand = deal_hu([c(1, "espada"), c(1, "basto"), c(4, "copa")], [c(3, "oro"), c(2, "oro"), c(5, "copa")])
    hand.step(A.TRUCO)
    hand.step(A.TRUCO)
    hand.step(A.TRUCO)
    assert A.TRUCO not in hand.legal_actions()
    hand.step(A.QUIERO)
    assert hand.truco_level == 3


def test_envido_accepted_goes_to_best_hand():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "oro")], [c(7, "oro"), c(6, "espada"), c(5, "oro")])
    hand.step(A.ENVIDO)
    hand.step(A.QUIERO)
    assert hand.envido_done
    assert hand.points == [0, 2]
    assert hand.revealed_envido[1] == 32
    assert hand.current_player == 0


def test_envido_tie_goes_to_mano():
    hand = deal_hu([c(7, "copa"), c(6, "copa"), c(4, "oro")], [c(7, "oro"), c(6, "oro"), c(4, "copa")], mano=1)
    hand.step(A.ENVIDO)
    hand.step(A.QUIERO)
    assert hand.points == [0, 2]


def test_envido_envido_real_rejected():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "oro")], [c(7, "oro"), c(6, "espada"), c(5, "oro")])
    hand.step(A.ENVIDO)
    hand.step(A.ENVIDO)
    assert A.ENVIDO not in hand.legal_actions()
    hand.step(A.REAL_ENVIDO)
    assert hand.current_player == 1
    hand.step(A.NO_QUIERO)
    assert hand.points == [4, 0]


def test_first_envido_rejected_is_one_point():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "oro")], [c(7, "oro"), c(6, "espada"), c(5, "oro")])
    hand.step(A.REAL_ENVIDO)
    hand.step(A.NO_QUIERO)
    assert hand.points == [1, 0]


def test_falta_envido_value():
    hand = deal_hu(
        [c(7, "oro"), c(6, "oro"), c(4, "copa")],
        [c(1, "espada"), c(2, "basto"), c(3, "copa")],
        scores=(20, 12),
    )
    hand.step(A.FALTA_ENVIDO)
    hand.step(A.QUIERO)
    assert hand.points[0] == 10
    assert hand.done and hand.winner_team == 0


def test_envido_primero_keeps_truco_pending():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "oro")], [c(7, "oro"), c(6, "espada"), c(5, "oro")])
    hand.step(A.TRUCO)
    assert A.ENVIDO in hand.legal_actions()
    hand.step(A.ENVIDO)
    assert hand.current_player == 0
    hand.step(A.QUIERO)
    assert hand.points == [0, 2]
    assert hand.pending is not None and hand.pending.kind == "truco"
    assert hand.current_player == 1
    hand.step(A.QUIERO)
    assert hand.truco_level == 1


def test_no_envido_after_playing_or_truco_accepted():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "oro")], [c(7, "oro"), c(6, "espada"), c(5, "oro")])
    hand.step(A.PLAY_0)
    assert A.ENVIDO in hand.legal_actions()
    hand.step(A.TRUCO)
    hand.step(A.QUIERO)
    assert A.ENVIDO not in hand.legal_actions()


def test_mazo_gives_current_value():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "oro")], [c(7, "oro"), c(6, "espada"), c(5, "oro")])
    hand.step(A.TRUCO)
    hand.step(A.QUIERO)
    hand.step(A.MAZO)
    assert hand.points == [0, 2]


def test_illegal_action_raises():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "oro")], [c(7, "oro"), c(6, "espada"), c(5, "oro")])
    with pytest.raises(IllegalAction):
        hand.step(A.QUIERO)


def test_flor_scores_three():
    rules = Rules(num_players=2, target=30, flor=True, signals=False)
    hand = deal_hu([c(7, "oro"), c(6, "oro"), c(5, "oro")], [c(1, "espada"), c(2, "basto"), c(3, "copa")], rules=rules)
    assert A.FLOR in hand.legal_actions()
    hand.step(A.FLOR)
    assert hand.points == [3, 0]
    assert A.ENVIDO not in hand.legal_actions()


def test_contraflor_best_flor_takes_six():
    rules = Rules(num_players=2, target=30, flor=True, signals=False)
    hand = deal_hu([c(4, "oro"), c(5, "oro"), c(6, "oro")], [c(7, "copa"), c(6, "copa"), c(5, "copa")], rules=rules)
    hand.step(A.FLOR)
    assert hand.points == [0, 6]


def test_flor_answers_envido():
    rules = Rules(num_players=2, target=30, flor=True, signals=False)
    hand = deal_hu([c(4, "copa"), c(1, "espada"), c(2, "basto")], [c(7, "oro"), c(6, "oro"), c(5, "oro")], rules=rules)
    hand.step(A.ENVIDO)
    assert A.FLOR in hand.legal_actions()
    hand.step(A.FLOR)
    assert hand.points == [0, 3]
    assert hand.pending is None
    assert hand.current_player == 0


def test_signal_phase_in_pairs():
    hand = TrucoHand(PAREJAS, rng=random.Random(1), mano=2)
    assert hand.signal_phase
    seen = []
    for _ in range(4):
        seat = hand.current_player
        seen.append(seat)
        hand.step(A.signal_action(seat))
    assert seen == [2, 3, 0, 1]
    assert not hand.signal_phase
    assert hand.current_player == 2
    assert hand.signal_view[0][2] == 2
    assert hand.signal_view[0][1] is None
    assert hand.signal_view[1][3] == 3


def test_spy_prob_one_reveals_everything():
    rules = Rules(num_players=4, signals=True, spy_prob=1.0)
    hand = TrucoHand(rules, rng=random.Random(1))
    for _ in range(4):
        hand.step(A.signal_action(5))
    assert all(v == 5 for row in hand.signal_view for v in row)


def test_scrambled_partner_signal():
    rules = Rules(num_players=4, signals=True, spy_prob=0.0, signal_vocab=8)
    hits = 0
    for seed in range(200):
        hand = TrucoHand(rules, rng=random.Random(seed), scramble_teams=(0,))
        for _ in range(4):
            hand.step(A.signal_action(3))
        hits += hand.signal_view[2][0] == 3
        assert hand.signal_view[1][3] == 3
    assert hits < 80


def test_pairs_round_order_and_team_win():
    rules = Rules(num_players=4, signals=False)
    hands = [
        [c(1, "espada"), c(4, "copa"), c(4, "oro")],
        [c(3, "oro"), c(5, "copa"), c(5, "oro")],
        [c(1, "basto"), c(6, "copa"), c(6, "oro")],
        [c(2, "oro"), c(7, "copa"), c(4, "basto")],
    ]
    hand = TrucoHand(rules, rng=random.Random(0), hands=hands, mano=0)
    for _ in range(4):
        hand.step(A.PLAY_0)
    assert hand.round_winners == [0]
    assert hand.current_player == 0


def test_same_team_tie_is_not_parda():
    rules = Rules(num_players=4, signals=False)
    hands = [
        [c(3, "oro"), c(4, "copa"), c(4, "oro")],
        [c(2, "oro"), c(5, "copa"), c(5, "oro")],
        [c(3, "copa"), c(6, "copa"), c(6, "oro")],
        [c(2, "copa"), c(7, "copa"), c(4, "basto")],
    ]
    hand = TrucoHand(rules, rng=random.Random(0), hands=hands, mano=0)
    for _ in range(4):
        hand.step(A.PLAY_0)
    assert hand.round_winners == [0]


def test_random_playouts_always_terminate():
    rng = random.Random(123)
    for rules in (HU, PAREJAS, Rules(num_players=4, flor=True, signals=True)):
        for _ in range(400):
            hand = TrucoHand(rules, rng=random.Random(rng.random()), scores=(rng.randrange(30), rng.randrange(30)), mano=rng.randrange(rules.num_players))
            steps = 0
            while not hand.done:
                hand.step(rng.choice(hand.legal_actions()))
                steps += 1
                assert steps < 200
            assert sum(hand.points) > 0
            assert min(hand.points) >= 0


def test_clone_is_independent():
    hand = deal_hu([c(4, "copa"), c(5, "copa"), c(6, "oro")], [c(7, "oro"), c(6, "espada"), c(5, "oro")])
    copy = hand.clone()
    copy.step(A.PLAY_0)
    assert hand.table[0][0] is None
    assert copy.table[0][0] is not None


def test_match_runs_to_target():
    rng = random.Random(7)
    match = TrucoMatch(Rules(num_players=4, target=15), rng=rng)
    while not match.done:
        hand = match.new_hand()
        while not hand.done:
            hand.step(rng.choice(hand.legal_actions()))
        match.close_hand()
    assert match.winner in (0, 1)
    assert max(match.scores) >= 15
