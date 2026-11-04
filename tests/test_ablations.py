import random

import numpy as np
import pytest

from agents.ablations import CONDITIONS, VALUES, ConstantSignalAgent, condition_setup, run_ablations, token_marginal
from agents.arena import SignalNoise, head_to_head
from agents.heuristic import HeuristicAgent
from truco import actions as A
from truco.engine import TrucoHand
from truco.rules import Rules

PAREJAS = Rules(num_players=4, signals=True, spy_prob=0.2)
SPIES = Rules(num_players=4, signals=True, spy_prob=1.0)


def _one_hot(token, vocab=8):
    return tuple(1.0 if i == token else 0.0 for i in range(vocab))


def _signal_all(hand, token):
    for _ in range(hand.n):
        hand.step(A.signal_action(token))
    return hand


def test_rival_noise_only_touches_the_spied_signals():
    hand = TrucoHand(SPIES, rng=random.Random(0), scramble_rival_teams=(0,), noise=(None, _one_hot(6)))
    _signal_all(hand, 3)
    assert hand.signal_view[0][1] == 6 and hand.signal_view[0][3] == 6
    assert hand.signal_view[2][1] == 6 and hand.signal_view[2][3] == 6
    assert hand.signal_view[0][2] == 3 and hand.signal_view[2][0] == 3
    assert hand.signal_view[1][0] == 3 and hand.signal_view[3][2] == 3


def test_blind_team_never_spies():
    hand = TrucoHand(SPIES, rng=random.Random(0), blind_teams=(1,))
    _signal_all(hand, 5)
    assert hand.signal_view[1][0] is None and hand.signal_view[3][2] is None
    assert hand.signal_view[1][3] == 5
    assert hand.signal_view[0][1] == 5


def test_partner_noise_uses_the_given_distribution():
    hand = TrucoHand(PAREJAS, rng=random.Random(0), scramble_teams=(0,), noise=(_one_hot(7), None))
    _signal_all(hand, 2)
    assert hand.signal_view[0][2] == 7 and hand.signal_view[2][0] == 7
    assert hand.signal_view[1][3] == 2


def test_noise_without_distribution_keeps_old_random_stream():
    a = TrucoHand(PAREJAS, rng=random.Random(9), scramble_teams=(1,))
    b = TrucoHand(PAREJAS, rng=random.Random(9), scramble_teams=(1,), noise=None)
    _signal_all(a, 4)
    _signal_all(b, 4)
    assert a.signal_view == b.signal_view


def test_signal_noise_maps_a_and_b_to_teams():
    noise = SignalNoise(partner_a=True, rivals_b=True, blind_a=True, dist_a=_one_hot(1), dist_b=_one_hot(2))
    first = noise.for_match(0)
    assert first["scramble_teams"] == (0,)
    assert first["scramble_rival_teams"] == (1,)
    assert first["blind_teams"] == (0,)
    assert first["noise"] == (_one_hot(1), _one_hot(2))
    second = noise.for_match(1)
    assert second["scramble_teams"] == (1,)
    assert second["scramble_rival_teams"] == (0,)
    assert second["noise"] == (_one_hot(2), _one_hot(1))
    assert SignalNoise().for_match(0)["noise"] is None


def test_scramble_flag_and_noise_object_agree():
    agents = ([HeuristicAgent(seed=1)], [HeuristicAgent(seed=2)])
    old = head_to_head(PAREJAS, *agents, 8, seed=3, scramble_a=True)
    agents = ([HeuristicAgent(seed=1)], [HeuristicAgent(seed=2)])
    new = head_to_head(PAREJAS, *agents, 8, seed=3, noise=SignalNoise(partner_a=True))
    assert old.to_dict() == new.to_dict()
    assert len(new.outcomes) == 8 and sum(new.outcomes) == new.wins_a
    assert sum(new.lengths) == new.hands
    lo, hi = new.point_diff_ci95()
    assert lo <= new.point_diff_per_hand <= hi


def test_constant_signal_agent():
    agent = ConstantSignalAgent(HeuristicAgent(seed=0), 5)
    hand = TrucoHand(PAREJAS, rng=random.Random(2))
    assert agent.act(hand, hand.current_player) == A.signal_action(5)
    _signal_all(hand, 0)
    assert not A.is_signal(agent.act(hand, hand.current_player))


def test_token_marginal_is_a_distribution():
    marginal = token_marginal(HeuristicAgent(seed=0), PAREJAS, hands=200, seed=1)
    assert marginal.shape == (8,)
    assert np.isclose(marginal.sum(), 1.0)
    assert marginal.max() < 1.0
    mute = token_marginal(HeuristicAgent(seed=0, use_signals=False), PAREJAS, hands=50, seed=1)
    assert mute[0] == 1.0


def test_condition_setup_covers_every_condition():
    marginal = np.full(8, 1 / 8)
    marginal[3] = 0.5
    seen = set()
    for condition in CONDITIONS:
        noise, constant = condition_setup(condition, marginal, marginal)
        seen.add(condition)
        if condition == "seña_constante":
            assert noise is None and constant == 3
        elif condition == "normal":
            assert noise is None and constant is None
        else:
            assert noise is not None and constant is None
    assert condition_setup("compañero_mezclado", marginal, marginal)[0].partner_a
    assert condition_setup("rivales_mezclados", marginal, marginal)[0].rivals_a
    assert condition_setup("rivales_ocultos", marginal, marginal)[0].blind_a
    both = condition_setup("ambos_mezclados", marginal, marginal)[0]
    assert both.partner_a and both.rivals_a and not both.partner_b
    assert seen == set(CONDITIONS)
    with pytest.raises(ValueError):
        condition_setup("otra", marginal, marginal)


def test_run_ablations_with_heuristics():
    summary = run_ablations(PAREJAS, "heuristic", "heuristic", matches=6, seed=0, marginal_hands=100)
    assert set(summary["condiciones"]) == set(CONDITIONS)
    assert set(summary["valores"]) == set(VALUES)
    for entry in summary["condiciones"].values():
        assert entry["matches"] == 6
        assert 0.0 <= entry["ci95"][0] <= entry["win_rate"] <= entry["ci95"][1] <= 1.0
    assert summary["condiciones"]["seña_constante"]["seña"] == int(np.argmax(summary["marginal_a"]))
    for value in summary["valores"].values():
        assert value["ci95"][0] <= value["diff"] <= value["ci95"][1]
    with pytest.raises(ValueError):
        run_ablations(Rules(num_players=4, signals=False), "heuristic", "random", matches=2)

