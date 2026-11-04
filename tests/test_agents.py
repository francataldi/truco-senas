import random

import numpy as np
import torch

from agents.arena import head_to_head
from agents.base import RandomAgent
from agents.cfr import MCCFR, CFRAgent, infoset_key, regret_matching
from agents.heuristic import HeuristicAgent, classic_signal
from agents.ppo.agent import PPOAgent
from agents.ppo.net import NetConfig, build_net, load_checkpoint, save_checkpoint
from agents.ppo.rollout import _gae, run_rollouts
from agents.signal_analysis import collect_signals, mutual_information, summarize
from truco import actions as A
from truco.cards import Card
from truco.encoding import action_mask, encode, obs_size
from truco.engine import TrucoHand
from truco.rules import Rules

PAREJAS = Rules(num_players=4, signals=True)
HU = Rules(num_players=2, signals=False)


def test_encoding_shape_is_stable():
    size = obs_size(PAREJAS)
    rng = random.Random(0)
    hand = TrucoHand(PAREJAS, rng=rng)
    while not hand.done:
        v = encode(hand, hand.current_player)
        assert v.shape == (size,)
        assert np.isfinite(v).all()
        hand.step(rng.choice(hand.legal_actions()))


def test_mask_matches_legal_actions():
    hand = TrucoHand(PAREJAS, rng=random.Random(3))
    mask = action_mask(hand, A.num_actions(PAREJAS.signal_vocab))
    assert set(np.flatnonzero(mask)) == set(hand.legal_actions())


def test_heuristic_beats_random():
    stats = head_to_head(PAREJAS, [HeuristicAgent(seed=1)], [RandomAgent(seed=2)], 40, seed=0)
    assert stats.win_rate > 0.8


def test_classic_signal_convention():
    cards = [Card(1, "espada").index, Card(4, "copa").index, Card(5, "oro").index]
    assert classic_signal(cards, 8) == 0


def test_regret_matching():
    assert np.allclose(regret_matching(np.array([1.0, 3.0, -2.0])), [0.25, 0.75, 0.0])
    assert np.allclose(regret_matching(np.zeros(4)), 0.25)


def test_mccfr_runs_and_agent_plays():
    solver = MCCFR(HU, seed=0)
    for _ in range(20):
        solver.iterate()
    table = solver.average_strategy()
    assert len(table) > 0
    agent = CFRAgent(table, seed=0)
    stats = head_to_head(HU, [agent], [RandomAgent(seed=1)], 10, seed=0)
    assert stats.matches == 10


def test_infoset_key_hides_rival_cards():
    rng = random.Random(5)
    hand = TrucoHand(HU, rng=rng)
    other = hand.clone()
    other.hands[1] = list(reversed(other.hands[1]))
    assert infoset_key(hand, 0) == infoset_key(other, 0)


def test_gae_terminal_reward():
    adv, ret = _gae([0.0, 0.0, 0.0], 1.0, gamma=1.0, lam=1.0)
    assert np.allclose(ret, 1.0)
    assert np.allclose(adv, 1.0)


def test_rollouts_and_checkpoint(tmp_path):
    net = build_net(PAREJAS, NetConfig(hidden=32, depth=1))
    settings = {"hands": 12, "parallel": 4, "gamma": 1.0, "lam": 0.95, "reward_scale": 4.0, "opponent_prob": 0.5, "zero_score_prob": 0.5}
    batch = run_rollouts(net, build_net(PAREJAS, NetConfig(hidden=32, depth=1)), PAREJAS, settings, seed=0)
    n = batch["action"].shape[0]
    assert n > 0
    assert batch["obs"].shape == (n, obs_size(PAREJAS))
    assert batch["mask"][np.arange(n), batch["action"]].all()
    assert batch["signal"].any()
    path = tmp_path / "net.pt"
    save_checkpoint(path, net, PAREJAS)
    loaded, rules, _ = load_checkpoint(path)
    assert rules == PAREJAS
    obs = torch.from_numpy(batch["obs"][:3].astype(np.float32))
    mask = torch.from_numpy(batch["mask"][:3])
    assert torch.allclose(net(obs, mask)[0], loaded(obs, mask)[0])


def test_ppo_agent_plays_legal_moves():
    net = build_net(PAREJAS, NetConfig(hidden=32, depth=1))
    stats = head_to_head(PAREJAS, [PPOAgent(net, seed=0)], [HeuristicAgent(seed=1)], 4, seed=0)
    assert stats.matches == 4


def test_signal_analysis_detects_classic_convention():
    agents = [HeuristicAgent(seed=i) for i in range(4)]
    data = collect_signals(PAREJAS, agents, 300, seed=0)
    summary = summarize(data, PAREJAS.signal_vocab)
    assert summary["mi_carta_mas_alta_bits"] > 0.8
    tokens = np.array([0, 0, 1, 1])
    assert abs(mutual_information(tokens, np.array([0, 0, 1, 1])) - 1.0) < 1e-9


def test_ppo_evaluate_exposes_policy_value_and_beliefs():
    net = build_net(PAREJAS, NetConfig(hidden=32, depth=1))
    agent = PPOAgent(net, seed=0, rules=PAREJAS)
    rng = random.Random(4)
    hand = TrucoHand(PAREJAS, rng=rng)
    while not hand.done:
        seat = hand.current_player
        probs, value, beliefs = agent.evaluate(hand, seat)
        legal = hand.legal_actions()
        assert abs(probs[legal].sum() - 1.0) < 1e-5
        assert np.isfinite(value)
        assert beliefs.shape == (8,) and ((beliefs >= 0) & (beliefs <= 1)).all()
        hand.step(agent.act(hand, seat))


def test_ppo_without_signals_plays_at_signal_table():
    mute_rules = Rules(num_players=4, signals=False)
    net = build_net(mute_rules, NetConfig(hidden=32, depth=1))
    agent = PPOAgent(net, seed=0, rules=mute_rules)
    hand = TrucoHand(PAREJAS, rng=random.Random(2))
    assert agent.is_mute(hand)
    assert agent.act(hand, hand.current_player) == A.signal_action(0)
    assert encode(hand, 0, signals=False).shape == (obs_size(mute_rules),)
    stats = head_to_head(PAREJAS, [agent], [HeuristicAgent(seed=1)], 3, seed=0)
    assert stats.matches == 3
