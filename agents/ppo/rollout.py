from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch

from truco.encoding import action_mask, encode
from truco.engine import TrucoHand
from truco.features import hand_features
from truco.rules import Rules

from .net import NetConfig, TrucoNet, build_net


def _scores(rng: random.Random, rules: Rules, zero_prob: float) -> tuple[int, int]:
    if rng.random() < zero_prob:
        return (0, 0)
    return (rng.randrange(rules.target), rng.randrange(rules.target))


def _aux_target(hand: TrucoHand, seat: int) -> np.ndarray:
    other = (seat + 2) % 4 if hand.n == 4 else (seat + 1) % 2
    return hand_features(hand.hands[other])


def _gae(values: list[float], reward: float, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in range(n - 1, -1, -1):
        r = reward if t == n - 1 else 0.0
        delta = r + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def run_rollouts(net: TrucoNet, opponent: TrucoNet | None, rules: Rules, settings: dict, seed: int) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    n_hands = int(settings["hands"])
    parallel = int(settings["parallel"])
    gamma = float(settings["gamma"])
    lam = float(settings["lam"])
    reward_scale = float(settings["reward_scale"])
    opp_prob = float(settings["opponent_prob"])
    zero_prob = float(settings["zero_score_prob"])
    n_actions = net.n_actions

    columns: dict[str, list] = {k: [] for k in ("obs", "mask", "action", "logp", "value", "adv", "ret", "aux", "signal")}
    tokens: Counter = Counter()
    reasons: Counter = Counter()
    rewards: list[float] = []
    started = 0
    vs_snapshot = 0

    def new_game() -> dict:
        nonlocal started, vs_snapshot
        started += 1
        hand = TrucoHand(
            rules,
            rng=random.Random(rng.random()),
            scores=_scores(rng, rules, zero_prob),
            mano=rng.randrange(rules.num_players),
        )
        frozen = rng.randrange(2) if opponent is not None and rng.random() < opp_prob else None
        vs_snapshot += frozen is not None
        learners = [s for s in range(rules.num_players) if frozen is None or s % 2 != frozen]
        return {"hand": hand, "frozen": frozen, "traj": {s: [] for s in learners}}

    def finish(game: dict) -> None:
        hand: TrucoHand = game["hand"]
        reasons[hand.end_reason] += 1
        for seat, steps in game["traj"].items():
            if not steps:
                continue
            reward = hand.team_reward(seat % 2) / reward_scale
            rewards.append(reward)
            adv, ret = _gae([s[4] for s in steps], reward, gamma, lam)
            aux = _aux_target(hand, seat)
            for i, (obs, mask, action, logp, value, is_signal) in enumerate(steps):
                columns["obs"].append(obs)
                columns["mask"].append(mask)
                columns["action"].append(action)
                columns["logp"].append(logp)
                columns["value"].append(value)
                columns["adv"].append(adv[i])
                columns["ret"].append(ret[i])
                columns["aux"].append(aux)
                columns["signal"].append(is_signal)

    active = [new_game() for _ in range(min(parallel, n_hands))]
    while active:
        groups: dict[str, list] = {"learn": [], "frozen": []}
        for i, game in enumerate(active):
            hand: TrucoHand = game["hand"]
            seat = hand.current_player
            key = "frozen" if game["frozen"] is not None and seat % 2 == game["frozen"] else "learn"
            groups[key].append((i, seat, encode(hand, seat), action_mask(hand, n_actions)))
        chosen: dict[int, int] = {}
        for key, model in (("learn", net), ("frozen", opponent)):
            items = groups[key]
            if not items:
                continue
            obs = torch.from_numpy(np.stack([it[2] for it in items]))
            mask = torch.from_numpy(np.stack([it[3] for it in items]))
            with torch.no_grad():
                logits, values, _ = model(obs, mask)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
            for j, (i, seat, o, m) in enumerate(items):
                a = int(acts[j])
                chosen[i] = a
                if key == "learn":
                    hand = active[i]["hand"]
                    if hand.signal_phase:
                        tokens[a] += 1
                    active[i]["traj"][seat].append((o, m, a, float(logps[j]), float(values[j]), hand.signal_phase))
        survivors = []
        for i, game in enumerate(active):
            game["hand"].step(chosen[i])
            if game["hand"].done:
                finish(game)
                if started < n_hands:
                    survivors.append(new_game())
            else:
                survivors.append(game)
        active = survivors

    batch = {
        "obs": np.asarray(columns["obs"], dtype=np.float16),
        "mask": np.asarray(columns["mask"], dtype=bool),
        "action": np.asarray(columns["action"], dtype=np.int64),
        "logp": np.asarray(columns["logp"], dtype=np.float32),
        "value": np.asarray(columns["value"], dtype=np.float32),
        "adv": np.asarray(columns["adv"], dtype=np.float32),
        "ret": np.asarray(columns["ret"], dtype=np.float32),
        "aux": np.asarray(columns["aux"], dtype=np.float32),
        "signal": np.asarray(columns["signal"], dtype=bool),
    }
    batch["stats"] = {
        "hands": n_hands,
        "vs_snapshot": vs_snapshot,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "tokens": dict(tokens),
        "reasons": dict(reasons),
    }
    return batch


def collect(args: tuple) -> dict:
    state, opponent_state, rules_dict, net_dict, settings, seed = args
    torch.set_num_threads(1)
    rules = Rules(**rules_dict)
    cfg = NetConfig(**net_dict)
    net = build_net(rules, cfg)
    net.load_state_dict(state)
    net.eval()
    opponent = None
    if opponent_state is not None:
        opponent = build_net(rules, cfg)
        opponent.load_state_dict(opponent_state)
        opponent.eval()
    return run_rollouts(net, opponent, rules, settings, seed)
