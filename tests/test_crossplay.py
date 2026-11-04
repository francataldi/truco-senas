import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from agents.crossplay import best_permutation, dictionary_alignment, first_signal_distribution, run_crossplay
from agents.heuristic import HeuristicAgent
from agents.ppo.agent import PPOAgent
from agents.ppo.net import NetConfig, build_net, load_checkpoint, save_checkpoint
from agents.ppo.train import TrainConfig, train
from truco.rules import Rules

ROOT = Path(__file__).resolve().parents[1]
PAREJAS = Rules(num_players=4, signals=True)


def _script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_hot(tokens, vocab):
    out = np.zeros((len(tokens), vocab))
    out[np.arange(len(tokens)), tokens] = 1.0
    return out


def test_alignment_detects_permuted_languages():
    rng = np.random.default_rng(0)
    meaning = rng.integers(0, 4, 4000)
    permutation = np.array([2, 0, 3, 1])
    same = dictionary_alignment(_one_hot(meaning, 4), _one_hot(meaning, 4))
    permuted = dictionary_alignment(_one_hot(meaning, 4), _one_hot(permutation[meaning], 4))
    noise = dictionary_alignment(_one_hot(meaning, 4), _one_hot(rng.integers(0, 4, 4000), 4))
    assert same["acuerdo_directo"] == 1.0 and same["mi_normalizada"] > 0.99
    assert permuted["acuerdo_directo"] < 0.3
    assert permuted["acuerdo_permutado"] == 1.0
    assert permuted["permutacion"] == permutation.tolist()
    assert permuted["mi_normalizada"] > 0.99
    assert noise["mi_normalizada"] < 0.01
    assert abs(noise["acuerdo_permutado"] - 0.25) < 0.05


def test_best_permutation_on_known_joint():
    joint = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
    score, mapping = best_permutation(joint)
    assert mapping == [1, 2, 0]
    assert np.isclose(score, 1.0)
    big = np.eye(10)[::-1] / 10
    score, mapping = best_permutation(big)
    assert np.isclose(score, 1.0) and mapping == list(range(9, -1, -1))


@pytest.fixture
def seed_checkpoints(tmp_path):
    paths = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        path = tmp_path / f"seed_{seed}" / "last.pt"
        path.parent.mkdir()
        save_checkpoint(path, build_net(PAREJAS, NetConfig(hidden=16, depth=1)), PAREJAS, {"iteration": 1})
        paths.append(path)
    return paths


def test_first_signal_distribution(seed_checkpoints):
    agent = PPOAgent.from_checkpoint(str(seed_checkpoints[0]), seed=0)
    probs = first_signal_distribution(agent, PAREJAS, 50, seed=3)
    assert probs.shape == (50, 8)
    assert np.allclose(probs.sum(1), 1.0)
    classic = first_signal_distribution(HeuristicAgent(seed=0), PAREJAS, 50, seed=3)
    assert set(np.unique(classic)) <= {0.0, 1.0}
    assert np.allclose(classic.sum(1), 1.0)


def test_run_crossplay_summary(seed_checkpoints):
    specs = [f"ppo:{p.as_posix()}" for p in seed_checkpoints]
    summary = run_crossplay(PAREJAS, ["s0", "s1"], specs, opponent="random", matches=4, seed=0, marginal_hands=40, signal_hands=60, bootstrap=50)
    assert np.array(summary["matriz"], dtype=float).shape == (2, 2)
    kinds = sorted(c["tipo"] for c in summary["celdas"])
    assert kinds == ["cross-play", "mezclado", "mezclado", "self-play", "self-play"]
    assert set(summary["agregado"]) == {"self_play", "cross_play", "self_play_mezclado"}
    assert summary["agregado"]["self_play"]["n"] == 8
    assert set(summary["diferencias"]) == {"self_menos_cross", "self_menos_mezclado", "cross_menos_mezclado"}
    assert len(summary["diccionarios"]) == 1
    assert 0.0 <= summary["diccionarios"][0]["acuerdo_permutado"] <= 1.0
    again = run_crossplay(PAREJAS, ["s0", "s1"], specs, opponent="random", matches=4, seed=0, marginal_hands=40, signal_hands=0, bootstrap=50)
    assert again["matriz"] == summary["matriz"]


def test_seed_overrides():
    module = _script("train_ppo")
    parse = module.parser().parse_args
    cfg = module.apply_overrides(TrainConfig(out_dir="runs/x"), parse(["--seed", "3"]))
    assert cfg.seed == 3 and cfg.out_dir == "runs/x_seed3"
    cfg = module.apply_overrides(TrainConfig(out_dir="runs/sem/seed_{seed}"), parse(["--seed", "4"]))
    assert cfg.out_dir == "runs/sem/seed_4"
    cfg = module.apply_overrides(TrainConfig(out_dir="runs/sem/seed_{seed}", seed=2), parse([]))
    assert cfg.out_dir == "runs/sem/seed_2"
    cfg = module.apply_overrides(TrainConfig(out_dir="runs/x"), parse(["--seed", "5", "--out", "otra"]))
    assert cfg.out_dir == "otra" and cfg.seed == 5
    with open(ROOT / "configs" / "ppo_semillas.yaml", encoding="utf-8") as f:
        assert "{seed}" in yaml.safe_load(f)["out_dir"]


def _tiny_config(tmp_path):
    data = {
        "rules": {"num_players": 4, "signals": True},
        "net": {"hidden": 16, "depth": 1},
        "iterations": 1,
        "workers": 1,
        "hands_per_worker": 6,
        "parallel_hands": 3,
        "minibatch": 64,
        "epochs": 1,
        "eval_every": 1,
        "eval_matches": 1,
        "save_every": 1,
        "device": "cpu",
        "amp": False,
        "out_dir": str(tmp_path / "semillas" / "seed_{seed}"),
    }
    path = tmp_path / "chico.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_train_seeds_script(tmp_path):
    path = _tiny_config(tmp_path)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, str(ROOT / "scripts" / "train_seeds.py"), "--config", str(path), "--seeds", "3", "4"]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, cwd=ROOT, timeout=900)
    assert done.returncode == 0, done.stderr
    for seed in (3, 4):
        run = tmp_path / "semillas" / f"seed_{seed}"
        assert (run / "last.pt").exists()
        assert json.loads((run / "config.json").read_text(encoding="utf-8"))["seed"] == seed
    a = load_checkpoint(tmp_path / "semillas" / "seed_3" / "last.pt")[0]
    b = load_checkpoint(tmp_path / "semillas" / "seed_4" / "last.pt")[0]
    assert not torch.equal(a.body[0].weight, b.body[0].weight)


def test_training_with_a_seed_is_reproducible(tmp_path):
    path = _tiny_config(tmp_path)
    states = []
    for name in ("a", "b"):
        cfg = TrainConfig.from_yaml(path)
        cfg.seed = 7
        cfg.out_dir = str(tmp_path / name)
        train(cfg)
        states.append(load_checkpoint(Path(cfg.out_dir) / "last.pt")[0].state_dict())
    for key in states[0]:
        assert torch.allclose(states[0][key], states[1][key]), key
