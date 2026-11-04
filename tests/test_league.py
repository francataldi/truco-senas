import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from agents.cfr import MCCFR
from agents.league import Entrant, checkpoint_iteration, compatible, parse_entrant, round_robin, run_checkpoints, summarize_league, win_matrix
from agents.ppo.net import NetConfig, build_net, save_checkpoint
from agents.ratings import ELO_PER_LOGIT, bootstrap_ratings, expected_win_matrix, fit_bradley_terry, log_likelihood, rating_table
from agents.registry import spec_label, spec_rules
from truco.rules import Rules

ROOT = Path(__file__).resolve().parents[1]
PAREJAS = Rules(num_players=4, signals=True)
HU = Rules(num_players=2, signals=False)


def synthetic_wins(truth, games, seed=0):
    rng = np.random.default_rng(seed)
    k = len(truth)
    wins = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            p = 1.0 / (1.0 + np.exp(-(truth[i] - truth[j])))
            w = rng.binomial(games, p)
            wins[i, j] = w
            wins[j, i] = games - w
    return wins


def test_bradley_terry_recovers_known_ratings():
    truth = np.array([-1.2, -0.3, 0.1, 0.6, 1.5])
    truth = truth - truth.mean()
    wins = synthetic_wins(truth, 4000, seed=1)
    fitted = fit_bradley_terry(wins, prior_sd=None)
    assert np.abs(fitted - truth).max() < 0.06
    assert np.isclose(fitted.mean(), 0.0)
    boot = bootstrap_ratings(wins, samples=300, seed=2, prior_sd=None)
    lo, hi = np.quantile(boot, 0.005, axis=0), np.quantile(boot, 0.995, axis=0)
    assert np.all(lo <= truth) and np.all(truth <= hi)
    assert np.all(hi - lo < 0.3)
    assert log_likelihood(wins, fitted) >= log_likelihood(wins, truth) - 1e-6


def test_bradley_terry_recovers_ratings_from_sparse_schedules():
    truth = np.array([0.0, 0.8, 1.6, -0.8])
    truth = truth - truth.mean()
    wins = synthetic_wins(truth, 6000, seed=3)
    wins[0, 2] = wins[2, 0] = 0
    fitted = fit_bradley_terry(wins, prior_sd=None)
    assert np.abs(fitted - truth).max() < 0.08


def test_prior_keeps_perfect_separation_finite():
    wins = np.array([[0.0, 50.0], [0.0, 0.0]])
    ratings = fit_bradley_terry(wins, prior_sd=4.0)
    assert np.all(np.isfinite(ratings))
    assert 3.0 < ratings[0] - ratings[1] < 20.0
    p = expected_win_matrix(ratings)
    assert np.isnan(p[0, 0]) and p[0, 1] > 0.95


def test_rating_table_anchors_and_orders():
    truth = np.array([-1.0, 0.0, 1.0])
    wins = synthetic_wins(truth, 500, seed=4)
    rows = rating_table(["malo", "medio", "bueno"], wins, samples=200, seed=0, anchor="medio")
    assert [r["nombre"] for r in rows] == ["bueno", "medio", "malo"]
    medio = next(r for r in rows if r["nombre"] == "medio")
    assert medio["elo"] == 0.0 and medio["elo_lo"] == 0.0 and medio["elo_hi"] == 0.0
    bueno = rows[0]
    assert bueno["elo_lo"] < bueno["elo"] < bueno["elo_hi"]
    assert abs(bueno["elo"] / ELO_PER_LOGIT - 1.0) < 0.35
    assert bueno["p_mejor"] > 0.9
    assert sum(r["partidos"] for r in rows) == 2 * 3 * 500


def test_parse_entrant_and_labels():
    assert parse_entrant("mejor=heuristic") == Entrant("mejor", "heuristic")
    assert parse_entrant("heuristic-mudo") == Entrant("heuristico_mudo", "heuristic-mudo")
    assert spec_label("ppo:runs/parejas_senas/iter_000500.pt") == "parejas_senas@500"
    assert spec_label("ppo-greedy:runs/parejas_senas/last.pt") == "parejas_senas@final_greedy"
    assert spec_label("cfr:runs/cfr/heads_up.npz") == "cfr_heads_up"
    assert checkpoint_iteration("iter_003000.pt") == 3000
    assert checkpoint_iteration("last.pt") is None


def test_run_checkpoints_picks_spread(tmp_path):
    for it in range(50, 1050, 50):
        (tmp_path / f"iter_{it:06d}.pt").write_bytes(b"")
    (tmp_path / "last.pt").write_bytes(b"")
    picked = [checkpoint_iteration(p) for p in run_checkpoints(tmp_path, per_run=3)]
    assert picked == [50, 550, 1000]
    assert [checkpoint_iteration(p) for p in run_checkpoints(tmp_path, iterations=[100, 990])] == [100, 1000]
    assert [checkpoint_iteration(p) for p in run_checkpoints(tmp_path, per_run=1)] == [1000]
    empty = tmp_path / "vacio"
    empty.mkdir()
    assert run_checkpoints(empty) == []


@pytest.fixture
def tiny_agents(tmp_path):
    ppo = tmp_path / "corrida" / "iter_000010.pt"
    ppo.parent.mkdir()
    save_checkpoint(ppo, build_net(PAREJAS, NetConfig(hidden=16, depth=1)), PAREJAS, {"iteration": 10})
    mute = tmp_path / "muda" / "iter_000010.pt"
    mute.parent.mkdir()
    no_signals = Rules(num_players=4, signals=False)
    save_checkpoint(mute, build_net(no_signals, NetConfig(hidden=16, depth=1)), no_signals, {"iteration": 10})
    solver = MCCFR(HU, seed=0)
    for _ in range(5):
        solver.iterate()
    cfr = tmp_path / "cfr.npz"
    solver.policy().save(cfr)
    return {"ppo": f"ppo:{ppo.as_posix()}", "mute": f"ppo:{mute.as_posix()}", "cfr": f"cfr:{cfr.as_posix()}", "dir": tmp_path}


def test_compatibility_rules(tiny_agents):
    assert spec_rules(tiny_agents["ppo"]) == PAREJAS
    assert spec_rules(tiny_agents["cfr"]) == HU
    assert spec_rules("heuristic") is None
    assert compatible(tiny_agents["ppo"], PAREJAS)[0]
    assert compatible(tiny_agents["mute"], PAREJAS)[0]
    assert not compatible(tiny_agents["ppo"], HU)[0]
    assert not compatible(tiny_agents["cfr"], PAREJAS)[0]
    assert compatible(tiny_agents["cfr"], HU)[0]
    assert not compatible(tiny_agents["ppo"], Rules(num_players=4, signal_vocab=4))[0]


def test_round_robin_mixes_signal_and_mute_agents(tiny_agents):
    entrants = [
        Entrant("ppo", tiny_agents["ppo"]),
        Entrant("ppo_mudo", tiny_agents["mute"]),
        Entrant("heuristico", "heuristic"),
        Entrant("random", "random"),
    ]
    results = round_robin(PAREJAS, entrants, matches=4, seed=3)
    assert len(results) == 6
    assert [(r["i"], r["j"]) for r in results] == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    again = round_robin(PAREJAS, entrants, matches=4, seed=3)
    assert [r["wins"] for r in results] == [r["wins"] for r in again]
    wins = win_matrix(4, results)
    assert np.all(wins + wins.T == np.where(np.eye(4) > 0, 0, 4))
    summary = summarize_league(entrants, results, samples=50, seed=0)
    assert summary["ancla"] == "heuristico"
    assert {r["nombre"] for r in summary["ratings"]} == {e.name for e in entrants}
    assert len(summary["partidos"]) == 6


def test_heads_up_league_with_cfr(tiny_agents):
    entrants = [Entrant("cfr", tiny_agents["cfr"]), Entrant("heuristico", "heuristic"), Entrant("random", "random")]
    results = round_robin(HU, entrants, matches=4, seed=0)
    summary = summarize_league(entrants, results, samples=30, seed=0)
    assert len(summary["ratings"]) == 3


def test_league_script_writes_outputs(tiny_agents):
    out = tiny_agents["dir"] / "liga"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "league.py"),
        "--runs",
        str(tiny_agents["dir"] / "corrida"),
        str(tiny_agents["dir"] / "muda"),
        "--specs",
        tiny_agents["cfr"],
        "--baselines",
        "heuristic",
        "random",
        "--matches",
        "2",
        "--bootstrap",
        "20",
        "--workers",
        "1",
        "--out",
        str(out),
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env, cwd=ROOT, timeout=600)
    assert done.returncode == 0, done.stderr
    assert "salteo cfr_cfr" in done.stdout
    for name in ("liga.json", "ratings.csv", "partidos.csv", "liga.png", "liga_matriz.png"):
        assert (out / name).exists(), name
    summary = json.loads((out / "liga.json").read_text(encoding="utf-8"))
    assert set(summary["nombres"]) == {"corrida@10", "muda@10", "heuristico", "random"}
    assert (out / "ratings.csv").read_text(encoding="utf-8").startswith("nombre,spec,rating,elo")
