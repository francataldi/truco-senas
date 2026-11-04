import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agents.ablations import run_ablations
from agents.crossplay import run_crossplay
from agents.league import Entrant, round_robin, summarize_league
from agents.ppo.net import NetConfig, build_net, save_checkpoint
from agents.report import build_report, discover, groups_of, learning_series, read_jsonl
from agents.stats import wilson_interval
from truco.rules import Rules

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
METRIC_FIELDS = ("iteration", "hands", "mean_reward", "entropy", "signal_info", "win_vs_heuristic")


def fake_run(folder: Path, signals: bool, rates: list[float], checkpoint: bool = True) -> None:
    folder.mkdir(parents=True)
    rules = Rules(num_players=4, signals=signals)
    (folder / "config.json").write_text(json.dumps({"rules": rules.to_dict(), "seed": 0}), encoding="utf-8")
    with open(folder / "eval.jsonl", "w", encoding="utf-8") as f:
        for i, rate in enumerate(rates):
            row = {
                "iteration": 25 * (i + 1),
                "vs_heuristic": {"matches": 100, "win_rate": rate, "ci95": [0, 1]},
                "vs_random": {"matches": 25, "win_rate": min(1.0, rate + 0.2), "ci95": [0, 1]},
            }
            if signals:
                row["vs_self_scrambled"] = {"matches": 50, "win_rate": 0.5 + rate / 10, "ci95": [0, 1]}
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps({"iteration": 25, "vs_heuristic": {"matches": 100, "win_rate": rates[0], "ci95": [0, 1]}}) + "\n")
    with open(folder / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(METRIC_FIELDS))
        writer.writeheader()
        for it in range(1, 25 * len(rates) + 1):
            row = {"iteration": it, "hands": 512 * it, "mean_reward": 0.01 * it, "entropy": 1.0 / it, "signal_info": 0.002 * it}
            writer.writerow({**row, "win_vs_heuristic": ""})
    if checkpoint:
        save_checkpoint(folder / "last.pt", build_net(rules, NetConfig(hidden=16, depth=1)), rules, {"iteration": 25 * len(rates)})


@pytest.fixture(scope="module")
def fake_runs(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    fake_run(runs / "parejas_senas", True, [0.4, 0.55, 0.7])
    fake_run(runs / "parejas_sin_senas", False, [0.35, 0.5, 0.6])
    fake_run(runs / "semillas" / "seed_0", True, [0.3, 0.5], checkpoint=False)
    fake_run(runs / "semillas" / "seed_1", True, [0.32, 0.48], checkpoint=False)
    rules = Rules(num_players=4, signals=True)
    entrants = [Entrant("heuristico", "heuristic"), Entrant("heuristico_mudo", "heuristic-mudo"), Entrant("random", "random")]
    league = summarize_league(entrants, round_robin(rules, entrants, matches=4, seed=0), samples=30)
    (runs / "liga").mkdir()
    (runs / "liga" / "liga.json").write_text(json.dumps({**league, "rules": rules.to_dict()}, ensure_ascii=False), encoding="utf-8")
    cross = run_crossplay(rules, ["a", "b"], ["heuristic", "heuristic-mudo"], opponent="random", matches=2, marginal_hands=20, signal_hands=30, bootstrap=20)
    (runs / "crossplay").mkdir()
    (runs / "crossplay" / "crossplay.json").write_text(json.dumps(cross, ensure_ascii=False), encoding="utf-8")
    ablations = run_ablations(rules, "heuristic", "heuristic", matches=2, marginal_hands=20)
    ablations["run"] = "parejas_senas"
    (runs / "parejas_senas" / "ablaciones").mkdir()
    (runs / "parejas_senas" / "ablaciones" / "ablaciones.json").write_text(json.dumps(ablations, ensure_ascii=False), encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_cfr.py"),
        "--iterations",
        "12",
        "--workers",
        "1",
        "--report-every",
        "4",
        "--eval-matches",
        "2",
        "--out",
        str(runs / "cfr" / "heads_up.npz"),
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=ENV, cwd=ROOT, timeout=600)
    assert done.returncode == 0, done.stderr
    return runs


def test_train_cfr_script_outputs(fake_runs):
    folder = fake_runs / "cfr"
    summary = json.loads((folder / "heads_up.json").read_text(encoding="utf-8"))
    assert summary["iterations"] == 12
    assert summary["infosets"] > 0
    assert summary["regret_mode"] == "linear"
    assert 0 < summary["policy_bytes_per_infoset"] < 64
    rows = list(csv.DictReader(open(folder / "curva_infosets.csv", encoding="utf-8")))
    assert [int(r["iteration"]) for r in rows] == [4, 8, 12]
    assert all(int(r["rss_bytes"]) > 0 for r in rows)
    assert not (folder / "heads_up_partes").exists()


def test_discover_and_groups(fake_runs):
    found = discover(fake_runs, fake_runs / "reporte")
    names = [p.name for p in found["runs"]]
    assert names == ["parejas_senas", "parejas_sin_senas", "seed_0", "seed_1"]
    assert len(found["ligas"]) == 1 and len(found["crossplay"]) == 1 and len(found["ablaciones"]) == 1
    assert len(found["cfr_curvas"]) == 1
    groups = groups_of(found["runs"], fake_runs)
    assert sorted(groups) == ["principales", "semillas"]
    assert len(groups["semillas"]) == 2


def test_learning_series_uses_wilson_and_dedupes(fake_runs):
    rows = read_jsonl(fake_runs / "parejas_senas" / "eval.jsonl")
    assert [r["iteration"] for r in rows] == [25, 50, 75]
    series = learning_series(fake_runs / "parejas_senas", "senas", "vs_heuristic", use_hands=True)
    assert series["y"] == [0.4, 0.55, 0.7]
    assert series["x"][0] == pytest.approx(25 * 512 / 1e6)
    lo, hi = wilson_interval(70, 100)
    assert series["lo"][-1] == pytest.approx(lo) and series["hi"][-1] == pytest.approx(hi)
    assert learning_series(fake_runs / "parejas_sin_senas", "x", "vs_self_scrambled", use_hands=False) is None


def test_build_report_writes_every_figure(fake_runs):
    out = fake_runs / "reporte"
    report = build_report(fake_runs, out, signal_hands=60, bootstrap=20)
    expected = [
        "curvas_heuristico_principales.png",
        "curvas_random_principales.png",
        "curvas_senas_principales.png",
        "curvas_heuristico_semillas.png",
        "entrenamiento_principales.png",
        "diccionario_parejas_senas.png",
        "mi_por_rasgo.png",
        "liga_liga.png",
        "liga_matriz_liga.png",
        "crossplay_crossplay.png",
        "crossplay_matriz_crossplay.png",
        "ablaciones.png",
        "cfr_infosets_cfr.png",
        "resumen.json",
        "tablas.tex",
        "reporte.html",
    ]
    for name in expected:
        assert (out / name).exists(), name
        assert (out / name).stat().st_size > 0
    assert not (out / "diccionario_parejas_sin_senas.png").exists()
    assert "parejas_senas" in report["señas"]
    summary = json.loads((out / "resumen.json").read_text(encoding="utf-8"))
    finals = {(r["run"], r["medida"]): r for r in summary["curvas_finales"]}
    assert finals[("parejas_senas", "vs_heuristic")]["tasa"] == 0.7
    assert summary["cfr"][0]["politica"]["infosets"] > 0
    assert summary["ligas"][0]["ancla"] == "heuristico"
    html = (out / "reporte.html").read_text(encoding="utf-8")
    assert html.count("<img") == len(report["figuras"])
    assert "\\toprule" in (out / "tablas.tex").read_text(encoding="utf-8")
    again = build_report(fake_runs, out, signal_hands=60, bootstrap=20)
    assert len(again["figuras"]) == len(report["figuras"])


def test_report_script(fake_runs, tmp_path):
    out = tmp_path / "otro_reporte"
    cmd = [sys.executable, str(ROOT / "scripts" / "report.py"), "--runs", str(fake_runs), "--out", str(out), "--signal-hands", "40", "--bootstrap", "10"]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=ENV, cwd=ROOT, timeout=600)
    assert done.returncode == 0, done.stderr
    assert "figuras en" in done.stdout
    assert (out / "reporte.html").exists()
