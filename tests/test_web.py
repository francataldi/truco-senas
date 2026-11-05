import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.ppo.agent import PPOAgent
from agents.ppo.net import NetConfig, build_net, save_checkpoint
from agents.ppo.numpy_net import NumpyPPOAgent, erf, export_checkpoint, gelu, load_numpy_policy
from agents.ppo.policy import PolicyAgent
from demo.service import DemoService, Settings, handle_json, route
from truco.engine import TrucoHand
from truco.rules import Rules

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = Rules(num_players=4, signals=True)
MUTE = Rules(num_players=4, signals=False)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    torch.manual_seed(1)
    for name, rules in (("parejas_senas", SIGNALS), ("parejas_sin_senas", MUTE)):
        folder = root / name
        folder.mkdir()
        save_checkpoint(folder / "last.pt", build_net(rules, NetConfig(hidden=48, depth=2)), rules, {"iteration": 11})
        (folder / "eval.jsonl").write_text(json.dumps({"iteration": 11, "vs_heuristic": {"win_rate": 0.55}}) + "\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def numpy_runs(checkpoints, tmp_path_factory):
    root = tmp_path_factory.mktemp("web") / "runs"
    for name in ("parejas_senas", "parejas_sin_senas"):
        export_checkpoint(checkpoints / name / "last.pt", root / name / "last.npz")
        (root / name / "eval.jsonl").write_text((checkpoints / name / "eval.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    return root


def test_erf_and_gelu_match_torch():
    x = np.linspace(-6, 6, 2001)
    assert np.abs(erf(x) - torch.erf(torch.from_numpy(x)).numpy()).max() < 2e-7
    expected = torch.nn.functional.gelu(torch.from_numpy(x)).numpy()
    assert np.abs(gelu(x) - expected).max() < 1e-6


@pytest.mark.parametrize("name", ["parejas_senas", "parejas_sin_senas"])
def test_numpy_policy_matches_torch(checkpoints, numpy_runs, name):
    torch_agent = PPOAgent.from_checkpoint(str(checkpoints / name / "last.pt"))
    numpy_agent = NumpyPPOAgent.from_file(numpy_runs / name / "last.npz")
    assert isinstance(torch_agent, PolicyAgent) and isinstance(numpy_agent, PolicyAgent)
    rng = random.Random(0)
    worst = 0.0
    for k in range(25):
        hand = TrucoHand(torch_agent.rules, rng=random.Random(k), mano=k % 4)
        while not hand.done:
            seat = hand.current_player
            p1, v1, a1 = torch_agent.evaluate(hand, seat)
            p2, v2, a2 = numpy_agent.evaluate(hand, seat)
            worst = max(worst, float(np.abs(p1 - p2).max()), abs(v1 - v2), float(np.abs(a1 - a2).max()))
            hand.step(rng.choice(hand.legal_actions()))
    assert worst < 1e-4


def test_float16_export_keeps_decisions(checkpoints, tmp_path):
    path = export_checkpoint(checkpoints / "parejas_senas" / "last.pt", tmp_path / "half.npz", dtype=np.float16)
    net, rules, extra = load_numpy_policy(path)
    assert rules == SIGNALS and extra["iteration"] == 11
    full = PPOAgent.from_checkpoint(str(checkpoints / "parejas_senas" / "last.pt"))
    half = NumpyPPOAgent(net, rules=rules)
    hand = TrucoHand(rules, rng=random.Random(3), mano=0)
    p_full = full.probabilities(hand, hand.current_player)
    p_half = half.probabilities(hand, hand.current_player)
    assert np.abs(p_full - p_half).max() < 0.02
    assert int(np.argmax(p_full)) == int(np.argmax(p_half))


def test_numpy_agent_plays_legal_moves(numpy_runs):
    agent = NumpyPPOAgent.from_file(numpy_runs / "parejas_senas" / "last.npz", seed=5)
    for k in range(10):
        hand = TrucoHand(SIGNALS, rng=random.Random(100 + k), mano=k % 4)
        while not hand.done:
            action = agent.act(hand, hand.current_player)
            assert action in hand.legal_actions()
            hand.step(action)


@pytest.fixture()
def service(numpy_runs):
    return DemoService(Settings(roots=[numpy_runs], base=numpy_runs.parent, dictionary_hands=120, prefetch=False, torch_threads=None))


def test_service_discovers_numpy_policies(service):
    status, data = route(service, "GET", "/api/agents")
    assert status == 200
    specs = [a["spec"] for a in data["agents"]]
    assert "ppo:runs/parejas_senas/last.npz" in specs
    assert data["defaults"]["play"][2] == "ppo:runs/parejas_senas/last.npz"
    assert data["defaults"]["watch"][1] == "ppo:runs/parejas_sin_senas/last.npz"


def test_service_full_game_flow(service):
    status, game = route(service, "POST", "/api/games", {"mode": "watch", "target": 15, "seed": 4})
    assert status == 201
    gid, seq = game["id"], game["seq"]
    status, step = route(service, "POST", f"/api/games/{gid}/advance", {"steps": 40, "seq": seq})
    assert status == 200 and step["events"]
    status, snap = route(service, "GET", f"/api/games/{gid}?since=0")
    assert status == 200 and "state" in snap
    assert route(service, "DELETE", f"/api/games/{gid}")[0] == 204
    assert route(service, "GET", f"/api/games/{gid}")[0] == 404


def test_service_human_game_and_advice(service):
    status, game = route(service, "POST", "/api/games", {"mode": "play", "seed": 2})
    assert status == 201
    gid = game["id"]
    status, snap = route(service, "GET", f"/api/games/{gid}")
    legal = [item["action"] for item in snap["state"].get("legal", [])]
    status, advice = route(service, "GET", f"/api/games/{gid}/advice")
    assert status == 200 and "advice" in advice
    if legal:
        status, _ = route(service, "POST", f"/api/games/{gid}/action", {"action": legal[0], "seq": snap["seq"]})
        assert status == 200


@pytest.mark.parametrize(
    ("method", "url", "body", "expected"),
    [
        ("GET", "/api/nada", None, 404),
        ("POST", "/api/games", {"mode": "otro"}, 422),
        ("POST", "/api/games", {"seats": ["human"]}, 422),
        ("POST", "/api/games", {"spy_prob": 3}, 422),
        ("GET", "/api/dictionary", None, 422),
        ("GET", "/api/dictionary?spec=human", None, 422),
        ("GET", "/api/dictionary?spec=ppo:no/existe.npz", None, 422),
    ],
)
def test_service_rejects_bad_requests(service, method, url, body, expected):
    status, data = route(service, method, url, body)
    assert status == expected and data["detail"]


def test_handle_json_roundtrip(service):
    out = json.loads(handle_json(service, "GET", "/api/health"))
    assert out == {"status": 200, "data": {"ok": True, "games": 0}}
    assert json.loads(handle_json(service, "POST", "/api/games", "no es json"))["status"] == 400


def test_build_web_bundle_runs_without_torch(checkpoints, tmp_path):
    out = tmp_path / "docs"
    script = ROOT / "scripts" / "build_web.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--runs", str(checkpoints), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    html = (out / "index.html").read_text(encoding="utf-8")
    assert 'data-backend="local"' in html
    assert (out / ".nojekyll").exists() and (out / "js" / "local.js").exists()
    app = tmp_path / "app"
    with zipfile.ZipFile(out / "py" / "truco.zip") as zf:
        names = zf.namelist()
        assert "runs/parejas_senas/last.npz" in names and "demo/local.py" in names
        assert not any(n.endswith(".pt") for n in names)
        zf.extractall(app)
    code = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "from demo.local import setup, handle\n"
        "setup(hands=100)\n"
        "agents = json.loads(handle('GET', '/api/agents'))\n"
        "game = json.loads(handle('POST', '/api/games', json.dumps({'mode': 'watch', 'target': 15})))\n"
        "step = json.loads(handle('POST', '/api/games/' + game['data']['id'] + '/advance', json.dumps({'steps': 20})))\n"
        "import demo\n"
        "assert demo.__file__.startswith(sys.path[0])\n"
        "print(json.dumps([agents['status'], game['status'], step['status']]))\n"
    )
    code = f"import sys\nsys.path.insert(0, {str(app)!r})\n" + code
    proc = subprocess.run([sys.executable, "-c", code], cwd=app, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [200, 201, 200]
