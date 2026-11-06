import json
import random
import re

import pytest
import torch

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from agents.heuristic import HeuristicAgent
from agents.ppo.net import NetConfig, build_net, save_checkpoint
from agents.signal_analysis import collect_signals, summarize
from demo.api import Settings, create_app
from truco.rules import Rules

SIGNALS = Rules(num_players=4, signals=True)
MUTE = Rules(num_players=4, signals=False)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    torch.manual_seed(0)
    for name, rules in (("senas", SIGNALS), ("mudo", MUTE), ("con_archivo", SIGNALS)):
        folder = root / name
        folder.mkdir()
        save_checkpoint(folder / "last.pt", build_net(rules, NetConfig(hidden=32, depth=1)), rules, {"iteration": 7})
    data = collect_signals(SIGNALS, [HeuristicAgent(seed=i) for i in range(4)], 300, seed=0)
    (root / "con_archivo" / "senas").mkdir()
    summary = summarize(data, SIGNALS.signal_vocab)
    (root / "con_archivo" / "senas" / "diccionario.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    (root / "con_archivo" / "eval.jsonl").write_text(json.dumps({"iteration": 7, "vs_heuristic": {"win_rate": 0.61}}) + "\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def client(runs):
    settings = Settings(roots=[runs], base=runs.parent, dictionary_hands=150, prefetch=False, torch_threads=None)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def specs(client):
    agents = client.get("/api/agents").json()["agents"]
    found = {a["spec"].split("/")[-2] + ("-greedy" if a["kind"] == "ppo-greedy" else ""): a["spec"] for a in agents if a["family"] == "ppo"}
    return {"senas": found["senas"], "mudo": found["mudo"], "archivo": found["con_archivo"], "greedy": found["senas-greedy"]}


def last_state(events):
    return events[-1]["state"]


def play_out(client, game, rng, max_steps=2000):
    events = list(game["events"])
    seq = game["seq"]
    actions = 0
    for _ in range(max_steps):
        state = last_state(events)
        if state["phase"] == "match_over":
            return events, actions
        if state["phase"] == "hand_over":
            response = client.post(f"/api/games/{game['id']}/advance", json={"steps": 64, "seq": seq})
        else:
            assert state["turn"] == 0
            legal = [option["action"] for option in state["legal"]]
            assert legal
            response = client.post(f"/api/games/{game['id']}/action", json={"action": rng.choice(legal), "seq": seq})
            actions += 1
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["events"], "cada pedido tiene que avanzar la partida"
        events += body["events"]
        seq = body["seq"]
    raise AssertionError("la partida no terminó")


def test_agents_lists_checkpoints_and_builtins(client, specs):
    data = client.get("/api/agents").json()
    kinds = {a["spec"]: a for a in data["agents"]}
    assert {"heuristic", "heuristic-mudo", "random"} <= set(kinds)
    assert kinds[specs["senas"]]["signals"] is True
    assert kinds[specs["mudo"]]["mute"] is True
    assert kinds[specs["archivo"]]["win_vs_heuristic"] == 0.61
    assert kinds[specs["archivo"]]["dictionary"] == "archivo"
    assert data["defaults"]["play"][0] == "human"
    assert len(data["defaults"]["watch"]) == 4


def test_full_match_in_play_mode(client, specs):
    response = client.post(
        "/api/games",
        json={"mode": "play", "seats": ["human", "heuristic", specs["senas"], "random"], "seed": 3},
    )
    assert response.status_code == 201, response.text
    game = response.json()
    assert [s["role"] for s in game["seats"]] == ["Vos", "Rival derecho", "Compañero", "Rival izquierdo"]
    assert game["seats"][0]["language"] == game["seats"][2]["language"]
    events, actions = play_out(client, game, random.Random(0))
    assert actions > 0
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    final = last_state(events)
    assert final["phase"] == "match_over"
    assert max(final["scores"]) >= 30
    assert events[-1]["type"] == "match_end"
    assert events[-1]["winner"] == final["match"]["winner"]
    deals = sum(e["type"] == "deal" for e in events)
    assert deals == sum(e["type"] == "hand_end" for e in events) == final["hand"]
    replay = client.get(f"/api/games/{game['id']}").json()
    assert [e["seq"] for e in replay["events"]] == [e["seq"] for e in events]
    assert replay["state"] == final


def test_illegal_actions_are_rejected(client, specs):
    game = client.post("/api/games", json={"mode": "play", "seats": ["human", "heuristic", specs["senas"], "heuristic"], "seed": 11}).json()
    gid = game["id"]
    state = last_state(game["events"])
    assert state["phase"] == "signals" and state["turn"] == 0
    legal = {option["action"] for option in state["legal"]}
    illegal = next(a for a in range(19) if a not in legal)
    response = client.post(f"/api/games/{gid}/action", json={"action": illegal, "seq": game["seq"]})
    assert response.status_code == 422
    assert "ilegal" in response.json()["detail"]
    assert client.post(f"/api/games/{gid}/action", json={"action": 99}).status_code == 422
    assert client.post(f"/api/games/{gid}/action", json={"action": -1}).status_code == 422
    assert client.post(f"/api/games/{gid}/action", json={"action": min(legal), "seq": game["seq"] - 1}).status_code == 409
    assert client.get(f"/api/games/{gid}").json()["seq"] == game["seq"]
    seq = game["seq"]
    response = client.post(f"/api/games/{gid}/action", json={"action": min(legal), "seq": seq})
    assert response.status_code == 200
    events = game["events"] + response.json()["events"]
    seq = response.json()["seq"]
    while last_state(events)["phase"] != "hand_over":
        legal = last_state(events)["legal"]
        option = next((o for o in legal if o["kind"] in ("card", "mazo")), None) or next(o for o in legal if o["action"] == 8)
        body = client.post(f"/api/games/{gid}/action", json={"action": option["action"], "seq": seq}).json()
        events += body["events"]
        seq = body["seq"]
    response = client.post(f"/api/games/{gid}/action", json={"action": 0, "seq": seq})
    assert response.status_code == 409
    assert client.post("/api/games/no-existe/action", json={"action": 0}).status_code == 404
    assert client.delete(f"/api/games/{gid}").status_code == 204
    assert client.get(f"/api/games/{gid}").status_code == 404


def test_spectator_mode_runs_to_the_end(client, specs):
    seats = [specs["senas"], "heuristic", specs["senas"], specs["mudo"]]
    game = client.post("/api/games", json={"mode": "watch", "seats": seats, "seed": 5, "target": 15}).json()
    assert game["viewer"] is None
    assert client.post(f"/api/games/{game['id']}/action", json={"action": 0}).status_code == 409
    events = list(game["events"])
    seq = game["seq"]
    for _ in range(200):
        if last_state(events)["phase"] == "match_over":
            break
        body = client.post(f"/api/games/{game['id']}/advance", json={"steps": 7, "seq": seq}).json()
        assert body["events"]
        events += body["events"]
        seq = body["seq"]
    assert last_state(events)["phase"] == "match_over"
    assert max(last_state(events)["scores"]) >= 15
    assert client.post(f"/api/games/{game['id']}/advance", json={"steps": 5}).status_code == 409
    thoughts = [e["thought"] for e in events if e.get("thought")]
    assert thoughts
    for event in events:
        if event["type"] == "signal":
            assert event["token"] is not None
        thought = event.get("thought")
        if event["seat"] in (0, 2) and event["type"] in ("play", "call", "answer", "mazo"):
            assert thought["kind"] == "policy"
            assert abs(sum(o["p"] for o in thought["options"]) - 1) < 1e-3
            assert isinstance(thought["value"], float)
            assert len(thought["beliefs"]) == 8
        if event["seat"] == 3 and event["type"] == "signal":
            assert thought["kind"] == "mute"
            assert event["token"] == 0
        if event["seat"] == 1 and thought:
            assert thought["kind"] == "rules"
    hands = [e for e in events if e["type"] == "hand_end"]
    assert all(set(e["checks"]) == {"0", "1", "2", "3"} for e in hands)


def test_play_mode_hides_rival_information(client, specs):
    game = client.post(
        "/api/games",
        json={"mode": "play", "seats": ["human", specs["senas"], specs["senas"], specs["senas"]], "seed": 21, "spy_prob": 0.0},
    ).json()
    events, _ = play_out(client, game, random.Random(1))
    for event in events:
        state = event["state"]
        if state["phase"] in ("signals", "play"):
            for seat in (1, 3):
                info = state["seats"][seat]
                assert all(card is None for card, played in zip(info["cards"], info["played"]) if not played)
                assert info["envido"] is None
                assert state["signals"][seat]["token"] is None
            assert state["seats"][0]["envido"] is not None
            if event["seat"] in (1, 3):
                assert event["thought"] is None
            if event["type"] == "signal" and event["seat"] in (1, 3):
                assert event["token"] is None
        if state["phase"] in ("hand_over", "match_over"):
            assert all(card is not None for seat in state["seats"] for card in seat["cards"])
    partner = [e for e in events if e["seat"] == 2 and e["type"] == "signal"]
    assert partner and all(e["token"] is not None and e["thought"] for e in partner)


def test_partner_mind_stays_hidden_until_the_hand_ends(client, specs):
    game = client.post("/api/games", json={"mode": "play", "seats": ["human", "heuristic", specs["senas"], "heuristic"], "seed": 1}).json()
    events, _ = play_out(client, game, random.Random(3))
    hand, hidden = [], 0
    for event in events:
        hand.append(event)
        if event["type"] != "hand_end":
            continue
        partner_cards = [card["name"] for card in event["state"]["seats"][2]["cards"]]
        revealed = {mind["ref"]: mind for mind in event["minds"]}
        for before in hand[:-1]:
            thought = before["thought"]
            if before["seat"] != 2 or not thought:
                continue
            assert thought["hidden"] and thought["options"] is None and thought["value"] is None
            unplayed = [name for name, card in zip(partner_cards, before["state"]["seats"][2]["cards"]) if card is None]
            public = {k: v for k, v in before.items() if k not in ("state", "thought")}
            text = json.dumps([public, {k: v for k, v in thought.items() if k != "beliefs"}], ensure_ascii=False)
            assert not any(re.search(rf"(?<![0-9]){re.escape(name)}", text) for name in unplayed)
            full = revealed[thought["ref"]]
            assert full["options"] and full["chosen"] == thought["chosen"]
            hidden += 1
        hand = []
    assert hidden
    game = client.post("/api/games", json={"mode": "play", "seats": ["human", "heuristic", specs["senas"], "heuristic"], "seed": 1, "open_cards": True}).json()
    events, _ = play_out(client, game, random.Random(3))
    partner = [e["thought"] for e in events if e["seat"] == 2 and e["thought"]]
    assert partner and all(t["options"] and not t.get("hidden") for t in partner)
    game = client.post("/api/games", json={"mode": "play", "seats": ["human", "heuristic", "heuristic", "heuristic"], "seed": 1}).json()
    events, _ = play_out(client, game, random.Random(3))
    rules = [e["thought"] for e in events if e["seat"] == 2 and e["thought"]]
    assert rules and not any(t.get("hidden") for t in rules)


def test_open_cards_reveal_rivals(client, specs):
    game = client.post(
        "/api/games",
        json={"mode": "play", "seats": ["human", specs["senas"], "heuristic", specs["senas"]], "seed": 2, "open_cards": True},
    ).json()
    state = last_state(game["events"])
    assert all(card is not None for seat in state["seats"] for card in seat["cards"])
    rival_signals = [e for e in game["events"] if e["type"] == "signal" and e["seat"] in (1, 3)]
    assert all(e["token"] is not None and e["thought"] for e in rival_signals)


def test_invalid_games_are_rejected(client, specs, runs, tmp_path):
    outside = tmp_path / "afuera.pt"
    save_checkpoint(outside, build_net(SIGNALS, NetConfig(hidden=8, depth=1)), SIGNALS)
    cases = [
        {"mode": "play", "seats": ["human", "heuristic", "nadie", "heuristic"]},
        {"mode": "play", "seats": ["human", "heuristic", f"ppo:{outside}", "heuristic"]},
        {"mode": "play", "seats": ["human", "heuristic", "ppo:runs/no/existe.pt", "heuristic"]},
        {"mode": "play", "seats": ["heuristic", "heuristic", "heuristic", "heuristic"]},
        {"mode": "play", "seats": ["heuristic", "human", "heuristic", "heuristic"]},
        {"mode": "watch", "seats": ["human", "heuristic", "heuristic", "heuristic"]},
        {"mode": "watch", "seats": ["heuristic", "heuristic"]},
        {"mode": "play", "target": 21},
        {"mode": "jugar"},
        {"mode": "watch", "scramble": [2]},
        {"mode": "play", "spy_prob": 1.5},
    ]
    for body in cases:
        response = client.post("/api/games", json=body)
        assert response.status_code == 422, body
    response = client.post("/api/games", json={"mode": "play", "seats": ["human", "heuristic", "cfr:runs/cfr/heads_up.npz", "heuristic"]})
    assert response.status_code == 422
    assert "1 contra 1" in response.json()["detail"]


def test_dictionaries_from_file_computed_and_mute(client, specs):
    from_file = client.get("/api/dictionary", params={"spec": specs["archivo"]}).json()
    assert from_file["source"] == "archivo"
    assert from_file["tokens"][0]["label"] == "ancho de espada"
    assert from_file["tokens"][0]["gesture"] == "cejas"
    computed = client.get("/api/dictionary", params={"spec": "heuristic"}).json()
    assert computed["source"] == "calculado"
    assert [t["label"] for t in computed["tokens"]][:2] == ["ancho de espada", "ancho de basto"]
    assert computed["mi_bits"] > 0.8
    mute = client.get("/api/dictionary", params={"spec": specs["mudo"]}).json()
    assert mute["source"] == "mudo"
    noise = client.get("/api/dictionary", params={"spec": specs["senas"]}).json()
    assert noise["source"] == "calculado" and len(noise["tokens"]) == 8
    assert client.get("/api/dictionary", params={"spec": "nadie"}).status_code == 422


def test_advice_on_human_turn(client, specs):
    game = client.post("/api/games", json={"mode": "play", "seats": ["human", "heuristic", specs["greedy"], "heuristic"], "seed": 8}).json()
    advice = client.get(f"/api/games/{game['id']}/advice").json()["advice"]
    assert advice["kind"] == "policy"
    assert advice["seat"] == 0
    assert advice["context"].startswith("Elegís")
    assert all(o["action"] >= 11 for o in advice["options"])
    assert abs(sum(o["p"] for o in advice["options"]) - 1) < 1e-3
    assert advice["chosen"] == max(advice["options"], key=lambda o: o["p"])["action"]


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "Truco con señas" in page.text
    for path in ("/js/main.js", "/js/cards.js", "/css/theme.css", "/favicon.svg"):
        assert client.get(path).status_code == 200, path
    assert client.get("/api/health").json()["ok"] is True


def test_defaults_prefer_the_canonical_runs(tmp_path):
    from demo.catalog import Catalog

    torch.manual_seed(0)
    for name, rules in (("a_sin", MUTE), ("parejas_sin_senas", MUTE), ("b_con", SIGNALS), ("parejas_senas", SIGNALS)):
        folder = tmp_path / "runs" / name
        folder.mkdir(parents=True)
        save_checkpoint(folder / "last.pt", build_net(rules, NetConfig(hidden=8, depth=1)), rules)
    defaults = Catalog([tmp_path / "runs"], base=tmp_path).default_specs()
    assert defaults["partner"] == "ppo:runs/parejas_senas/last.pt"
    assert defaults["team_b"] == "ppo:runs/parejas_sin_senas/last.pt"
    assert defaults["rivals"] == "heuristic"
    empty = Catalog([tmp_path / "nada"], base=tmp_path).default_specs()
    assert empty == {"partner": "heuristic", "rivals": "heuristic", "team_a": "heuristic", "team_b": "heuristic"}
