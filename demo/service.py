from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .catalog import HUMAN, Catalog, SpecError
from .dictionary import DictionaryService, dictionary_file
from .session import GameError, GameFactory, GameStore

MODES = ("play", "watch")


class ServiceError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Settings:
    roots: list[Path] = field(default_factory=lambda: [Path("runs")])
    base: Path | None = None
    allowed: list[Path] = field(default_factory=list)
    partner: str | None = None
    rivals: str | None = None
    team_a: str | None = None
    team_b: str | None = None
    dictionary_hands: int = 4000
    prefetch: bool = True
    max_games: int = 64
    seed: int = 0
    torch_threads: int | None = 1


class DemoService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        if self.settings.torch_threads:
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None:
                torch.set_num_threads(self.settings.torch_threads)
        self.catalog = Catalog(self.settings.roots, base=self.settings.base, allowed=self.settings.allowed)
        self.dictionaries = DictionaryService(self.catalog, hands=self.settings.dictionary_hands, seed=self.settings.seed)
        self.factory = GameFactory(self.catalog, self.dictionaries)
        self.store = GameStore(self.settings.max_games)

    def defaults(self) -> dict[str, list[str]]:
        found = self.catalog.default_specs()
        s = self.settings
        partner = s.partner or found["partner"]
        rivals = s.rivals or found["rivals"]
        team_a = s.team_a or found["team_a"]
        team_b = s.team_b or found["team_b"]
        return {
            "play": [HUMAN, rivals, partner, rivals],
            "watch": [team_a, team_b, team_a, team_b],
        }

    def prefetch(self):
        wanted = []
        for specs in self.defaults().values():
            for spec in specs:
                if spec != HUMAN:
                    try:
                        wanted.append(self.catalog.resolve(spec))
                    except SpecError as exc:
                        print(f"aviso: {exc}", flush=True)
        wanted += [info for info in self.catalog.discover() if info.kind == "ppo"]
        return self.dictionaries.prefetch(list(dict.fromkeys(wanted)))

    def health(self) -> dict:
        return {"ok": True, "games": len(self.store)}

    def agents(self) -> dict:
        items = []
        for info in self.catalog.discover():
            item = info.to_dict()
            path = dictionary_file(info)
            item["dictionary"] = "archivo" if path is not None and path.is_file() else ("mudo" if info.mute else "calculado")
            item["dictionary_ready"] = self.dictionaries.ready(info)
            items.append(item)
        return {"agents": items, "defaults": self.defaults(), "targets": [15, 30]}

    def dictionary(self, spec: str) -> dict:
        try:
            info = self.catalog.resolve(spec)
        except SpecError as exc:
            raise ServiceError(422, str(exc)) from exc
        if info.kind == HUMAN:
            raise ServiceError(422, "el humano no tiene diccionario propio")
        return {"spec": info.spec, "label": info.label, **self.dictionaries.get(info).to_dict()}

    def new_game(
        self,
        mode: str = "play",
        seats: list[str] | None = None,
        target: int = 30,
        seed: int | None = None,
        open_cards: bool = False,
        spy_prob: float = 0.2,
        scramble: list[int] | tuple[int, ...] = (),
    ) -> dict:
        specs = seats or self.defaults()[mode]
        game = self.factory.create(
            mode,
            specs,
            target=target,
            seed=seed,
            open_cards=open_cards,
            spy_prob=spy_prob,
            scramble=tuple(sorted(set(scramble))),
        )
        self.store.add(game)
        return {**game.describe(), "seq": game.seq, "events": game.since(0)}

    def get_game(self, game_id: str, since: int = 0) -> dict:
        game = self.store.get(game_id)
        with game.lock:
            return {**game.describe(), "seq": game.seq, "events": game.since(since), "state": game.state()}

    def act(self, game_id: str, action: int, seq: int | None = None) -> dict:
        game = self.store.get(game_id)
        events = game.act(action, expected_seq=seq)
        return {"seq": game.seq, "events": events}

    def advance(self, game_id: str, steps: int = 64, seq: int | None = None) -> dict:
        game = self.store.get(game_id)
        events = game.advance(steps, expected_seq=seq)
        return {"seq": game.seq, "events": events}

    def advice(self, game_id: str) -> dict:
        return {"advice": self.store.get(game_id).advice()}

    def remove(self, game_id: str) -> None:
        self.store.remove(game_id)


GAME = re.compile(r"^/api/games/([^/]+)(/action|/advance|/advice)?$")


def _int(value, name: str, low: int | None = None, high: int | None = None, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ServiceError(422, f"{name} tiene que ser un número entero")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ServiceError(422, f"{name} tiene que ser un número entero") from exc
    if (low is not None and number < low) or (high is not None and number > high):
        raise ServiceError(422, f"{name} fuera de rango")
    return number


def _new_game_args(body: dict) -> dict:
    mode = body.get("mode", "play")
    if mode not in MODES:
        raise ServiceError(422, "mode tiene que ser play o watch")
    seats = body.get("seats")
    if seats is not None and (not isinstance(seats, list) or len(seats) != 4 or not all(isinstance(s, str) for s in seats)):
        raise ServiceError(422, "seats tiene que tener 4 agentes")
    spy = body.get("spy_prob", 0.2)
    try:
        spy = float(spy)
    except (TypeError, ValueError) as exc:
        raise ServiceError(422, "spy_prob tiene que ser un número") from exc
    if not 0.0 <= spy <= 1.0:
        raise ServiceError(422, "spy_prob tiene que estar entre 0 y 1")
    scramble = body.get("scramble") or []
    if not isinstance(scramble, list) or len(scramble) > 2:
        raise ServiceError(422, "scramble admite hasta 2 equipos")
    return {
        "mode": mode,
        "seats": seats,
        "target": _int(body.get("target"), "target", default=30),
        "seed": _int(body.get("seed"), "seed"),
        "open_cards": bool(body.get("open_cards", False)),
        "spy_prob": spy,
        "scramble": [_int(t, "scramble", 0, 1) for t in scramble],
    }


def route(service: DemoService, method: str, url: str, body: dict | None = None) -> tuple[int, object]:
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
    body = body or {}
    method = method.upper()
    try:
        if path == "/api/health" and method == "GET":
            return 200, service.health()
        if path == "/api/agents" and method == "GET":
            return 200, service.agents()
        if path == "/api/dictionary" and method == "GET":
            if "spec" not in query:
                raise ServiceError(422, "falta spec")
            return 200, service.dictionary(query["spec"])
        if path == "/api/games" and method == "POST":
            return 201, service.new_game(**_new_game_args(body))
        match = GAME.match(path)
        if match:
            game_id, tail = match.group(1), match.group(2)
            if tail is None and method == "GET":
                return 200, service.get_game(game_id, _int(query.get("since"), "since", 0, default=0))
            if tail is None and method == "DELETE":
                service.remove(game_id)
                return 204, None
            if tail == "/action" and method == "POST":
                return 200, service.act(game_id, _int(body.get("action"), "action", 0), _int(body.get("seq"), "seq"))
            if tail == "/advance" and method == "POST":
                return 200, service.advance(game_id, _int(body.get("steps"), "steps", 1, 500, 64), _int(body.get("seq"), "seq"))
            if tail == "/advice" and method == "GET":
                return 200, service.advice(game_id)
        raise ServiceError(404, f"no existe {method} {path}")
    except ServiceError as exc:
        return exc.status, {"detail": exc.message}
    except GameError as exc:
        return exc.status, {"detail": exc.message}
    except SpecError as exc:
        return 422, {"detail": str(exc)}


def handle_json(service: DemoService, method: str, url: str, body: str | None = None) -> str:
    try:
        payload = json.loads(body) if body else None
    except json.JSONDecodeError:
        return json.dumps({"status": 400, "data": {"detail": "el cuerpo no es JSON"}})
    status, data = route(service, method, url, payload if isinstance(payload, dict) else None)
    return json.dumps({"status": status, "data": data}, ensure_ascii=False)
