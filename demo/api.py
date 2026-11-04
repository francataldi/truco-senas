from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import truco

from .catalog import HUMAN, Catalog, SpecError
from .dictionary import DictionaryService, dictionary_file
from .session import GameError, GameFactory, GameStore

WEB_DIR = Path(truco.__file__).resolve().parent / "web"


class WebFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


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


class NewGame(BaseModel):
    mode: Literal["play", "watch"] = "play"
    seats: list[str] | None = Field(default=None, min_length=4, max_length=4)
    target: int = 30
    seed: int | None = None
    open_cards: bool = False
    spy_prob: float = Field(default=0.2, ge=0.0, le=1.0)
    scramble: list[int] = Field(default_factory=list, max_length=2)


class ActionIn(BaseModel):
    action: int = Field(ge=0)
    seq: int | None = None


class AdvanceIn(BaseModel):
    steps: int = Field(default=64, ge=1, le=500)
    seq: int | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if settings.torch_threads:
        import torch

        torch.set_num_threads(settings.torch_threads)
    catalog = Catalog(settings.roots, base=settings.base, allowed=settings.allowed)
    dictionaries = DictionaryService(catalog, hands=settings.dictionary_hands, seed=settings.seed)
    factory = GameFactory(catalog, dictionaries)
    store = GameStore(settings.max_games)

    def defaults() -> dict[str, list[str]]:
        found = catalog.default_specs()
        partner = settings.partner or found["partner"]
        rivals = settings.rivals or found["rivals"]
        team_a = settings.team_a or found["team_a"]
        team_b = settings.team_b or found["team_b"]
        return {
            "play": [HUMAN, rivals, partner, rivals],
            "watch": [team_a, team_b, team_a, team_b],
        }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.prefetch:
            wanted = []
            for specs in defaults().values():
                for spec in specs:
                    if spec != HUMAN:
                        try:
                            wanted.append(catalog.resolve(spec))
                        except SpecError as exc:
                            print(f"aviso: {exc}", flush=True)
            wanted += [info for info in catalog.discover() if info.kind == "ppo"]
            dictionaries.prefetch(list(dict.fromkeys(wanted)))
        yield

    app = FastAPI(title="Truco con señas", version="1.0", lifespan=lifespan)
    app.state.catalog = catalog
    app.state.dictionaries = dictionaries
    app.state.store = store
    app.state.settings = settings

    @app.exception_handler(GameError)
    async def game_error(request: Request, exc: GameError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "games": len(store)}

    @app.get("/api/agents")
    def agents() -> dict:
        items = []
        for info in catalog.discover():
            item = info.to_dict()
            path = dictionary_file(info)
            item["dictionary"] = "archivo" if path is not None and path.is_file() else ("mudo" if info.mute else "calculado")
            item["dictionary_ready"] = dictionaries.ready(info)
            items.append(item)
        return {"agents": items, "defaults": defaults(), "targets": [15, 30]}

    @app.get("/api/dictionary")
    def dictionary(spec: str) -> dict:
        try:
            info = catalog.resolve(spec)
        except SpecError as exc:
            raise HTTPException(422, str(exc)) from exc
        if info.kind == HUMAN:
            raise HTTPException(422, "el humano no tiene diccionario propio")
        return {"spec": info.spec, "label": info.label, **dictionaries.get(info).to_dict()}

    @app.post("/api/games", status_code=201)
    def new_game(body: NewGame) -> dict:
        specs = body.seats or defaults()[body.mode]
        game = factory.create(
            body.mode,
            specs,
            target=body.target,
            seed=body.seed,
            open_cards=body.open_cards,
            spy_prob=body.spy_prob,
            scramble=tuple(sorted(set(body.scramble))),
        )
        store.add(game)
        return {**game.describe(), "seq": game.seq, "events": game.since(0)}

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str, since: int = 0) -> dict:
        game = store.get(game_id)
        with game.lock:
            return {**game.describe(), "seq": game.seq, "events": game.since(since), "state": game.state()}

    @app.post("/api/games/{game_id}/action")
    def act(game_id: str, body: ActionIn) -> dict:
        game = store.get(game_id)
        events = game.act(body.action, expected_seq=body.seq)
        return {"seq": game.seq, "events": events}

    @app.post("/api/games/{game_id}/advance")
    def advance(game_id: str, body: AdvanceIn | None = None) -> dict:
        body = body or AdvanceIn()
        game = store.get(game_id)
        events = game.advance(body.steps, expected_seq=body.seq)
        return {"seq": game.seq, "events": events}

    @app.get("/api/games/{game_id}/advice")
    def advice(game_id: str) -> dict:
        game = store.get(game_id)
        return {"advice": game.advice()}

    @app.delete("/api/games/{game_id}", status_code=204)
    def delete_game(game_id: str) -> None:
        store.remove(game_id)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/", WebFiles(directory=WEB_DIR, check_dir=False), name="web")
    return app
