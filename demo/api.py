from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import truco

from .catalog import SpecError
from .service import DemoService, ServiceError, Settings
from .session import GameError

__all__ = ["Settings", "create_app"]

WEB_DIR = Path(truco.__file__).resolve().parent / "web"


class WebFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class NewGame(BaseModel):
    mode: Literal["play", "watch"] = "play"
    seats: list[str] | None = Field(default=None, min_length=2, max_length=4)
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
    service = DemoService(settings)
    settings = service.settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.prefetch:
            service.prefetch()
        yield

    app = FastAPI(title="Truco con señas", version="1.0", lifespan=lifespan)
    app.state.service = service
    app.state.catalog = service.catalog
    app.state.dictionaries = service.dictionaries
    app.state.store = service.store
    app.state.settings = settings

    @app.exception_handler(GameError)
    async def game_error(request: Request, exc: GameError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.get("/api/health")
    def health() -> dict:
        return service.health()

    @app.get("/api/agents")
    def agents() -> dict:
        return service.agents()

    @app.get("/api/dictionary")
    def dictionary(spec: str) -> dict:
        return service.dictionary(spec)

    @app.post("/api/games", status_code=201)
    def new_game(body: NewGame) -> dict:
        try:
            return service.new_game(body.mode, body.seats, body.target, body.seed, body.open_cards, body.spy_prob, body.scramble)
        except SpecError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str, since: int = 0) -> dict:
        return service.get_game(game_id, since)

    @app.post("/api/games/{game_id}/action")
    def act(game_id: str, body: ActionIn) -> dict:
        return service.act(game_id, body.action, body.seq)

    @app.post("/api/games/{game_id}/advance")
    def advance(game_id: str, body: AdvanceIn | None = None) -> dict:
        body = body or AdvanceIn()
        return service.advance(game_id, body.steps, body.seq)

    @app.get("/api/games/{game_id}/advice")
    def advice(game_id: str) -> dict:
        return service.advice(game_id)

    @app.delete("/api/games/{game_id}", status_code=204)
    def delete_game(game_id: str) -> None:
        service.remove(game_id)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/", WebFiles(directory=WEB_DIR, check_dir=False), name="web")
    return app
