from __future__ import annotations

from pathlib import Path

from .service import DemoService, Settings, handle_json

_service: DemoService | None = None


def setup(root: str = "runs", hands: int = 1500, seed: int = 0) -> None:
    global _service
    _service = DemoService(Settings(roots=[Path(root)], prefetch=False, dictionary_hands=hands, torch_threads=None, seed=seed))


def handle(method: str, url: str, body: str | None = None) -> str:
    if _service is None:
        setup()
    return handle_json(_service, method, url, body)
