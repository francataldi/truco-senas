from __future__ import annotations

import contextlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from agents.base import Agent, RandomAgent
from agents.heuristic import HeuristicAgent

HUMAN = "human"
BUILTIN = {
    "heuristic": ("heuristic", "Heurístico", "Reglas fijas y las señas clásicas del truco."),
    "heuristic-mudo": ("heuristic", "Heurístico mudo", "Las mismas reglas, pero sin hacer señas."),
    "random": ("random", "Aleatorio", "Elige al azar entre las acciones legales."),
}
PATH_KINDS = ("ppo", "ppo-greedy")
HEADS_UP_KINDS = ("cfr", "cfr-greedy")
DEFAULT_REWARD_SCALE = 4.0


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class AgentInfo:
    spec: str
    kind: str
    family: str
    label: str
    description: str
    path: Path | None = None
    details: dict = field(default_factory=dict, compare=False)

    @property
    def greedy(self) -> bool:
        return self.kind.endswith("-greedy")

    @property
    def mute(self) -> bool:
        if self.kind == "heuristic-mudo":
            return True
        return self.family == "ppo" and not self.details.get("signals", True)

    def to_dict(self) -> dict:
        return {
            "spec": self.spec,
            "kind": self.kind,
            "family": self.family,
            "label": self.label,
            "description": self.description,
            "mute": self.mute,
            **{k: v for k, v in self.details.items() if k != "reward_scale"},
        }


HUMAN_INFO = AgentInfo(HUMAN, HUMAN, HUMAN, "Vos", "Jugás vos.")


def _last_eval(run_dir: Path) -> dict | None:
    path = run_dir / "eval.jsonl"
    if not path.exists():
        return None
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def _reward_scale(run_dir: Path) -> float:
    path = run_dir / "config.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text(encoding="utf-8")).get("reward_scale", DEFAULT_REWARD_SCALE))
        except (ValueError, json.JSONDecodeError):
            pass
    return DEFAULT_REWARD_SCALE


class Catalog:
    def __init__(self, roots: list[Path], base: Path | None = None, allowed: list[Path] | None = None) -> None:
        self.base = (base or Path.cwd()).resolve()
        self.roots = [self._absolute(Path(r)) for r in roots]
        self.allowed = {self._absolute(Path(p)) for p in (allowed or [])}
        self._nets: dict[Path, tuple] = {}
        self._infos: dict[str, AgentInfo] = {}
        self._lock = threading.Lock()

    def _absolute(self, path: Path) -> Path:
        return (path if path.is_absolute() else self.base / path).resolve()

    def display(self, path: Path) -> str:
        try:
            return path.relative_to(self.base).as_posix()
        except ValueError:
            return path.as_posix()

    def _permitted(self, path: Path) -> bool:
        if path in self.allowed:
            return True
        return any(path.is_relative_to(root) for root in self.roots)

    def discover(self) -> list[AgentInfo]:
        found: list[AgentInfo] = []
        seen: set[str] = set()
        paths: list[Path] = sorted(self.allowed)
        for root in self.roots:
            if root.is_dir():
                paths += sorted(root.rglob("last.pt"))
        for path in paths:
            if path.suffix != ".pt":
                continue
            for kind in PATH_KINDS:
                spec = f"{kind}:{self.display(path)}"
                if spec in seen:
                    continue
                seen.add(spec)
                with contextlib.suppress(SpecError):
                    found.append(self.resolve(spec))
        found += [self.resolve(spec) for spec in BUILTIN]
        order = {"ppo": 0, "heuristic": 1, "random": 2}
        return sorted(found, key=lambda info: order.get(info.family, 9))

    def resolve(self, spec: str) -> AgentInfo:
        spec = spec.strip()
        with self._lock:
            cached = self._infos.get(spec)
        if cached is not None:
            return cached
        info = self._resolve(spec)
        with self._lock:
            self._infos[spec] = info
        return info

    def _resolve(self, spec: str) -> AgentInfo:
        if spec == HUMAN:
            return HUMAN_INFO
        if spec in BUILTIN:
            family, label, description = BUILTIN[spec]
            return AgentInfo(spec, spec, family, label, description)
        kind, sep, raw = spec.partition(":")
        if kind in HEADS_UP_KINDS:
            raise SpecError(f"{spec!r} es una estrategia MCCFR de 1 contra 1 y esta mesa es de parejas")
        if kind not in PATH_KINDS or not sep or not raw:
            raise SpecError(f"agente desconocido: {spec!r}")
        path = self._absolute(Path(raw))
        if not path.is_file():
            raise SpecError(f"no existe el archivo {raw!r}")
        if not self._permitted(path):
            raise SpecError(f"{raw!r} está fuera de las carpetas permitidas")
        return self._ppo_info(f"{kind}:{self.display(path)}", kind, path)

    def _ppo_info(self, spec: str, kind: str, path: Path) -> AgentInfo:
        net, rules, extra = self.load_net(path)
        if rules.num_players != 4:
            raise SpecError(f"{path.name} es un checkpoint de {rules.num_players} jugadores y la mesa es de 4")
        run = path.parent.name if path.name == "last.pt" else f"{path.parent.name}/{path.stem}"
        details = {
            "run": run,
            "iteration": extra.get("iteration"),
            "signals": rules.signals,
            "spy_prob": rules.spy_prob,
            "signal_vocab": rules.signal_vocab,
            "hidden": net.cfg.hidden,
            "depth": net.cfg.depth,
            "reward_scale": _reward_scale(path.parent),
        }
        evaluation = _last_eval(path.parent)
        if evaluation and "vs_heuristic" in evaluation:
            details["win_vs_heuristic"] = evaluation["vs_heuristic"].get("win_rate")
        parts = [f"Red {net.cfg.depth}×{net.cfg.hidden}"]
        if details["iteration"]:
            parts.append(f"{details['iteration']} iteraciones")
        if details.get("win_vs_heuristic") is not None:
            parts.append(f"gana {round(100 * details['win_vs_heuristic'])}% al heurístico")
        parts.append("con señas" if rules.signals else "entrenada sin señas")
        if kind == "ppo-greedy":
            parts.append("elige siempre la acción más probable")
        label = f"PPO{' greedy' if kind == 'ppo-greedy' else ''} · {run}"
        return AgentInfo(spec, kind, "ppo", label, " · ".join(parts) + ".", path, details)

    def load_net(self, path: Path) -> tuple:
        from agents.ppo.net import load_checkpoint

        with self._lock:
            cached = self._nets.get(path)
        if cached is None:
            try:
                cached = load_checkpoint(str(path))
            except Exception as exc:
                raise SpecError(f"no se pudo cargar {path.name}: {exc}") from exc
            with self._lock:
                self._nets[path] = cached
        return cached

    def make(self, info: AgentInfo, seed: int | None = None) -> Agent:
        if info.kind == "heuristic":
            return HeuristicAgent(seed=seed)
        if info.kind == "heuristic-mudo":
            return HeuristicAgent(seed=seed, use_signals=False)
        if info.kind == "random":
            return RandomAgent(seed=seed)
        if info.family == "ppo":
            from agents.ppo.agent import PPOAgent

            net, rules, _ = self.load_net(info.path)
            return PPOAgent(net, greedy=info.greedy, seed=seed, rules=rules)
        raise SpecError(f"{info.spec!r} no es un agente")

    def reward_scale(self, info: AgentInfo) -> float:
        return float(info.details.get("reward_scale", DEFAULT_REWARD_SCALE))

    def default_specs(self) -> dict[str, str]:
        ppo = [info for info in self.discover() if info.kind == "ppo"]
        with_signals = [info for info in ppo if info.details.get("signals")]
        preferred = [info for info in with_signals if info.details.get("run") == "parejas_senas"]
        partner = (preferred or with_signals or ppo or [self.resolve("heuristic")])[0].spec
        without = [info for info in ppo if not info.details.get("signals")]
        control = [info for info in without if info.details.get("run") == "parejas_sin_senas"]
        return {
            "partner": partner,
            "rivals": "heuristic",
            "team_a": partner,
            "team_b": (control or without)[0].spec if without else "heuristic",
        }
