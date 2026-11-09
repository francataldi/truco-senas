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
    "heuristic": (
        "heuristic",
        "Jugador clásico",
        "Un jugador de reglas fijas que usa las señas de siempre del truco.",
        "No es una inteligencia artificial: lo programó una persona, regla por regla. "
        "Canta envido casi siempre que tiene 27 o más, canta truco cuando tiene cartas altas o su compañero le avisó que tiene una, "
        "y muy de vez en cuando miente. Sus señas son las de siempre: cejas por el ancho de espada, guiño por el de basto, "
        "boca torcida por los sietes bravos. Las IA se midieron contra él.",
    ),
    "heuristic-mudo": (
        "heuristic",
        "Jugador clásico sin señas",
        "El jugador clásico, pero sin hacer ni mirar señas.",
        "Mismas reglas que el clásico, pero callado: pone siempre la misma cara y no se fija en las señas de su compañero. "
        "Sirve para ver cuánto le ayudan las señas al clásico.",
    ),
    "random": (
        "random",
        "Jugador al azar",
        "Juega cualquier cosa, sin pensar.",
        "Elige entre las jugadas permitidas como si tirara un dado. "
        "Es el rival más fácil, y sirve para comprobar que los demás de verdad aprendieron algo.",
    ),
}
PATH_KINDS = ("ppo", "ppo-greedy")
POLICY_SUFFIXES = (".pt", ".npz")
HEADS_UP_KINDS = ("cfr", "cfr-greedy")
HEADS_UP_FILES = ("heads_up.npz",)
BUILTIN_PLAYERS = {"heuristic": [2, 4], "heuristic-mudo": [4], "random": [2, 4]}
DEFAULT_REWARD_SCALE = 4.0
RUN_NAMES = {
    "parejas_senas": "IA con señas",
    "parejas_sin_senas": "IA sin señas",
    "parejas_espias": "IA entrenada con espías",
    "duelo": "IA de mano a mano",
}


class SpecError(ValueError):
    pass


def summarize_ppo(rules) -> str:
    if rules.num_players == 2:
        return "Una IA que aprendió a jugar mano a mano."
    if not rules.signals:
        return "Una IA que juega sin hacer señas y sin mirar las de nadie."
    if rules.spy_prob >= 0.5:
        return "Una IA que aprendió con rivales que le espiaban las señas."
    return "Una IA que se inventó sus propias señas para hablar con su compañero."


def describe_ppo(run: str, rules, win: float | None, greedy: bool) -> str:
    parts = [
        "Aprendió a jugar sola, sin que nadie le explique cómo: "
        "jugó millones de manos en pareja con copias de sí misma y se fue quedando con lo que la hacía ganar."
    ]
    if rules.num_players == 2:
        parts[0] = parts[0].replace("en pareja con copias de sí misma", "mano a mano contra copias de sí misma")
        parts.append("Juega solo mano a mano, sin señas.")
    elif not rules.signals:
        parts.append(
            "Lo hizo en un truco donde no existen las señas, así que juega solo con sus cartas y con lo que pasa en la mesa "
            "(en la partida pone siempre la misma cara). Sirve para medir cuánto ganan las otras gracias a las señas."
        )
    elif rules.spy_prob >= 0.5:
        parts.append(
            "Le hace gestos a su compañero antes de cada mano, pero se crió en mesas donde los rivales veían la mitad de esos gestos, "
            "así que avisar sus cartas también le servía al rival."
        )
        if run == "parejas_espias":
            parts.append("Por eso casi dejó de hacerlo: sus gestos dicen muy poco, apenas una pista de si tiene mucho envido.")
    else:
        parts.append(
            "Antes de cada mano le hace un gesto a su compañero (cejas, guiño, boca torcida…) para contarle qué cartas tiene. "
            "Qué significa cada gesto lo inventó ella, así que no es el código de siempre del truco: "
            "en la partida, el panel de señas muestra qué quiere decir cada uno."
        )
        if run == "parejas_senas":
            parts.append("Igual llegó sola a algo parecido a las señas de la gente: con «boca a la derecha» avisa que tiene un ancho.")
    if greedy:
        parts.append(
            "Esta versión «sin dudar» hace siempre la jugada que más le gusta; la normal a veces elige otra, "
            "como si tirara un dado cargado, y por eso es más difícil de leer."
        )
    if win is not None:
        who = "la versión normal gana" if greedy else "gana"
        wins = round(100 * win)
        if rules.signals or rules.num_players == 2:
            parts.append(f"Contra el jugador clásico, {who} {wins} de cada 100 partidos.")
        else:
            parts.append(
                f"Contra el jugador clásico, en una mesa donde nadie hace señas, {who} {wins} de cada 100 partidos. "
                "Ese número no se compara directo con el de las otras IA."
            )
    return " ".join(parts)


def describe_cfr(win: float | None, greedy: bool) -> str:
    parts = [
        "No es una red neuronal: es una tabla enorme que dice qué conviene hacer en cada situación, "
        "armada con un método de teoría de juegos (MCCFR) que también se usa para el póker. "
        "Jugó un millón de repartos de práctica contra sí mismo y, en cada situación, anotó cuánto se arrepentía "
        "de no haber elegido otra jugada; de a poco fue eligiendo más seguido lo que menos arrepentimiento le dejaba.",
        "Para que la tabla entre en memoria agrupa situaciones parecidas, como «tengo una carta altísima y 28 de envido». "
        "Juega solo mano a mano, sin señas.",
    ]
    if greedy:
        parts.append(
            "Esta versión «sin dudar» hace siempre la jugada que más le gusta. La normal mezcla a propósito, "
            "como un buen jugador de póker, para que no lo puedan leer."
        )
    if win is not None:
        who = "la versión normal gana" if greedy else "gana"
        parts.append(f"Contra el jugador clásico, mano a mano, {who} {round(100 * win)} de cada 100 partidos.")
    return " ".join(parts)


@dataclass(frozen=True)
class AgentInfo:
    spec: str
    kind: str
    family: str
    label: str
    description: str
    path: Path | None = None
    details: dict = field(default_factory=dict, compare=False)
    summary: str = ""

    @property
    def greedy(self) -> bool:
        return self.kind.endswith("-greedy")

    @property
    def players(self) -> list[int]:
        return list(self.details.get("players", [2, 4] if self.kind == HUMAN else [4]))

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
            "summary": self.summary,
            "mute": self.mute,
            "players": self.players,
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
                for suffix in POLICY_SUFFIXES:
                    paths += sorted(root.rglob(f"last{suffix}"))
                for name in HEADS_UP_FILES:
                    paths += sorted(root.rglob(name))
        for path in paths:
            if path.suffix not in POLICY_SUFFIXES:
                continue
            kinds = HEADS_UP_KINDS if path.name in HEADS_UP_FILES else PATH_KINDS
            for kind in kinds:
                spec = f"{kind}:{self.display(path)}"
                if spec in seen:
                    continue
                seen.add(spec)
                with contextlib.suppress(SpecError):
                    found.append(self.resolve(spec))
        found += [self.resolve(spec) for spec in BUILTIN]
        order = {"ppo": 0, "cfr": 1, "heuristic": 2, "random": 3}
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
            family, label, summary, description = BUILTIN[spec]
            return AgentInfo(spec, spec, family, label, description, details={"players": BUILTIN_PLAYERS[spec]}, summary=summary)
        kind, sep, raw = spec.partition(":")
        if kind not in PATH_KINDS + HEADS_UP_KINDS or not sep or not raw:
            raise SpecError(f"agente desconocido: {spec!r}")
        path = self._absolute(Path(raw))
        if not path.is_file():
            raise SpecError(f"no existe el archivo {raw!r}")
        if not self._permitted(path):
            raise SpecError(f"{raw!r} está fuera de las carpetas permitidas")
        if kind in HEADS_UP_KINDS:
            return self._cfr_info(f"{kind}:{self.display(path)}", kind, path)
        return self._ppo_info(f"{kind}:{self.display(path)}", kind, path)

    def _cfr_info(self, spec: str, kind: str, path: Path) -> AgentInfo:
        policy = self.load_policy(path)
        if policy.rules.num_players != 2:
            raise SpecError(f"{path.name} no es una estrategia de 1 contra 1")
        sidecar = path.with_suffix(".json")
        meta = {}
        if sidecar.is_file():
            with contextlib.suppress(ValueError):
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
        win = (meta.get("vs_heuristic") or {}).get("win_rate")
        details = {"players": [2], "signals": False, "infosets": len(policy), "iterations": meta.get("iterations", policy.meta.get("iterations"))}
        if win is not None:
            details["win_vs_heuristic"] = win
        greedy = kind == "cfr-greedy"
        label = f"Bot de teoría de juegos{' · sin dudar' if greedy else ''}"
        summary = "Un bot que busca la estrategia más difícil de aprovechar, como en el póker."
        return AgentInfo(spec, kind, "cfr", label, describe_cfr(win, greedy), path, details, summary=summary)

    def load_policy(self, path: Path):
        with self._lock:
            cached = self._nets.get(path)
        if cached is None:
            try:
                from agents.cfr import Policy

                cached = Policy.load(path)
            except Exception as exc:
                raise SpecError(f"no se pudo cargar {path.name}: {exc}") from exc
            with self._lock:
                self._nets[path] = cached
        return cached

    def _ppo_info(self, spec: str, kind: str, path: Path) -> AgentInfo:
        net, rules, extra = self.load_net(path)
        run = path.parent.name if path.stem == "last" else f"{path.parent.name}/{path.stem}"
        details = {
            "run": run,
            "iteration": extra.get("iteration"),
            "players": [rules.num_players],
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
        duplicate = path.parent / "vs_heuristico.json"
        if duplicate.is_file():
            with contextlib.suppress(ValueError):
                details["win_vs_heuristic"] = json.loads(duplicate.read_text(encoding="utf-8")).get("win_rate", details.get("win_vs_heuristic"))
        name = RUN_NAMES.get(path.parent.name if path.stem == "last" else "", f"IA · {run}")
        label = f"{name}{' · sin dudar' if kind == 'ppo-greedy' else ''}"
        description = describe_ppo(run, rules, details.get("win_vs_heuristic"), kind == "ppo-greedy")
        return AgentInfo(spec, kind, "ppo", label, description, path, details, summary=summarize_ppo(rules))

    def load_net(self, path: Path) -> tuple:
        with self._lock:
            cached = self._nets.get(path)
        if cached is None:
            try:
                if path.suffix == ".npz":
                    from agents.ppo.numpy_net import load_numpy_policy

                    cached = load_numpy_policy(path)
                else:
                    from agents.ppo.net import load_checkpoint

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
        if info.family == "cfr":
            from agents.cfr import CFRAgent

            return CFRAgent(self.load_policy(info.path), greedy=info.greedy, seed=seed)
        if info.family == "ppo":
            net, rules, _ = self.load_net(info.path)
            if info.path.suffix == ".npz":
                from agents.ppo.numpy_net import NumpyPPOAgent

                return NumpyPPOAgent(net, greedy=info.greedy, seed=seed, rules=rules)
            from agents.ppo.agent import PPOAgent

            return PPOAgent(net, greedy=info.greedy, seed=seed, rules=rules)
        raise SpecError(f"{info.spec!r} no es un agente")

    def reward_scale(self, info: AgentInfo) -> float:
        return float(info.details.get("reward_scale", DEFAULT_REWARD_SCALE))

    def default_specs(self) -> dict[str, str]:
        found = self.discover()
        ppo = [info for info in found if info.kind == "ppo" and 4 in info.players]
        duel = [info for info in found if info.kind in ("ppo", "cfr") and 2 in info.players]
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
            "duel": duel[0].spec if duel else "heuristic",
        }
