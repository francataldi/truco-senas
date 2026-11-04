from __future__ import annotations

import itertools
import json
import math
import threading
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from agents.signal_analysis import TOP_CARD_NAMES, collect_signals, summarize, top_card_bucket
from truco.cards import N_CARDS
from truco.features import FEATURE_NAMES, hand_features
from truco.rules import Rules

from .catalog import AgentInfo, Catalog

GESTURES = (
    ("cejas", "levantó las cejas", "levantaste las cejas"),
    ("guiño", "guiñó un ojo", "guiñaste un ojo"),
    ("boca a la derecha", "torció la boca a la derecha", "torciste la boca a la derecha"),
    ("boca a la izquierda", "torció la boca a la izquierda", "torciste la boca a la izquierda"),
    ("labio mordido", "se mordió el labio", "te mordiste el labio"),
    ("lengua afuera", "sacó la lengua", "sacaste la lengua"),
    ("ojos cerrados", "cerró los ojos", "cerraste los ojos"),
    ("beso", "tiró un beso", "tiraste un beso"),
)
FEATURE_LABELS = {
    "ancho_espada": "ancho de espada",
    "ancho_basto": "ancho de basto",
    "siete_espada": "7 de espada",
    "siete_oro": "7 de oro",
    "algun_tres": "algún tres",
    "envido_27": "27 o más de envido",
    "envido_30": "30 o más de envido",
    "dos_bravas": "dos cartas bravas",
}
CONCEPTS = (
    ("ancho de espada", "feature", ("ancho_espada",)),
    ("ancho de basto", "feature", ("ancho_basto",)),
    ("7 de espada", "feature", ("siete_espada",)),
    ("7 de oro", "feature", ("siete_oro",)),
    ("un ancho", "top", ("ancho",)),
    ("siete bravo", "top", ("siete bravo",)),
    ("un tres", "top", ("tres",)),
    ("un dos", "top", ("dos",)),
    ("30 o más de envido", "feature", ("envido_30",)),
    ("27 o más de envido", "feature", ("envido_27",)),
    ("dos cartas bravas", "feature", ("dos_bravas",)),
    ("no tengo nada", "top", ("falso o 12", "11 o 10", "baja")),
)
UNCLEAR = "sin significado claro"
UNUSED = "no la usa"
MIN_P = 0.35
MIN_LIFT = 1.25
MIN_SCORE = 0.1


def concept_holds(label: str | None, cards: list[int]) -> bool | None:
    for name, kind, keys in CONCEPTS:
        if name == label:
            if kind == "feature":
                features = hand_features(cards)
                return bool(any(features[FEATURE_NAMES.index(k)] > 0.5 for k in keys))
            return TOP_CARD_NAMES[top_card_bucket(cards)] in keys
    return None


def gesture(token: int) -> tuple[str, str, str]:
    if token < len(GESTURES):
        return GESTURES[token]
    return (f"seña {token}", f"hizo la seña {token}", f"hiciste la seña {token}")


@lru_cache(maxsize=1)
def base_rates() -> tuple[np.ndarray, np.ndarray]:
    feats = []
    tops = []
    for combo in itertools.combinations(range(N_CARDS), 3):
        cards = list(combo)
        feats.append(hand_features(cards))
        tops.append(top_card_bucket(cards))
    top = np.bincount(np.array(tops), minlength=len(TOP_CARD_NAMES)) / len(tops)
    return np.array(feats).mean(0), top


@dataclass
class TokenMeaning:
    token: int
    gesture: str
    label: str
    confidence: float | None
    frequency: float
    concepts: list[dict] = field(default_factory=list)
    features: list[dict] = field(default_factory=list)


@dataclass
class Dictionary:
    source: str
    hands: int
    mi_bits: float | None
    entropy_bits: float | None
    tokens: list[TokenMeaning]
    path: str | None = None

    def label(self, token: int | None) -> str | None:
        if token is None or not 0 <= token < len(self.tokens):
            return None
        return self.tokens[token].label

    def to_dict(self) -> dict:
        return asdict(self)


def _concept_probability(kind: str, keys: tuple[str, ...], features: np.ndarray, top: np.ndarray) -> float:
    if kind == "feature":
        return float(sum(features[FEATURE_NAMES.index(k)] for k in keys))
    return float(sum(top[TOP_CARD_NAMES.index(k)] for k in keys))


def from_summary(summary: dict, source: str, vocab: int = 8, path: str | None = None) -> Dictionary:
    cond = np.asarray(summary.get("p_rasgo_dada_seña") or np.zeros((vocab, len(FEATURE_NAMES))), dtype=float)
    vocab = max(vocab, cond.shape[0])
    exact_feat, exact_top = base_rates()
    base_feat = np.asarray(summary.get("p_rasgo_base") or exact_feat, dtype=float)
    entries = {int(e["seña"]): e for e in summary.get("diccionario", [])}
    freq = np.zeros(vocab)
    top_given = np.zeros((vocab, len(TOP_CARD_NAMES)))
    for t, entry in entries.items():
        if t < vocab:
            freq[t] = float(entry.get("frecuencia", 0.0))
            dist = entry.get("p_carta_mas_alta", {})
            top_given[t] = [float(dist.get(name, 0.0)) for name in TOP_CARD_NAMES]
    base_top = freq @ top_given if freq.sum() > 0 else exact_top
    tokens = []
    for t in range(vocab):
        name = gesture(t)[0]
        if freq[t] <= 0 or t >= cond.shape[0]:
            tokens.append(TokenMeaning(t, name, UNUSED, None, 0.0))
            continue
        scored = []
        for label, kind, keys in CONCEPTS:
            p = min(1.0, _concept_probability(kind, keys, cond[t], top_given[t]))
            base = _concept_probability(kind, keys, base_feat, base_top)
            lift = p / max(base, 1e-9)
            score = p * math.log2(lift) if p >= MIN_P and lift >= MIN_LIFT else 0.0
            scored.append((score, label, p, lift))
        scored.sort(key=lambda item: -item[0])
        best_score, best_label, best_p, _ = scored[0]
        label = best_label if best_score >= MIN_SCORE else UNCLEAR
        concepts = [
            {"label": lab, "p": round(p, 3), "lift": round(lift, 2)}
            for score, lab, p, lift in scored[:3]
            if score > 0
        ]
        features = [
            {"name": feat, "label": FEATURE_LABELS[feat], "p": round(float(cond[t, i]), 3), "base": round(float(base_feat[i]), 3)}
            for i, feat in enumerate(FEATURE_NAMES)
        ]
        tokens.append(
            TokenMeaning(
                t,
                name,
                label,
                round(best_p, 3) if label != UNCLEAR else None,
                round(float(freq[t]), 4),
                concepts,
                features,
            )
        )
    return Dictionary(
        source=source,
        hands=int(summary.get("señas_observadas", 0)),
        mi_bits=summary.get("mi_carta_mas_alta_bits"),
        entropy_bits=summary.get("entropia_señas_bits"),
        tokens=tokens,
        path=path,
    )


def mute_dictionary(vocab: int = 8) -> Dictionary:
    tokens = [TokenMeaning(t, gesture(t)[0], "siempre la misma, no informa nada" if t == 0 else UNUSED, None, 1.0 if t == 0 else 0.0) for t in range(vocab)]
    return Dictionary("mudo", 0, 0.0, 0.0, tokens)


def dictionary_file(info: AgentInfo) -> Path | None:
    if info.path is None or info.family != "ppo":
        return None
    return info.path.parent / "senas" / "diccionario.json"


class _Slot:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.value: Dictionary | None = None


class DictionaryService:
    def __init__(self, catalog: Catalog, hands: int = 4000, seed: int = 0, rules: Rules | None = None) -> None:
        self.catalog = catalog
        self.hands = hands
        self.seed = seed
        self.rules = rules or Rules(num_players=4, signals=True)
        self._slots: dict[str, _Slot] = {}
        self._lock = threading.Lock()

    def key(self, info: AgentInfo) -> str:
        if info.path is not None:
            return f"{info.family}:{info.path}"
        return info.kind

    def ready(self, info: AgentInfo) -> bool:
        with self._lock:
            slot = self._slots.get(self.key(info))
        return slot is not None and slot.value is not None

    def get(self, info: AgentInfo) -> Dictionary:
        with self._lock:
            slot = self._slots.setdefault(self.key(info), _Slot())
        with slot.lock:
            if slot.value is None:
                slot.value = self._build(info)
            return slot.value

    def prefetch(self, infos: list[AgentInfo]) -> threading.Thread:
        def run() -> None:
            for info in infos:
                try:
                    self.get(info)
                except Exception as exc:
                    print(f"no se pudo preparar el diccionario de {info.spec}: {exc}", flush=True)

        thread = threading.Thread(target=run, name="diccionarios", daemon=True)
        thread.start()
        return thread

    def _build(self, info: AgentInfo) -> Dictionary:
        vocab = self.rules.signal_vocab
        if info.mute:
            return mute_dictionary(vocab)
        path = dictionary_file(info)
        if path is not None and path.is_file():
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                return from_summary(summary, "archivo", vocab, self.catalog.display(path))
            except (ValueError, KeyError, TypeError) as exc:
                print(f"{path} no se pudo leer ({exc}), lo calculo de nuevo", flush=True)
        agent = self.catalog.make(info, seed=self.seed)
        data = collect_signals(self.rules, [agent] * self.rules.num_players, self.hands, seed=self.seed)
        return from_summary(summarize(data, vocab), "calculado", vocab)
