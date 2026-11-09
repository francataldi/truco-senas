from __future__ import annotations

import random

import numpy as np

from truco import actions as A
from truco.cards import STRENGTH
from truco.engine import TrucoHand
from truco.features import FEATURE_NAMES, hand_features
from truco.rules import Rules

from .base import Agent

TOP_CARD_NAMES = ("ancho", "siete bravo", "tres", "dos", "falso o 12", "11 o 10", "baja")


def top_card_bucket(cards: list[int]) -> int:
    best = max(STRENGTH[c] for c in cards)
    if best >= 13:
        return 0
    if best >= 11:
        return 1
    if best == 10:
        return 2
    if best == 9:
        return 3
    if best >= 7:
        return 4
    if best >= 5:
        return 5
    return 6


def random_scores(rng: random.Random, rules: Rules, zero_score_prob: float) -> tuple[int, int]:
    if zero_score_prob >= 1.0 or rng.random() < zero_score_prob:
        return (0, 0)
    return (rng.randrange(rules.target), rng.randrange(rules.target))


def collect_signals(rules: Rules, agents: list[Agent], hands: int, seed: int = 0, zero_score_prob: float = 1.0) -> dict:
    rng = random.Random(seed)
    tokens, feats, tops = [], [], []
    for _ in range(hands):
        scores = random_scores(rng, rules, zero_score_prob)
        hand = TrucoHand(rules, rng=random.Random(rng.random()), scores=scores, mano=rng.randrange(rules.num_players))
        while hand.signal_phase:
            seat = hand.current_player
            action = agents[seat].act(hand, seat)
            tokens.append(A.signal_token(action))
            feats.append(hand_features(hand.hands[seat]))
            tops.append(top_card_bucket(hand.hands[seat]))
            hand.step(action)
    return {"tokens": np.array(tokens), "features": np.array(feats), "top": np.array(tops)}


def _codes(x: np.ndarray) -> tuple[np.ndarray, int]:
    values, codes = np.unique(np.asarray(x), return_inverse=True)
    return codes.reshape(-1), len(values)


def _mi_codes(xi: np.ndarray, kx: int, yi: np.ndarray, ky: int) -> float:
    joint = np.bincount(xi * ky + yi, minlength=kx * ky).reshape(kx, ky).astype(np.float64)
    joint /= max(joint.sum(), 1.0)
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    xi, kx = _codes(x)
    yi, ky = _codes(y)
    return _mi_codes(xi, kx, yi, ky)


def mutual_information_ci(x: np.ndarray, y: np.ndarray, samples: int = 200, seed: int = 0) -> tuple[float, float]:
    xi, kx = _codes(x)
    yi, ky = _codes(y)
    if len(xi) < 2 or samples <= 0:
        value = _mi_codes(xi, kx, yi, ky)
        return (value, value)
    rng = np.random.default_rng(seed)
    values = [_mi_codes(xi[idx], kx, yi[idx], ky) for idx in rng.integers(0, len(xi), size=(samples, len(xi)))]
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def mutual_information_null(x: np.ndarray, y: np.ndarray, samples: int = 200, seed: int = 0, level: float = 0.95) -> float:
    xi, kx = _codes(x)
    yi, ky = _codes(y)
    if samples <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    values = [_mi_codes(rng.permutation(xi), kx, yi, ky) for _ in range(samples)]
    return float(np.quantile(values, level))


def guess_accuracy(tokens: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xi, kx = _codes(tokens)
    yi, ky = _codes(y)
    joint = np.bincount(xi * ky + yi, minlength=kx * ky).reshape(kx, ky).astype(np.float64)
    total = max(joint.sum(), 1.0)
    return float(joint.sum(0).max() / total), float(joint.max(1).sum() / total)


def entropy_bits(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def summarize(data: dict, vocab: int, bootstrap: int = 200, seed: int = 0) -> dict:
    tokens, feats, tops = data["tokens"], data["features"], data["top"]
    n = len(tokens)
    freq = np.bincount(tokens, minlength=vocab) / max(n, 1)
    base = feats.mean(0)
    conditional = np.zeros((vocab, feats.shape[1]))
    top_given = np.zeros((vocab, len(TOP_CARD_NAMES)))
    for t in range(vocab):
        sel = tokens == t
        if sel.any():
            conditional[t] = feats[sel].mean(0)
            top_given[t] = np.bincount(tops[sel], minlength=len(TOP_CARD_NAMES)) / sel.sum()
    dictionary = []
    for t in range(vocab):
        if freq[t] == 0:
            continue
        lift = conditional[t] / np.maximum(base, 1e-9)
        order = np.argsort(-lift)
        dictionary.append(
            {
                "seña": t,
                "frecuencia": round(float(freq[t]), 4),
                "carta_mas_alta_tipica": TOP_CARD_NAMES[int(np.argmax(top_given[t]))],
                "p_carta_mas_alta": {TOP_CARD_NAMES[i]: round(float(top_given[t, i]), 3) for i in range(len(TOP_CARD_NAMES))},
                "rasgos_que_anuncia": [
                    {"rasgo": FEATURE_NAMES[i], "p_dada_seña": round(float(conditional[t, i]), 3), "lift": round(float(lift[i]), 2)}
                    for i in order[:3]
                    if lift[i] > 1.2
                ],
            }
        )
    def ci(y: np.ndarray, offset: int) -> list[float]:
        lo, hi = mutual_information_ci(tokens, y, samples=bootstrap, seed=seed + offset)
        return [round(lo, 4), round(hi, 4)]

    def null(y: np.ndarray, offset: int) -> float:
        return round(mutual_information_null(tokens, y, samples=bootstrap, seed=seed + 100 + offset), 4)

    return {
        "señas_observadas": int(n),
        "entropia_señas_bits": round(entropy_bits(tokens), 4),
        "mi_carta_mas_alta_bits": round(mutual_information(tokens, tops), 4),
        "mi_carta_mas_alta_ci95": ci(tops, 0),
        "mi_carta_mas_alta_nula_p95": null(tops, 0),
        "entropia_carta_mas_alta_bits": round(entropy_bits(tops), 4),
        "acierto_carta_mas_alta": dict(zip(("sin_seña", "con_seña"), (round(v, 4) for v in guess_accuracy(tokens, tops)))),
        "mi_por_rasgo_bits": {FEATURE_NAMES[i]: round(mutual_information(tokens, feats[:, i]), 4) for i in range(feats.shape[1])},
        "mi_por_rasgo_ci95": {FEATURE_NAMES[i]: ci(feats[:, i], i + 1) for i in range(feats.shape[1])},
        "mi_por_rasgo_nula_p95": {FEATURE_NAMES[i]: null(feats[:, i], i + 1) for i in range(feats.shape[1])},
        "entropia_por_rasgo_bits": {FEATURE_NAMES[i]: round(entropy_bits(feats[:, i]), 4) for i in range(feats.shape[1])},
        "bootstrap": bootstrap,
        "diccionario": dictionary,
        "p_rasgo_dada_seña": conditional.round(4).tolist(),
        "p_rasgo_base": base.round(4).tolist(),
    }


def plot_dictionary(summary: dict, path: str, title: str | None = None) -> None:
    from .plots import INK, diverging_cmap, pyplot

    plt = pyplot()
    cond = np.array(summary["p_rasgo_dada_seña"])
    base = np.array(summary["p_rasgo_base"])
    lift = cond / np.maximum(base, 1e-9)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(np.log2(np.maximum(lift, 1e-3)), cmap=diverging_cmap(), vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(cond.shape[0]), [f"seña {i}" for i in range(cond.shape[0])])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(cond.shape[0]):
        for j in range(cond.shape[1]):
            strong = abs(np.log2(max(lift[i, j], 1e-3))) > 1.8
            ax.text(j, i, f"{cond[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if strong else INK)
    bar = fig.colorbar(im, ax=ax, label="log2 lift sobre la base")
    bar.outline.set_visible(False)
    ax.set_title(title or f"Qué anuncia cada seña · MI con la carta más alta: {summary['mi_carta_mas_alta_bits']:.2f} bits", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
