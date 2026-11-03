from __future__ import annotations

import numpy as np

from .cards import STRENGTH, Card, envido_points

FEATURE_NAMES = (
    "ancho_espada",
    "ancho_basto",
    "siete_espada",
    "siete_oro",
    "algun_tres",
    "envido_27",
    "envido_30",
    "dos_bravas",
)

_ANCHO_ESPADA = Card(1, "espada").index
_ANCHO_BASTO = Card(1, "basto").index
_SIETE_ESPADA = Card(7, "espada").index
_SIETE_ORO = Card(7, "oro").index


def hand_features(cards: list[int]) -> np.ndarray:
    env = envido_points(cards)
    bravas = sum(STRENGTH[c] >= 11 for c in cards)
    return np.array(
        [
            _ANCHO_ESPADA in cards,
            _ANCHO_BASTO in cards,
            _SIETE_ESPADA in cards,
            _SIETE_ORO in cards,
            any(STRENGTH[c] == 10 for c in cards),
            env >= 27,
            env >= 30,
            bravas >= 2,
        ],
        dtype=np.float32,
    )


N_FEATURES = len(FEATURE_NAMES)
