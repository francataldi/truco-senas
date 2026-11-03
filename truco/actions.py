from __future__ import annotations

PLAY_0 = 0
PLAY_1 = 1
PLAY_2 = 2
TRUCO = 3
ENVIDO = 4
REAL_ENVIDO = 5
FALTA_ENVIDO = 6
QUIERO = 7
NO_QUIERO = 8
MAZO = 9
FLOR = 10
SIGNAL_BASE = 11

PLAY_ACTIONS = (PLAY_0, PLAY_1, PLAY_2)
ENVIDO_ACTIONS = (ENVIDO, REAL_ENVIDO, FALTA_ENVIDO)

_NAMES = {
    PLAY_0: "jugar carta 1",
    PLAY_1: "jugar carta 2",
    PLAY_2: "jugar carta 3",
    TRUCO: "truco",
    ENVIDO: "envido",
    REAL_ENVIDO: "real envido",
    FALTA_ENVIDO: "falta envido",
    QUIERO: "quiero",
    NO_QUIERO: "no quiero",
    MAZO: "irse al mazo",
    FLOR: "flor",
}

TRUCO_NAMES = ("truco", "retruco", "vale cuatro")


def num_actions(signal_vocab: int) -> int:
    return SIGNAL_BASE + signal_vocab


def signal_action(token: int) -> int:
    return SIGNAL_BASE + token


def is_signal(action: int) -> bool:
    return action >= SIGNAL_BASE


def signal_token(action: int) -> int:
    return action - SIGNAL_BASE


def action_name(action: int, truco_level: int = 0) -> str:
    if is_signal(action):
        return f"seña {signal_token(action)}"
    if action == TRUCO:
        return TRUCO_NAMES[min(truco_level, 2)]
    return _NAMES[action]
