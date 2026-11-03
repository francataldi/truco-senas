from __future__ import annotations

from dataclasses import dataclass

SUITS = ("espada", "basto", "oro", "copa")
RANKS = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
N_CARDS = len(SUITS) * len(RANKS)


@dataclass(frozen=True)
class Card:
    rank: int
    suit: str

    @property
    def index(self) -> int:
        return SUITS.index(self.suit) * len(RANKS) + RANKS.index(self.rank)

    @property
    def strength(self) -> int:
        return truco_strength(self.rank, self.suit)

    @property
    def envido_value(self) -> int:
        return self.rank if self.rank <= 7 else 0

    def __str__(self) -> str:
        return f"{self.rank} de {self.suit}"


def truco_strength(rank: int, suit: str) -> int:
    if rank == 1 and suit == "espada":
        return 14
    if rank == 1 and suit == "basto":
        return 13
    if rank == 7 and suit == "espada":
        return 12
    if rank == 7 and suit == "oro":
        return 11
    if rank == 3:
        return 10
    if rank == 2:
        return 9
    if rank == 1:
        return 8
    if rank == 12:
        return 7
    if rank == 11:
        return 6
    if rank == 10:
        return 5
    if rank == 7:
        return 4
    if rank == 6:
        return 3
    if rank == 5:
        return 2
    return 1


DECK = tuple(Card(rank, suit) for suit in SUITS for rank in RANKS)
STRENGTH = tuple(card.strength for card in DECK)
ENVIDO_VALUE = tuple(card.envido_value for card in DECK)
SUIT_OF = tuple(SUITS.index(card.suit) for card in DECK)
MAX_STRENGTH = 14
MAX_ENVIDO = 33


def card_from_index(index: int) -> Card:
    return DECK[index]


def envido_points(cards: list[int] | tuple[int, ...]) -> int:
    best = 0
    for i in range(len(cards)):
        best = max(best, ENVIDO_VALUE[cards[i]])
        for j in range(i + 1, len(cards)):
            if SUIT_OF[cards[i]] == SUIT_OF[cards[j]]:
                best = max(best, 20 + ENVIDO_VALUE[cards[i]] + ENVIDO_VALUE[cards[j]])
    return best


def has_flor(cards: list[int] | tuple[int, ...]) -> bool:
    return len(cards) == 3 and SUIT_OF[cards[0]] == SUIT_OF[cards[1]] == SUIT_OF[cards[2]]


def flor_points(cards: list[int] | tuple[int, ...]) -> int:
    return 20 + sum(ENVIDO_VALUE[c] for c in cards)


def sort_by_strength(cards: list[int]) -> list[int]:
    return sorted(cards, key=lambda c: (-STRENGTH[c], SUIT_OF[c], c))
