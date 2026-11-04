from __future__ import annotations

import contextlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from truco import actions as A
from truco.cards import N_CARDS, STRENGTH
from truco.engine import PARDA, TrucoHand
from truco.rules import Rules

from .base import Agent
from .heuristic import HeuristicAgent

ABSTRACTION_VERSION = 2
EMPTY = -1
GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
BYTES_PER_INFOSET_BOUND = 128
SHARED_ENTRIES_PER_INFOSET = 4.0
SHARED_BYTES_PER_INFOSET = 64
DEFAULT_MAX_INFOSETS = 8_000_000
REGRET_MODES = ("linear", "plus", "vanilla")

_E, _R, _F = A.ENVIDO, A.REAL_ENVIDO, A.FALTA_ENVIDO
ENVIDO_SEQUENCES = (
    (_E,),
    (_E, _E),
    (_E, _E, _R),
    (_E, _E, _R, _F),
    (_E, _E, _F),
    (_E, _R),
    (_E, _R, _F),
    (_E, _F),
    (_R,),
    (_R, _F),
    (_F,),
)
_SEQUENCE_INDEX = {seq: i + 1 for i, seq in enumerate(ENVIDO_SEQUENCES)}
_ENVIDO_STATES = len(ENVIDO_SEQUENCES) + 9


def _bucket_desc(value: int, edges: tuple[int, ...]) -> int:
    for i, edge in enumerate(edges):
        if value >= edge:
            return i
    return len(edges)


def _bucket_asc(value: int, edges: tuple[int, ...]) -> int:
    return sum(value >= edge for edge in edges)


@dataclass(frozen=True)
class Abstraction:
    card_edges: tuple[int, ...] = (13, 11, 9, 5)
    rival_edges: tuple[int, ...] = (11, 9, 5)
    need_edges: tuple[int, ...] = (16,)
    envido_edges: tuple[int, ...] = (24, 27, 29, 31)
    revealed_edges: tuple[int, ...] = (27, 31)
    version: int = ABSTRACTION_VERSION
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("card_edges", "rival_edges", "need_edges", "envido_edges", "revealed_edges"):
            object.__setattr__(self, name, tuple(int(x) for x in getattr(self, name)))
        if 13 + len(self.revealed_edges) >= _ENVIDO_STATES - 3:
            raise ValueError("demasiados rangos de envido revelado")
        cards = len(self.card_edges) + 2
        rivals = len(self.rival_edges) + 2
        radices = (
            1 << (A.SIGNAL_BASE - A.TRUCO),
            2,
            len(self.need_edges) + 1,
            len(self.need_edges) + 1,
            7,
            cards**3,
            cards,
            rivals,
            8,
            rivals**2,
            len(self.envido_edges) + 2,
            _ENVIDO_STATES,
            6,
            7,
        )
        if math.prod(radices) >= 1 << 63:
            raise ValueError("la abstracción no entra en 63 bits")
        self._cache["card"] = tuple(_bucket_desc(STRENGTH[c], self.card_edges) for c in range(N_CARDS))
        self._cache["rival"] = tuple(_bucket_desc(STRENGTH[c], self.rival_edges) for c in range(N_CARDS))
        self._cache["envido"] = tuple(_bucket_asc(v, self.envido_edges) for v in range(34))
        self._cache["revealed"] = tuple(_bucket_asc(v, self.revealed_edges) for v in range(34))
        self._cache["need"] = tuple(_bucket_asc(v, self.need_edges) for v in range(128))
        self._cache["sizes"] = (len(self.need_edges) + 1, cards, rivals, cards**3, len(self.envido_edges) + 2)
        self._cache["radices"] = radices

    def to_dict(self) -> dict:
        return {
            "card_edges": list(self.card_edges),
            "rival_edges": list(self.rival_edges),
            "need_edges": list(self.need_edges),
            "envido_edges": list(self.envido_edges),
            "revealed_edges": list(self.revealed_edges),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Abstraction:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})

    @property
    def key_space(self) -> int:
        return math.prod(self._cache["radices"])

    def need_bucket(self, need: int) -> int:
        return _bucket_asc(need, self.need_edges)

    def key(self, hand: TrucoHand, seat: int) -> int:
        cache = self._cache
        card_bucket = cache["card"]
        rival_bucket = cache["rival"]
        need_bucket = cache["need"]
        n_need, n_cards, n_rivals, n_mine, n_env = cache["sizes"]
        me = seat & 1
        opp = 1 - seat
        mask = 0
        for a in hand.legal_actions():
            if a >= A.TRUCO:
                mask |= 1 << (a - A.TRUCO)
        target = hand.rules.target
        scores = hand.scores
        points = hand.points
        k = mask * 2 + (hand.mano == seat)
        k = k * n_need + need_bucket[min(127, max(0, target - scores[me] - points[me]))]
        k = k * n_need + need_bucket[min(127, max(0, target - scores[1 - me] - points[1 - me]))]
        rnd = hand.round
        winners = hand.round_winners
        if rnd == 0:
            rs = 0
        else:
            w = winners[0]
            rs = (3 if w == PARDA else (1 if w == me else 2)) if rnd == 1 else (6 if w == PARDA else (4 if w == me else 5))
        k = k * 7 + rs
        cards = hand.hands[seat]
        played = hand.played_slot[seat]
        mine = 0
        strengths = []
        for i in range(3):
            if not played[i]:
                c = cards[i]
                mine = mine * n_cards + card_bucket[c] + 1
                strengths.append(STRENGTH[c])
        k = k * n_mine + mine
        row = hand.table[rnd] if rnd < 3 else (None, None)
        my_card = row[seat]
        opp_card = row[opp]
        my_t = 0 if my_card is None else card_bucket[my_card] + 1
        opp_t = 0
        rel = 0
        if opp_card is not None and my_card is None:
            opp_t = rival_bucket[opp_card] + 1
            s = STRENGTH[opp_card]
            beats = 0
            tie = 0
            for x in strengths:
                if x > s:
                    beats += 1
                elif x == s:
                    tie = 1
            rel = beats * 2 + tie
        k = (k * n_cards + my_t) * n_rivals + opp_t
        k = k * 8 + rel
        prev = 0
        if rnd >= 1:
            a = rival_bucket[hand.table[0][opp]] + 1
            if rnd >= 2:
                b = rival_bucket[hand.table[1][opp]] + 1
                if a > b:
                    a, b = b, a
                prev = a * n_rivals + b
            else:
                prev = a
        k = k * n_rivals * n_rivals + prev
        if not hand.envido_done and rnd == 0 and hand.truco_level == 0:
            env = cache["envido"][hand.envido_of[seat]] + 1
        else:
            env = 0
        k = k * n_env + env
        pending = hand.pending
        if pending is not None and pending.kind == "envido":
            es = _SEQUENCE_INDEX[tuple(hand.envido_calls)]
        elif hand.flor_winner is not None:
            es = _ENVIDO_STATES - 1
        elif hand.envido_done and hand.envido_winner is not None:
            revealed = hand.revealed_envido
            if revealed[seat] is not None:
                es = 12
            elif revealed[opp] is not None:
                es = 13 + cache["revealed"][revealed[opp]]
            elif hand.envido_winner == me:
                es = _ENVIDO_STATES - 2
            else:
                es = _ENVIDO_STATES - 3
        else:
            es = 0
        k = k * _ENVIDO_STATES + es
        if pending is None:
            pe = 0
        elif pending.kind == "truco":
            pe = pending.level
        else:
            pe = 5 if hand.suspended is not None else 4
        level = hand.truco_level
        tr = 0 if level == 0 else 1 + (level - 1) * 2 + (hand.truco_holder != me)
        return (k * 6 + pe) * 7 + tr


DEFAULT_ABSTRACTION = Abstraction()


def infoset_key(hand: TrucoHand, seat: int, abstraction: Abstraction = DEFAULT_ABSTRACTION) -> int:
    return abstraction.key(hand, seat)


def regret_matching(regrets) -> np.ndarray:
    positive = np.maximum(np.asarray(regrets, dtype=np.float64), 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    return np.full(len(positive), 1.0 / len(positive))


def _part_from_slots(slot_keys, slot_meta, strategy, used_entries: int) -> TablePart:
    keys_view = np.frombuffer(slot_keys, dtype=np.int64)
    meta_view = np.frombuffer(slot_meta, dtype=np.int64)
    used = np.flatnonzero(keys_view)
    keys = keys_view[used] - 1
    meta = meta_view[used]
    del keys_view, meta_view
    order = np.argsort(keys, kind="stable")
    keys, meta = keys[order], meta[order]
    values = np.frombuffer(strategy, dtype=np.float32)[:used_entries].copy()
    return TablePart(keys=keys, sizes=(meta & 0xFF).astype(np.uint8), offsets=meta >> 8, strategy=values)


class InfosetTable:
    def __init__(self, max_infosets: int | None = None, bits: int = 12, load: float = 0.6) -> None:
        self.max_infosets = max_infosets
        self.load = load
        self.regret = array("f")
        self.strategy = array("f")
        self.count = 0
        self.dropped = 0
        self._zeros = [array("f", [0.0]) * n for n in range(32)]
        self._init_slots(bits)

    def _init_slots(self, bits: int) -> None:
        self.bits = bits
        self.shift = 64 - bits
        self.mask = (1 << bits) - 1
        self.slot_keys = array("q", [0]) * (1 << bits)
        self.slot_meta = array("q", [0]) * (1 << bits)
        self.limit = int(self.load * (1 << bits))

    def __len__(self) -> int:
        return self.count

    def find(self, key: int) -> int:
        stored = key + 1
        keys = self.slot_keys
        mask = self.mask
        slot = ((key * GOLDEN) & MASK64) >> self.shift
        while True:
            k = keys[slot]
            if k == stored:
                return self.slot_meta[slot] >> 8
            if k == 0:
                return -1
            slot = (slot + 1) & mask

    def find_or_add(self, key: int, size: int) -> int:
        stored = key + 1
        keys = self.slot_keys
        mask = self.mask
        slot = ((key * GOLDEN) & MASK64) >> self.shift
        while True:
            k = keys[slot]
            if k == stored:
                return self.slot_meta[slot] >> 8
            if k == 0:
                break
            slot = (slot + 1) & mask
        if self.max_infosets is not None and self.count >= self.max_infosets:
            self.dropped += 1
            return -1
        offset = len(self.regret)
        keys[slot] = stored
        self.slot_meta[slot] = (offset << 8) | size
        zeros = self._zeros[size]
        self.regret.extend(zeros)
        self.strategy.extend(zeros)
        self.count += 1
        if self.count > self.limit:
            self._grow()
        return offset

    def _grow(self) -> None:
        old_keys = self.slot_keys
        old_meta = self.slot_meta
        self._init_slots(self.bits + 1)
        keys = self.slot_keys
        metas = self.slot_meta
        mask = self.mask
        shift = self.shift
        for stored, meta in zip(old_keys, old_meta):
            if stored == 0:
                continue
            slot = (((stored - 1) * GOLDEN) & MASK64) >> shift
            while keys[slot] != 0:
                slot = (slot + 1) & mask
            keys[slot] = stored
            metas[slot] = meta

    def nbytes(self) -> int:
        return sum(sys.getsizeof(x) for x in (self.slot_keys, self.slot_meta, self.regret, self.strategy))

    def bytes_per_infoset(self) -> float:
        return self.nbytes() / max(1, self.count)

    def export(self) -> TablePart:
        return _part_from_slots(self.slot_keys, self.slot_meta, self.strategy, len(self.strategy))


class SharedTable:
    BLOCKS = ("keys", "meta", "regret", "strategy", "counters")

    def __init__(
        self,
        max_infosets: int,
        entries_per_infoset: float = SHARED_ENTRIES_PER_INFOSET,
        load: float = 0.5,
        names: dict | None = None,
        lock=None,
    ) -> None:
        from multiprocessing import shared_memory

        self.max_infosets = max(1, int(max_infosets))
        self.entries_per_infoset = float(entries_per_infoset)
        self.load = float(load)
        sizes = self.block_sizes(self.max_infosets, self.entries_per_infoset, self.load)
        slots = sizes["keys"] // 8
        self.bits = slots.bit_length() - 1
        self.shift = 64 - self.bits
        self.mask = slots - 1
        self.entries = sizes["regret"] // 4
        self.owner = names is None
        self._blocks = {}
        try:
            for name in self.BLOCKS:
                if self.owner:
                    self._blocks[name] = shared_memory.SharedMemory(create=True, size=sizes[name])
                else:
                    self._blocks[name] = shared_memory.SharedMemory(name=names[name])
        except BaseException:
            self._release_blocks()
            raise
        self.allocated = sum(sizes.values())
        self.slot_keys = self._blocks["keys"].buf.cast("q")
        self.slot_meta = self._blocks["meta"].buf.cast("q")
        self.regret = self._blocks["regret"].buf.cast("f")
        self.strategy = self._blocks["strategy"].buf.cast("f")
        self.counters = self._blocks["counters"].buf.cast("q")
        self.lock = lock if lock is not None else contextlib.nullcontext()

    @staticmethod
    def block_sizes(max_infosets: int, entries_per_infoset: float = SHARED_ENTRIES_PER_INFOSET, load: float = 0.5) -> dict:
        slots = 1 << max(4, (int(math.ceil(max(1, max_infosets) / load)) - 1).bit_length())
        entries = int(max(1, max_infosets) * entries_per_infoset) + 64
        return {"keys": 8 * slots, "meta": 8 * slots, "regret": 4 * entries, "strategy": 4 * entries, "counters": 64}

    @classmethod
    def bytes_for(cls, max_infosets: int, entries_per_infoset: float = SHARED_ENTRIES_PER_INFOSET, load: float = 0.5) -> int:
        return sum(cls.block_sizes(max_infosets, entries_per_infoset, load).values())

    def spec(self) -> dict:
        return {
            "max_infosets": self.max_infosets,
            "entries_per_infoset": self.entries_per_infoset,
            "load": self.load,
            "names": {name: block.name for name, block in self._blocks.items()},
        }

    @classmethod
    def attach(cls, spec: dict, lock=None) -> SharedTable:
        return cls(spec["max_infosets"], spec["entries_per_infoset"], spec["load"], names=spec["names"], lock=lock)

    def __len__(self) -> int:
        return int(self.counters[0])

    @property
    def count(self) -> int:
        return int(self.counters[0])

    @property
    def dropped(self) -> int:
        return int(self.counters[2])

    def add_iterations(self, n: int) -> int:
        with self.lock:
            self.counters[3] += n
            return int(self.counters[3])

    def find(self, key: int) -> int:
        stored = key + 1
        keys = self.slot_keys
        mask = self.mask
        slot = ((key * GOLDEN) & MASK64) >> self.shift
        while True:
            k = keys[slot]
            if k == stored:
                return self.slot_meta[slot] >> 8
            if k == 0:
                return -1
            slot = (slot + 1) & mask

    def find_or_add(self, key: int, size: int) -> int:
        stored = key + 1
        keys = self.slot_keys
        mask = self.mask
        slot = ((key * GOLDEN) & MASK64) >> self.shift
        while True:
            k = keys[slot]
            if k == stored:
                return self.slot_meta[slot] >> 8
            if k == 0:
                break
            slot = (slot + 1) & mask
        with self.lock:
            while True:
                k = keys[slot]
                if k == stored:
                    return self.slot_meta[slot] >> 8
                if k == 0:
                    break
                slot = (slot + 1) & mask
            counters = self.counters
            count = counters[0]
            used = counters[1]
            if count >= self.max_infosets or used + size > self.entries:
                counters[2] += 1
                return -1
            self.slot_meta[slot] = (used << 8) | size
            keys[slot] = stored
            counters[0] = count + 1
            counters[1] = used + size
            return used

    def nbytes(self) -> int:
        return self.allocated

    def bytes_per_infoset(self) -> float:
        count = max(1, self.count)
        return 16.0 / self.load + 8.0 * int(self.counters[1]) / count

    def export(self) -> TablePart:
        return _part_from_slots(self.slot_keys, self.slot_meta, self.strategy, int(self.counters[1]))

    def _release_blocks(self) -> None:
        for block in self._blocks.values():
            block.close()
            if self.owner:
                block.unlink()
        self._blocks = {}

    def close(self) -> None:
        for name in ("slot_keys", "slot_meta", "regret", "strategy", "counters"):
            view = getattr(self, name, None)
            if view is not None:
                view.release()
                setattr(self, name, None)
        self._release_blocks()


@dataclass
class TablePart:
    keys: np.ndarray
    sizes: np.ndarray
    offsets: np.ndarray
    strategy: np.ndarray

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("keys", "sizes", "offsets", "strategy"):
            np.save(folder / f"{name}.npy", getattr(self, name))

    @classmethod
    def load(cls, folder: str | Path, mmap: bool = True) -> TablePart:
        folder = Path(folder)
        mode = "r" if mmap else None
        return cls(**{name: np.load(folder / f"{name}.npy", mmap_mode=mode) for name in ("keys", "sizes", "offsets", "strategy")})

    def slice(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keys = np.asarray(self.keys[lo:hi])
        sizes = np.asarray(self.sizes[lo:hi]).astype(np.int64)
        if len(keys) == 0:
            return keys, sizes, np.zeros(0)
        starts = np.asarray(self.offsets[lo:hi])
        local = np.zeros(len(sizes), dtype=np.int64)
        np.cumsum(sizes[:-1], out=local[1:])
        index = np.repeat(starts - local, sizes) + np.arange(int(sizes.sum()))
        return keys, sizes, np.asarray(self.strategy[index], dtype=np.float64)


def _merge_segments(segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.concatenate([s[0] for s in segments])
    sizes = np.concatenate([s[1] for s in segments])
    sums = np.concatenate([s[2] for s in segments])
    if len(keys) == 0:
        return keys.astype(np.int64), sizes.astype(np.int64), sums
    uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    usizes = sizes[first]
    if not np.array_equal(usizes[inverse], sizes):
        raise ValueError("un mismo infoset aparece con distinta cantidad de acciones")
    uoff = np.zeros(len(uniq), dtype=np.int64)
    np.cumsum(usizes[:-1], out=uoff[1:])
    soff = np.zeros(len(keys), dtype=np.int64)
    np.cumsum(sizes[:-1], out=soff[1:])
    dst = np.repeat(uoff[inverse] - soff, sizes) + np.arange(len(sums))
    merged = np.bincount(dst, weights=sums, minlength=int(usizes.sum()))
    return uniq, usizes, merged


def _boundaries(parts: list[TablePart], rows_per_chunk: int) -> list[int]:
    total = sum(len(p.keys) for p in parts)
    chunks = max(1, total // max(1, rows_per_chunk))
    if chunks == 1:
        return []
    samples = np.unique(np.concatenate([np.asarray(p.keys[:: max(1, len(p.keys) // (64 * chunks))]) for p in parts]))
    picks = np.linspace(0, len(samples), chunks + 1)[1:-1].astype(np.int64)
    return sorted({int(samples[min(i, len(samples) - 1)]) for i in picks})


def merge_parts(parts: list[TablePart], rows_per_chunk: int = 2_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = _boundaries(parts, rows_per_chunk)
    out_keys, out_sizes, out_probs = [], [], []
    cuts = [[0, *[int(np.searchsorted(p.keys, e)) for e in edges], len(p.keys)] for p in parts]
    for c in range(len(edges) + 1):
        segments = [p.slice(cut[c], cut[c + 1]) for p, cut in zip(parts, cuts)]
        keys, sizes, sums = _merge_segments(segments)
        if len(keys) == 0:
            continue
        offsets = np.zeros(len(keys), dtype=np.int64)
        np.cumsum(sizes[:-1], out=offsets[1:])
        totals = np.add.reduceat(sums, offsets)
        keep = totals > 0
        probs = sums / np.repeat(np.where(keep, totals, 1.0), sizes)
        out_keys.append(keys[keep])
        out_sizes.append(sizes[keep].astype(np.uint8))
        out_probs.append(probs[np.repeat(keep, sizes)].astype(np.float32))
    if not out_keys:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float32)
    return np.concatenate(out_keys), np.concatenate(out_sizes), np.concatenate(out_probs)


@dataclass
class Policy:
    keys: np.ndarray
    sizes: np.ndarray
    probs: np.ndarray
    rules: Rules
    abstraction: Abstraction
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.offsets = np.zeros(len(self.keys), dtype=np.int64)
        if len(self.keys) > 1:
            np.cumsum(self.sizes[:-1].astype(np.int64), out=self.offsets[1:])

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.sizes.nbytes + self.probs.nbytes + self.offsets.nbytes)

    def lookup(self, key: int) -> np.ndarray | None:
        i = int(np.searchsorted(self.keys, key))
        if i < len(self.keys) and int(self.keys[i]) == key:
            start = int(self.offsets[i])
            return self.probs[start : start + int(self.sizes[i])]
        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"rules": self.rules.to_dict(), "abstraction": self.abstraction.to_dict(), **self.meta}
        with open(path, "wb") as f:
            np.savez_compressed(f, keys=self.keys, sizes=self.sizes, probs=self.probs, meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            rules = Rules(**meta.pop("rules"))
            abstraction = Abstraction.from_dict(meta.pop("abstraction"))
            return cls(data["keys"], data["sizes"], data["probs"], rules, abstraction, meta)


def build_policy(parts: list[TablePart], rules: Rules, abstraction: Abstraction, meta: dict | None = None) -> Policy:
    keys, sizes, probs = merge_parts(parts)
    return Policy(keys, sizes, probs, rules, abstraction, dict(meta or {}))


def _sample_scores(rng: random.Random, target: int, zero_prob: float) -> tuple[int, int]:
    if rng.random() < zero_prob:
        return (0, 0)
    return (rng.randrange(target), rng.randrange(target))


class MCCFR:
    def __init__(
        self,
        rules: Rules,
        seed: int = 0,
        regret_mode: str = "linear",
        abstraction: Abstraction | None = None,
        max_infosets: int | None = None,
        zero_score_prob: float = 0.3,
        table: InfosetTable | SharedTable | None = None,
    ) -> None:
        if rules.num_players != 2:
            raise ValueError("MCCFR baseline is for heads-up truco")
        if regret_mode not in REGRET_MODES:
            raise ValueError(f"regret_mode tiene que ser uno de {REGRET_MODES}")
        self.rules = rules
        self.abstraction = abstraction or DEFAULT_ABSTRACTION
        self.key = self.abstraction.key
        self.rng = random.Random(seed)
        self.regret_mode = regret_mode
        self.plus = regret_mode == "plus"
        self.regret_weight = 1.0
        self.zero_score_prob = zero_score_prob
        self.table = table if table is not None else InfosetTable(max_infosets)
        self.iterations = 0
        self.visits = 0

    def _sigma(self, off: int, n: int) -> list[float]:
        regret = self.table.regret
        sigma = [0.0] * n
        total = 0.0
        for i in range(n):
            r = regret[off + i]
            if r > 0.0:
                sigma[i] = r
                total += r
        if total > 0.0:
            return [x / total for x in sigma]
        return [1.0 / n] * n

    def traverse(self, hand: TrucoHand, traverser: int, weight: float) -> float:
        if hand.done:
            return float(hand.team_reward(traverser))
        self.visits += 1
        seat = hand.current_player
        legal = hand.legal_actions()
        n = len(legal)
        table = self.table
        off = table.find_or_add(self.key(hand, seat), n)
        if off < 0:
            sigma = [1.0 / n] * n
        else:
            sigma = self._sigma(off, n)
        if seat == traverser:
            utils = [0.0] * n
            value = 0.0
            last = n - 1
            for i in range(n):
                if i == last:
                    child = hand
                else:
                    child = hand.clone(fresh_rng=False)
                child.step(legal[i])
                u = self.traverse(child, traverser, weight)
                utils[i] = u
                value += sigma[i] * u
            if off >= 0:
                regret = table.regret
                if self.plus:
                    for i in range(n):
                        r = regret[off + i] + utils[i] - value
                        regret[off + i] = 0.0 if r < 0.0 else r
                else:
                    w = self.regret_weight
                    for i in range(n):
                        regret[off + i] += w * (utils[i] - value)
            return value
        if off >= 0:
            strategy = table.strategy
            for i in range(n):
                strategy[off + i] += weight * sigma[i]
        x = self.rng.random()
        choice = n - 1
        acc = 0.0
        for i in range(n):
            acc += sigma[i]
            if x < acc:
                choice = i
                break
        hand.step(legal[choice])
        return self.traverse(hand, traverser, weight)

    def iterate(self) -> None:
        self.iterations += 1
        weight = 1.0 if self.regret_mode == "vanilla" else float(self.iterations)
        self.regret_weight = weight if self.regret_mode == "linear" else 1.0
        rng = self.rng
        hand = TrucoHand(
            self.rules,
            rng=random.Random(rng.random()),
            scores=_sample_scores(rng, self.rules.target, self.zero_score_prob),
            mano=rng.randrange(2),
        )
        for traverser in (0, 1):
            self.traverse(hand.clone(fresh_rng=False), traverser, weight)

    def stats(self) -> dict:
        return {
            "infosets": len(self.table),
            "table_bytes": self.table.nbytes(),
            "bytes_per_infoset": round(self.table.bytes_per_infoset(), 2),
            "dropped": self.table.dropped,
            "visits": self.visits,
        }

    def export(self) -> TablePart:
        return self.table.export()

    def policy(self) -> Policy:
        return build_policy([self.export()], self.rules, self.abstraction, {"iterations": self.iterations})

    def average_strategy(self) -> Policy:
        return self.policy()


def rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        if kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return 0
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def infosets_for_budget(memory_gb: float, bytes_per_infoset: int = SHARED_BYTES_PER_INFOSET) -> int:
    return max(1, int(memory_gb * (1 << 30) / bytes_per_infoset))


def table_capacity(memory_gb: float, max_infosets: int | None, workers: int) -> int:
    if workers <= 1:
        cap = infosets_for_budget(memory_gb, BYTES_PER_INFOSET_BOUND)
        return cap if max_infosets is None else max(1, min(cap, int(max_infosets)))
    budget = memory_gb * (1 << 30)
    cap = infosets_for_budget(memory_gb, SHARED_BYTES_PER_INFOSET)
    if max_infosets is not None:
        cap = max(1, min(cap, int(max_infosets)))
    while cap > 1 and SharedTable.bytes_for(cap) > budget:
        cap = int(cap * 0.95)
    return cap


def worker_seeds(seed: int, workers: int) -> list[int]:
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(workers)]


_SHARED: dict = {}


def _attach_shared(spec: dict, lock) -> None:
    _SHARED["table"] = SharedTable.attach(spec, lock)


def _make_solver(task: dict, table=None) -> MCCFR:
    return MCCFR(
        Rules(**task["rules"]),
        seed=task["seed"],
        abstraction=Abstraction.from_dict(task["abstraction"]),
        max_infosets=task["max_infosets"],
        zero_score_prob=task["zero_score_prob"],
        regret_mode=task["regret_mode"],
        table=table,
    )


def _run_iterations(solver: MCCFR, task: dict) -> list[dict]:
    curve = []
    start = time.perf_counter()
    total = task["iterations"]
    every = max(1, task["report_every"])
    shared = isinstance(solver.table, SharedTable)
    done = 0
    for i in range(1, total + 1):
        solver.iterate()
        if i % every == 0 or i == total:
            elapsed = time.perf_counter() - start
            overall = solver.table.add_iterations(i - done) if shared else i
            done = i
            row = {
                "worker": task["worker"],
                "iteration": i,
                "total_iterations": overall,
                "seconds": round(elapsed, 2),
                **solver.stats(),
                "rss_bytes": rss_bytes(),
            }
            curve.append(row)
            if task["verbose"]:
                print(
                    f"[worker {task['worker']}] {i}/{total} iteraciones ({overall} en total), {i / max(elapsed, 1e-9):.0f} it/s, "
                    f"{row['infosets']} infosets, {row['bytes_per_infoset']:.0f} B/infoset, "
                    f"RSS {row['rss_bytes'] / 2**20:.0f} MB, descartados {row['dropped']}",
                    flush=True,
                )
    return curve


def _shared_worker(task: dict) -> dict:
    solver = _make_solver(task, table=_SHARED["table"])
    curve = _run_iterations(solver, task)
    return {"worker": task["worker"], "curve": curve, "visits": solver.visits}


def train_parallel(
    rules: Rules,
    iterations: int,
    workers: int,
    seed: int = 0,
    report_every: int = 1000,
    memory_gb: float = 6.0,
    max_infosets: int | None = DEFAULT_MAX_INFOSETS,
    abstraction: Abstraction | None = None,
    zero_score_prob: float = 0.3,
    regret_mode: str = "linear",
    verbose: bool = True,
) -> tuple[Policy, list[dict]]:
    abstraction = abstraction or DEFAULT_ABSTRACTION
    workers = max(1, workers)
    per_worker = max(1, iterations // workers)
    cap = table_capacity(memory_gb, max_infosets, workers)
    tasks = [
        {
            "rules": rules.to_dict(),
            "abstraction": abstraction.to_dict(),
            "iterations": per_worker,
            "seed": s,
            "worker": w,
            "report_every": report_every,
            "max_infosets": cap,
            "zero_score_prob": zero_score_prob,
            "regret_mode": regret_mode,
            "verbose": verbose and w == 0,
        }
        for w, s in enumerate(worker_seeds(seed, workers))
    ]
    meta = {
        "iterations": per_worker * workers,
        "workers": workers,
        "seed": seed,
        "max_infosets": cap,
        "zero_score_prob": zero_score_prob,
        "regret_mode": regret_mode,
        "table": "dinamica" if workers == 1 else "compartida",
    }
    if workers == 1:
        solver = _make_solver(tasks[0])
        curve = _run_iterations(solver, tasks[0])
        meta.update(infosets_table=len(solver.table), dropped=solver.table.dropped, table_bytes=solver.table.nbytes())
        return build_policy([solver.export()], rules, abstraction, meta), curve
    ctx = mp.get_context("spawn")
    lock = ctx.Lock()
    table = SharedTable(cap, lock=lock)
    try:
        with ctx.Pool(workers, initializer=_attach_shared, initargs=(table.spec(), lock)) as pool:
            results = pool.map(_shared_worker, tasks)
        meta.update(infosets_table=len(table), dropped=table.dropped, table_bytes=table.nbytes())
        part = table.export()
    finally:
        table.close()
    results.sort(key=lambda r: r["worker"])
    curve = [row for r in results for row in r["curve"]]
    return build_policy([part], rules, abstraction, meta), curve


def save_table(policy: Policy, path: str | Path) -> None:
    policy.save(path)


def load_table(path: str | Path) -> Policy:
    return Policy.load(path)


class CFRAgent(Agent):
    name = "cfr"

    def __init__(self, policy: Policy, seed: int | None = None, greedy: bool = False) -> None:
        if policy.rules.num_players != 2:
            raise ValueError("la política CFR es para el 1 contra 1")
        self.policy = policy
        self.key = policy.abstraction.key
        self.rng = random.Random(seed)
        self.greedy = greedy
        self.fallback = HeuristicAgent(seed=seed)
        self.hits = 0
        self.misses = 0

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> CFRAgent:
        return cls(Policy.load(path), **kwargs)

    @property
    def coverage(self) -> float:
        return self.hits / max(1, self.hits + self.misses)

    def act(self, hand: TrucoHand, seat: int) -> int:
        legal = hand.legal_actions()
        probs = self.policy.lookup(self.key(hand, seat))
        if probs is None or len(probs) != len(legal):
            self.misses += 1
            return self.fallback.act(hand, seat)
        self.hits += 1
        if self.greedy:
            return legal[int(np.argmax(probs))]
        x = self.rng.random() * float(probs.sum())
        acc = 0.0
        for i, p in enumerate(probs.tolist()):
            acc += p
            if x < acc:
                return legal[i]
        return legal[-1]
