import random

import numpy as np
import pytest

from agents.arena import head_to_head
from agents.base import RandomAgent
from agents.cfr import (
    BYTES_PER_INFOSET_BOUND,
    DEFAULT_ABSTRACTION,
    MCCFR,
    SHARED_BYTES_PER_INFOSET,
    Abstraction,
    CFRAgent,
    InfosetTable,
    Policy,
    SharedTable,
    TablePart,
    infoset_key,
    infosets_for_budget,
    merge_parts,
    rss_bytes,
    table_capacity,
    train_parallel,
    worker_seeds,
)
from truco import actions as A
from truco.engine import TrucoHand
from truco.rules import Rules

HU = Rules(num_players=2, signals=False)


def _signature(legal):
    return tuple("p" if a in A.PLAY_ACTIONS else a for a in legal)


def _random_states(count, seed=0, rules=HU):
    rng = random.Random(seed)
    for _ in range(count):
        hand = TrucoHand(rules, rng=random.Random(rng.random()), scores=(rng.randrange(30), rng.randrange(30)), mano=rng.randrange(2))
        while not hand.done:
            yield hand
            hand.step(rng.choice(hand.legal_actions()))


def test_key_fixes_the_legal_actions():
    seen = {}
    for hand in _random_states(3000):
        seat = hand.current_player
        key = infoset_key(hand, seat)
        assert 0 <= key < 1 << 63
        signature = _signature(hand.legal_actions())
        assert seen.setdefault(key, signature) == signature


def test_key_buckets_scores():
    hands = [[0, 1, 2], [10, 11, 12]]

    def key(scores, abstraction=DEFAULT_ABSTRACTION):
        return infoset_key(TrucoHand(HU, rng=random.Random(0), scores=scores, hands=hands), 0, abstraction)

    assert Abstraction().need_edges == (16,)
    assert key((0, 0)) == key((5, 10))
    assert key((0, 0)) != key((20, 0))
    assert key((20, 0)) == key((26, 3))
    assert key((14, 0)) != key((15, 0))
    assert Abstraction().need_bucket(30) == 1
    assert Abstraction().need_bucket(2) == 0
    finer = Abstraction(need_edges=(4, 8, 16))
    assert key((20, 0), finer) != key((26, 0), finer)
    assert finer.need_bucket(30) == 3


def test_key_compacts_the_betting_history():
    hands = [[0, 1, 2], [10, 11, 12]]
    a = TrucoHand(HU, rng=random.Random(0), hands=hands, scores=(3, 4))
    b = TrucoHand(HU, rng=random.Random(0), hands=hands, scores=(7, 1))
    a.step(A.ENVIDO)
    b.step(A.REAL_ENVIDO)
    for hand in (a, b):
        hand.step(A.NO_QUIERO)
        assert hand.points == [1, 0]
    assert infoset_key(a, 0) == infoset_key(b, 0)
    c = TrucoHand(HU, rng=random.Random(0), hands=hands, scores=(3, 4))
    c.step(A.ENVIDO)
    c.step(A.QUIERO)
    assert infoset_key(c, 0) != infoset_key(a, 0)


def test_abstraction_roundtrip_and_space():
    abstraction = Abstraction(card_edges=(11, 9, 5), need_edges=(4, 16))
    again = Abstraction.from_dict(abstraction.to_dict())
    assert again == abstraction
    assert abstraction.key_space < 1 << 63
    with pytest.raises(ValueError):
        Abstraction(revealed_edges=(20, 22, 24, 26, 28))


def _fill(table, keys):
    offsets = [table.find_or_add(k, 2 + k % 5) for k in keys]
    expected = [0]
    for k in keys[:-1]:
        expected.append(expected[-1] + 2 + k % 5)
    return offsets, expected


def test_table_grows_and_finds_offsets():
    table = InfosetTable(bits=4)
    keys = random.Random(1).sample(range(1 << 62), 5000)
    offsets, expected = _fill(table, keys)
    assert offsets == expected
    assert all(table.find(k) == o for k, o in zip(keys, offsets))
    assert all(table.find_or_add(k, 2 + k % 5) == o for k, o in zip(keys, offsets))
    assert table.find(12345) == -1
    assert len(table) == 5000
    assert len(table.regret) == sum(2 + k % 5 for k in keys)
    assert table.nbytes() < 5000 * BYTES_PER_INFOSET_BOUND
    part = table.export()
    assert part.keys.tolist() == sorted(keys)
    assert np.all(np.diff(part.keys) > 0)


def test_shared_table_is_shared_and_bounded():
    owner = SharedTable(300)
    try:
        other = SharedTable.attach(owner.spec())
        keys = random.Random(2).sample(range(1 << 62), 400)
        offsets, expected = _fill(owner, keys[:200])
        assert offsets == expected
        assert all(other.find(k) == o for k, o in zip(keys[:200], offsets))
        more = [other.find_or_add(k, 3) for k in keys[200:]]
        assert len(owner) == 300 and other.count == 300
        assert -1 not in more[:100] and more[100:] == [-1] * 100
        assert owner.dropped == 100
        other.regret[more[0]] = 2.5
        assert owner.regret[more[0]] == 2.5
        assert owner.add_iterations(7) == 7 and other.add_iterations(3) == 10
        assert owner.bytes_per_infoset() < SHARED_BYTES_PER_INFOSET
        part = owner.export()
        assert len(part.keys) == 300
        assert part.strategy.shape[0] == int(owner.counters[1])
        other.close()
    finally:
        owner.close()


def test_table_respects_the_cap():
    solver = MCCFR(HU, seed=0, max_infosets=300)
    for _ in range(30):
        solver.iterate()
    assert len(solver.table) == 300
    assert solver.table.dropped > 0
    assert solver.stats()["infosets"] == 300


def test_solver_is_deterministic():
    def run(seed):
        solver = MCCFR(HU, seed=seed)
        for _ in range(25):
            solver.iterate()
        return solver.export()
    a, b, c = run(3), run(3), run(4)
    for name in ("keys", "sizes", "offsets", "strategy"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert not np.array_equal(a.keys, c.keys) or not np.array_equal(a.strategy, c.strategy)


def test_merge_sums_overlapping_parts():
    first = TablePart(
        keys=np.array([5, 9], dtype=np.int64),
        sizes=np.array([2, 3], dtype=np.uint8),
        offsets=np.array([0, 2], dtype=np.int64),
        strategy=np.array([1.0, 3.0, 1.0, 1.0, 2.0], dtype=np.float32),
    )
    second = TablePart(
        keys=np.array([1, 5, 7], dtype=np.int64),
        sizes=np.array([2, 2, 2], dtype=np.uint8),
        offsets=np.array([4, 0, 2], dtype=np.int64),
        strategy=np.array([3.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    keys, sizes, probs = merge_parts([first, second])
    assert keys.tolist() == [1, 5, 9]
    assert sizes.tolist() == [2, 2, 3]
    assert np.allclose(probs, [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.5])


def test_chunked_merge_matches_single_pass():
    parts = []
    for seed in (0, 1, 2):
        solver = MCCFR(HU, seed=seed)
        for _ in range(15):
            solver.iterate()
        parts.append(solver.export())
    whole = merge_parts(parts)
    chunked = merge_parts(parts, rows_per_chunk=500)
    for x, y in zip(whole, chunked):
        assert np.array_equal(x, y)


def test_merge_rejects_inconsistent_sizes():
    a = TablePart(np.array([3], dtype=np.int64), np.array([2], dtype=np.uint8), np.array([0], dtype=np.int64), np.ones(2, dtype=np.float32))
    b = TablePart(np.array([3], dtype=np.int64), np.array([3], dtype=np.uint8), np.array([0], dtype=np.int64), np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        merge_parts([a, b])


def test_policy_roundtrip_and_agent(tmp_path):
    solver = MCCFR(HU, seed=0)
    for _ in range(40):
        solver.iterate()
    policy = solver.policy()
    assert len(policy) > 0
    sums = np.add.reduceat(policy.probs.astype(np.float64), policy.offsets)
    assert np.allclose(sums, 1.0, atol=1e-5)
    path = tmp_path / "cfr.npz"
    policy.save(path)
    loaded = Policy.load(path)
    assert loaded.rules == HU
    assert loaded.abstraction == policy.abstraction
    assert np.array_equal(loaded.keys, policy.keys)
    assert np.array_equal(loaded.probs, policy.probs)
    agent = CFRAgent.from_file(path, seed=0)
    stats = head_to_head(HU, [agent], [RandomAgent(seed=1)], 6, seed=0)
    assert stats.matches == 6
    assert agent.hits > 0


def test_cfr_agent_rejects_pairs():
    policy = MCCFR(HU, seed=0).policy()
    policy.rules = Rules(num_players=4)
    with pytest.raises(ValueError):
        CFRAgent(policy)


def test_parallel_training_shares_one_bounded_table():
    policy, curve = train_parallel(HU, 40, 2, seed=7, report_every=10, max_infosets=400, verbose=False)
    assert {row["worker"] for row in curve} == {0, 1}
    assert all(row["infosets"] <= 400 for row in curve)
    assert max(row["iteration"] for row in curve) == 20
    assert max(row["total_iterations"] for row in curve) == 40
    assert policy.meta["table"] == "compartida"
    assert policy.meta["max_infosets"] == 400
    assert policy.meta["infosets_table"] == 400 and policy.meta["dropped"] > 0
    assert 0 < len(policy) <= 400


def test_single_worker_training_is_deterministic():
    runs = [train_parallel(HU, 15, 1, seed=3, report_every=5, max_infosets=5000, verbose=False) for _ in range(2)]
    (a, curve_a), (b, curve_b) = runs
    assert a.meta["table"] == "dinamica"
    assert np.array_equal(a.keys, b.keys) and np.array_equal(a.probs, b.probs)
    assert [r["infosets"] for r in curve_a] == [r["infosets"] for r in curve_b]


def test_budget_helpers():
    assert infosets_for_budget(1.0, bytes_per_infoset=128) == (1 << 30) // 128
    cap = table_capacity(1.0, None, 4)
    assert SharedTable.bytes_for(cap) <= 1 << 30 < SharedTable.bytes_for(int(cap * 1.1))
    assert table_capacity(6.0, 8_000_000, 5) == 8_000_000
    assert SharedTable.bytes_for(8_000_000) < 600 * 2**20
    assert table_capacity(1.0, 10, 4) == 10
    assert table_capacity(0.001, 10**9, 1) == int(0.001 * (1 << 30)) // BYTES_PER_INFOSET_BOUND
    seeds = worker_seeds(0, 3)
    assert len(set(seeds)) == 3
    assert seeds == worker_seeds(0, 3)
    assert set(seeds).isdisjoint(worker_seeds(1, 3))
    assert rss_bytes() > 0
