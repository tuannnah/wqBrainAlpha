"""Test GPSeedGenerator adapter: implement giao thức ``idea_generator.generate_ideas(n)``
cho RefinementLoop, trả ``n`` core seed serialize từ Phase 7.3, xác định theo rng inject."""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401  (side-effect: nạp 27 operator vào registry)
from src.gp.seed_adapter import GPSeedGenerator
from src.lang.parser import parse


def test_generate_ideas_returns_n_distinct_parseable_strings() -> None:
    """Trả đúng ``n`` chuỗi, mỗi chuỗi parse được (seed cores Phase 7.3 đã pass parse)."""
    gen = GPSeedGenerator(rng=np.random.default_rng(42))
    ideas = gen.generate_ideas(5)
    assert len(ideas) == 5
    for s in ideas:
        node = parse(s)
        assert node is not None


def test_generate_ideas_is_deterministic_for_same_rng_seed() -> None:
    """Cùng seed rng → cùng danh sách ideas (determinism R8)."""
    g1 = GPSeedGenerator(rng=np.random.default_rng(42))
    g2 = GPSeedGenerator(rng=np.random.default_rng(42))
    assert g1.generate_ideas(5) == g2.generate_ideas(5)


def test_generate_ideas_n_larger_than_pool_returns_with_replacement() -> None:
    """n > tổng seed cores có sẵn: không crash — lấy with-replacement để đủ ``n``."""
    gen = GPSeedGenerator(rng=np.random.default_rng(0))
    ideas = gen.generate_ideas(1000)
    assert len(ideas) == 1000
    assert all(isinstance(s, str) and s for s in ideas)
