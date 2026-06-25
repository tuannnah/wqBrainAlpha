"""Test Individual: bọc Node, lazy depth/complexity/hash qua visitor Phase 1,
fitness/generation mutable nhưng expr bất biến theo quy ước (không sửa tại chỗ)."""

from __future__ import annotations

from src.gp.individual import Individual
from src.lang.ast import Call, Constant, Field


def _alpha() -> Individual:
    expr = Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(5.0))),))
    return Individual(expr=expr)


def test_individual_starts_unevaluated():
    ind = _alpha()
    assert ind.is_evaluated() is False
    assert ind.fitness is None
    assert ind.generation == 0


def test_individual_depth_matches_visitor():
    ind = _alpha()
    assert ind.depth() == 3  # rank(ts_mean(close, 5)) -> rank>ts_mean>close = 3 tầng


def test_individual_complexity_counts_all_nodes():
    ind = _alpha()
    assert ind.complexity() == 4  # rank, ts_mean, close(field), 5(const)


def test_individual_canonical_hash_is_deterministic_and_matches_structurally_equal_tree():
    ind1 = _alpha()
    ind2 = _alpha()  # cây khác instance, cùng cấu trúc
    assert ind1.canonical_hash() == ind2.canonical_hash()


def test_individual_canonical_hash_differs_for_different_tree():
    ind1 = _alpha()
    other_expr = Call(op="rank", args=(Field("volume"),))
    ind2 = Individual(expr=other_expr)
    assert ind1.canonical_hash() != ind2.canonical_hash()


def test_setting_fitness_marks_evaluated_without_mutating_expr():
    ind = _alpha()
    original_expr = ind.expr
    ind.fitness = object()  # placeholder cho FitnessVector thật (Task 7.2) — chỉ test cờ is_evaluated
    assert ind.is_evaluated() is True
    assert ind.expr is original_expr  # expr không bị đổi khi set fitness
