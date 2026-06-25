"""Test variation.py: crossover/mutation tạo cây hợp lệ (hoặc giữ nguyên khi không sửa
được an toàn), depth cap được giữ, dedup theo canonical hash."""

from __future__ import annotations

import numpy as np

from src.gp.individual import Individual
from src.gp.variation import (
    crossover,
    dedup_population,
    hoist_mutation,
    point_mutation,
    subtree_mutation,
)
from src.lang.ast import Call, Constant, Field, Node
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, Serializer

_FIELDS = ("close", "volume", "returns")


def _tree_a() -> Node:
    return Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(5.0))),))


def _tree_b() -> Node:
    return Call(op="rank", args=(Call(op="ts_mean", args=(Field("volume"), Constant(10.0))),))


def test_crossover_respects_max_depth_on_both_children():
    rng = np.random.default_rng(0)
    a, b = crossover(_tree_a(), _tree_b(), rng, max_depth=7)
    assert DepthVisitor().visit(a) <= 7
    assert DepthVisitor().visit(b) <= 7


def test_crossover_is_deterministic_for_same_seed():
    a1, b1 = crossover(_tree_a(), _tree_b(), np.random.default_rng(5), max_depth=7)
    a2, b2 = crossover(_tree_a(), _tree_b(), np.random.default_rng(5), max_depth=7)
    assert Serializer().visit(a1) == Serializer().visit(a2)
    assert Serializer().visit(b1) == Serializer().visit(b2)


def test_point_mutation_changes_something_or_no_op_safely():
    rng = np.random.default_rng(1)
    registry = default_registry()
    mutated = point_mutation(_tree_a(), registry, rng, fields=_FIELDS)
    assert isinstance(mutated, Node)
    assert DepthVisitor().visit(mutated) >= 1


def test_point_mutation_does_not_mutate_input_in_place():
    original = _tree_a()
    serialized_before = Serializer().visit(original)
    point_mutation(original, default_registry(), np.random.default_rng(2), fields=_FIELDS)
    assert Serializer().visit(original) == serialized_before


def test_subtree_mutation_respects_max_depth():
    rng = np.random.default_rng(3)
    registry = default_registry()
    mutated = subtree_mutation(_tree_a(), registry, rng, fields=_FIELDS, max_depth=5)
    assert DepthVisitor().visit(mutated) <= 5


def test_hoist_mutation_shrinks_or_keeps_tree_depth():
    rng = np.random.default_rng(4)
    original = _tree_a()
    hoisted = hoist_mutation(original, rng)
    assert DepthVisitor().visit(hoisted) <= DepthVisitor().visit(original)


def test_hoist_mutation_on_single_leaf_returns_same_leaf():
    leaf = Field("close")
    hoisted = hoist_mutation(leaf, np.random.default_rng(6))
    assert hoisted == leaf


def test_dedup_population_removes_structural_duplicates_keeps_first():
    ind1 = Individual(expr=_tree_a())
    ind2 = Individual(expr=_tree_a())  # cùng cấu trúc, instance khác
    ind3 = Individual(expr=_tree_b())
    result = dedup_population([ind1, ind2, ind3])
    assert len(result) == 2
    assert result[0] is ind1
    assert result[1] is ind3
