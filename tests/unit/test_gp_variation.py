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
from src.lang.registry import ArgKind, OperatorRegistry, default_registry
from src.lang.visitors import DepthVisitor, Serializer

_FIELDS = ("close", "volume", "returns")


def _check_panel_invariant(node: Node, registry: OperatorRegistry) -> None:
    """Mỗi đối số ở slot ArgKind.PANEL của một Call phải là Call hoặc Field (tín hiệu),
    không bao giờ là Constant (literal số). Khoá regression cho lỗi typed GP Phase 7."""
    if not isinstance(node, Call):
        return
    spec = registry.get(node.op)
    for child, kind in zip(node.args, spec.signature):
        if kind is ArgKind.PANEL:
            assert not isinstance(child, Constant), (
                f"vi phạm bất biến PANEL: {node.op!r} có Constant ở slot PANEL"
            )
        _check_panel_invariant(child, registry)


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


# --- Type-aware Constant mutation: WINDOW phải int từ window_choices, SCALAR có thể float ---

def test_point_mutation_window_stays_int_from_choices():
    """Constant ở slot WINDOW (vd ts_mean(close, 5)) phải resample từ window_choices của
    chính operator cha — KHÔNG perturb Gaussian (float lẻ -> type-invalid theo WQ)."""
    registry = default_registry()
    spec = registry.get("ts_mean")
    allowed = set(spec.window_choices)
    tree = Call(op="ts_mean", args=(Field("close"), Constant(5.0)))
    rng = np.random.default_rng(123)
    # mutate nhiều lần, chỉ kiểm các lần điểm chọn rơi vào Constant
    for _ in range(60):
        mutated = point_mutation(tree, registry, rng, fields=_FIELDS)
        assert isinstance(mutated, Call)
        win_node = mutated.args[1]
        # nếu Constant đổi -> giá trị phải là int VÀ thuộc window_choices
        if isinstance(win_node, Constant) and win_node.value != 5.0:
            assert win_node.value.is_integer(), f"window phải là int, got {win_node.value!r}"
            assert int(win_node.value) in allowed, f"window {int(win_node.value)} ngoài choices {allowed}"


def test_point_mutation_scalar_can_perturb_gaussian():
    """Constant ở slot SCALAR -> perturb Gaussian (có thể ra float lẻ). Đăng ký 1
    operator tổng hợp tại chỗ có signature (PANEL, SCALAR) để kiểm — tránh phụ thuộc
    vào operator hiện hữu mà có thể có sig khác về sau."""
    from src.lang.registry import OpCategory, OperatorSpec
    registry = default_registry()
    registry.register(OperatorSpec(
        name="_test_panel_scalar", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.SCALAR),
        impl=lambda *_: None, bounded=False,
    ))
    try:
        tree = Call(op="_test_panel_scalar", args=(Field("close"), Constant(1.0)))
        saw_non_integer = False
        for seed in range(40):
            rng = np.random.default_rng(seed)
            mutated = point_mutation(tree, registry, rng, fields=_FIELDS)
            assert isinstance(mutated, Call)
            scalar_node = mutated.args[1]
            if isinstance(scalar_node, Constant) and not scalar_node.value.is_integer():
                saw_non_integer = True
                break
        assert saw_non_integer, "SCALAR perturb phải có thể ra giá trị non-integer ít nhất 1 lần"
    finally:
        # cleanup: xóa operator test để không rò ra test khác
        registry._ops.pop("_test_panel_scalar", None)


# --- "Cây con != cha mẹ" cho cả 3 variation operator ---

def test_point_mutation_produces_different_tree():
    """Có seed sao cho point_mutation thực sự thay đổi cây (không phải trùng giá trị)."""
    registry = default_registry()
    tree = _tree_a()
    serialized_orig = Serializer().visit(tree)
    saw_diff = False
    for seed in range(30):
        mutated = point_mutation(tree, registry, np.random.default_rng(seed), fields=_FIELDS)
        if Serializer().visit(mutated) != serialized_orig:
            saw_diff = True
            break
    assert saw_diff, "point_mutation phải sinh được cây khác cha mẹ với ít nhất 1 seed"


def test_subtree_mutation_produces_different_tree():
    registry = default_registry()
    tree = _tree_a()
    serialized_orig = Serializer().visit(tree)
    saw_diff = False
    for seed in range(30):
        mutated = subtree_mutation(tree, registry, np.random.default_rng(seed), fields=_FIELDS, max_depth=7)
        if Serializer().visit(mutated) != serialized_orig:
            saw_diff = True
            break
    assert saw_diff, "subtree_mutation phải sinh được cây khác cha mẹ với ít nhất 1 seed"


def test_crossover_produces_different_tree():
    """Với cha mẹ khác nhau, crossover phải sinh được ít nhất 1 con khác cả hai cha mẹ."""
    sa, sb = Serializer().visit(_tree_a()), Serializer().visit(_tree_b())
    saw_diff = False
    for seed in range(30):
        ca, cb = crossover(_tree_a(), _tree_b(), np.random.default_rng(seed), max_depth=7)
        sca, scb = Serializer().visit(ca), Serializer().visit(cb)
        if sca != sa or scb != sb:
            saw_diff = True
            break
    assert saw_diff, "crossover phải sinh được cây khác cha mẹ với ít nhất 1 seed"


# --- Determinism đối xứng cho point/subtree mutation ---

def test_point_mutation_is_deterministic_for_same_seed():
    registry = default_registry()
    m1 = point_mutation(_tree_a(), registry, np.random.default_rng(7), fields=_FIELDS)
    m2 = point_mutation(_tree_a(), registry, np.random.default_rng(7), fields=_FIELDS)
    assert Serializer().visit(m1) == Serializer().visit(m2)


def test_subtree_mutation_is_deterministic_for_same_seed():
    registry = default_registry()
    m1 = subtree_mutation(_tree_a(), registry, np.random.default_rng(8), fields=_FIELDS, max_depth=7)
    m2 = subtree_mutation(_tree_a(), registry, np.random.default_rng(8), fields=_FIELDS, max_depth=7)
    assert Serializer().visit(m1) == Serializer().visit(m2)


# --- Bất biến kiểu (typed GP): variation không bao giờ tạo Constant-at-PANEL ---

def _tree_with_constant_leaf() -> Node:
    """Cây có cả Constant ở slot WINDOW (5.0) lẫn signal — mồi cho hoist/subtree mutation
    để lộ lỗi tráo subtree vào/từ vị trí sai kiểu."""
    return Call(op="ts_mean", args=(Call(op="rank", args=(Field("close"),)), Constant(5.0)))


def test_subtree_mutation_preserves_type_invariant():
    """1000 lần subtree_mutation trên seed kinh điển: mọi cây kết quả phải giữ bất biến
    PANEL (không Constant ở slot PANEL)."""
    registry = default_registry()
    seed_tree = _tree_with_constant_leaf()
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        mutated = subtree_mutation(seed_tree, registry, rng, fields=_FIELDS, max_depth=7)
        _check_panel_invariant(mutated, registry)


def test_hoist_mutation_never_returns_constant_root():
    """1000 lần hoist trên cây có Constant leaf: gốc kết quả luôn là Call hoặc Field, KHÔNG
    bao giờ là Constant (Constant đứng trần không phải tín hiệu PANEL)."""
    seed_tree = _tree_with_constant_leaf()
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        hoisted = hoist_mutation(seed_tree, rng)
        assert not isinstance(hoisted, Constant), (
            f"seed={seed}: hoist trả Constant root — vi phạm bất biến PANEL"
        )


def test_crossover_never_swaps_constant_root():
    """Crossover cá thể chỉ-Constant với cây bình thường không được sinh Constant-at-PANEL
    ở bất kỳ con nào (root Constant không phải subtree PANEL-compatible)."""
    registry = default_registry()
    const_only = Constant(3.0)
    normal = _tree_a()
    for seed in range(200):
        rng = np.random.default_rng(seed)
        ca, cb = crossover(const_only, normal, rng, max_depth=7)
        _check_panel_invariant(ca, registry)
        _check_panel_invariant(cb, registry)
