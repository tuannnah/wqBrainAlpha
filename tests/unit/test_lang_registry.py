"""Test OperatorRegistry: đăng ký, lookup, lọc theo category/gp_usable."""

from __future__ import annotations

import pytest

from src.lang.registry import (
    ArgKind,
    OpCategory,
    OperatorRegistry,
    OperatorSpec,
    default_registry,
)


def _placeholder(*_args: object) -> object:
    raise NotImplementedError("placeholder test impl")


def test_register_and_get_roundtrip():
    reg = OperatorRegistry()
    spec = OperatorSpec(
        name="rank",
        category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,),
        impl=_placeholder,
        bounded=True,
    )
    reg.register(spec)
    assert reg.get("rank") is spec


def test_get_unknown_op_raises_keyerror():
    reg = OperatorRegistry()
    with pytest.raises(KeyError):
        reg.get("not_an_op")


def test_by_category_filters():
    reg = OperatorRegistry()
    rank_spec = OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True,
    )
    ts_mean_spec = OperatorSpec(
        name="ts_mean", category=OpCategory.TIME_SERIES,
        signature=(ArgKind.PANEL, ArgKind.WINDOW), impl=_placeholder, bounded=False,
    )
    reg.register(rank_spec)
    reg.register(ts_mean_spec)
    assert reg.by_category(OpCategory.CROSS_SECTIONAL) == [rank_spec]
    assert reg.by_category(OpCategory.TIME_SERIES) == [ts_mean_spec]


def test_gp_function_set_excludes_non_gp_usable():
    reg = OperatorRegistry()
    core = OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True, gp_usable=True,
    )
    wrapper = OperatorSpec(
        name="group_neutralize", category=OpCategory.GROUP,
        signature=(ArgKind.PANEL, ArgKind.GROUP), impl=_placeholder, bounded=False,
        gp_usable=False,
    )
    reg.register(core)
    reg.register(wrapper)
    fn_set = reg.gp_function_set()
    assert core in fn_set
    assert wrapper not in fn_set


def test_operator_spec_is_frozen():
    spec = OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True,
    )
    with pytest.raises(AttributeError):
        spec.name = "other"  # type: ignore[misc]


def test_default_registry_has_minimal_phase1_ops():
    """6 operator tối thiểu Phase 1 phải tồn tại trong default_registry(). Từ Phase 2,
    operators_local nạp impl thật cho các op này (đè placeholder _not_implemented), nên
    test KHÔNG còn kiểm tra hành vi raise NotImplementedError — việc đó phụ thuộc thứ tự
    import operators_local (REGISTRY là singleton toàn cục) và không còn đúng ngữ nghĩa.
    Test chỉ xác nhận op tồn tại + tên khớp, độc lập với việc operators_local đã import."""
    reg = default_registry()
    for name in ("rank", "ts_mean", "add", "subtract", "multiply", "divide"):
        spec = reg.get(name)
        assert spec.name == name


def test_default_registry_arithmetic_ops_are_panel_panel_binary():
    reg = default_registry()
    for name in ("add", "subtract", "multiply", "divide"):
        spec = reg.get(name)
        assert spec.signature == (ArgKind.PANEL, ArgKind.PANEL)


def test_default_registry_add_is_commutative_others_not():
    reg = default_registry()
    assert reg.get("add").commutative is True
    assert reg.get("subtract").commutative is False
    assert reg.get("divide").commutative is False


def test_neutralization_decay_delay_not_in_gp_function_set():
    """B5 stage separation: regression_neut/vector_neut (neutralization) và
    ts_decay_linear/ts_delay (config wrapper của PortfolioConfig Phase 3) KHÔNG được nằm
    trong function set GP — chúng là stage wrapper, không phải signal core."""
    import src.operators_local  # noqa: F401  (side-effect: nạp operator thật vào REGISTRY)

    reg = default_registry()
    fn_names = {spec.name for spec in reg.gp_function_set()}
    for excluded in ("regression_neut", "vector_neut", "ts_decay_linear", "ts_delay"):
        assert excluded not in fn_names, (
            f"operator stage-wrapper {excluded!r} không được có trong gp_function_set()"
        )
