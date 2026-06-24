"""Test parser: string FASTEXPR-subset -> AST, validate operator/arity qua registry."""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.lang.ast import Call, Constant, Field
from src.lang.parser import ParseError, parse
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec


def _placeholder(*_a: object) -> object:
    raise NotImplementedError


def _registry_with_rank_and_arith() -> OperatorRegistry:
    reg = OperatorRegistry()
    reg.register(OperatorSpec(
        name="rank", category=OpCategory.CROSS_SECTIONAL,
        signature=(ArgKind.PANEL,), impl=_placeholder, bounded=True,
    ))
    reg.register(OperatorSpec(
        name="ts_mean", category=OpCategory.TIME_SERIES,
        signature=(ArgKind.PANEL, ArgKind.WINDOW), impl=_placeholder, bounded=False,
    ))
    reg.register(OperatorSpec(
        name="add", category=OpCategory.ARITHMETIC,
        signature=(ArgKind.PANEL, ArgKind.PANEL), impl=_placeholder, bounded=False,
        commutative=True,
    ))
    return reg


def test_parse_field_leaf():
    node = parse("close", registry=_registry_with_rank_and_arith())
    assert node == Field("close")


def test_parse_number_leaf():
    node = parse("5", registry=_registry_with_rank_and_arith())
    assert node == Constant(5.0)


def test_parse_call_single_arg():
    node = parse("rank(close)", registry=_registry_with_rank_and_arith())
    assert node == Call(op="rank", args=(Field("close"),))


def test_parse_call_with_window():
    node = parse("ts_mean(close, 20)", registry=_registry_with_rank_and_arith())
    assert node == Call(op="ts_mean", args=(Field("close"), Constant(20.0)))


def test_parse_binary_plus_maps_to_add():
    node = parse("close + open", registry=_registry_with_rank_and_arith())
    assert node == Call(op="add", args=(Field("close"), Field("open")))


def test_parse_nested_call():
    node = parse("rank(ts_mean(close, 20))", registry=_registry_with_rank_and_arith())
    assert node == Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(20.0))),))


def test_parse_unknown_operator_raises_parse_error():
    with pytest.raises(ParseError, match="not_an_op"):
        parse("not_an_op(close)", registry=_registry_with_rank_and_arith())


def test_parse_wrong_arity_raises_parse_error():
    with pytest.raises(ParseError, match="arity"):
        parse("rank(close, open)", registry=_registry_with_rank_and_arith())


def test_parse_invalid_syntax_raises_parse_error():
    with pytest.raises(ParseError):
        parse("rank(", registry=_registry_with_rank_and_arith())


def test_parse_uses_default_registry_when_none_given():
    # default_registry() (Task 1.3) đã có "rank" đăng ký sẵn (impl placeholder Phase 1).
    node = parse("rank(close)")
    assert node == Call(op="rank", args=(Field("close"),))


def test_module_runs_as_main_and_prints_node():
    result = subprocess.run(
        [sys.executable, "-m", "src.lang.parser", "rank(close)"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "rank" in result.stdout
    assert "close" in result.stdout
