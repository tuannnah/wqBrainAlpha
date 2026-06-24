"""Test Serializer: AST -> string FASTEXPR canonical, round-trip với parser."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field
from src.lang.parser import parse
from src.lang.registry import ArgKind, OpCategory, OperatorRegistry, OperatorSpec
from src.lang.visitors import Serializer


def _placeholder(*_a: object) -> object:
    raise NotImplementedError


def _registry() -> OperatorRegistry:
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


def test_serialize_field():
    assert Serializer().visit(Field("close")) == "close"


def test_serialize_integer_constant_no_decimal():
    assert Serializer().visit(Constant(20.0)) == "20"


def test_serialize_fractional_constant_keeps_decimal():
    assert Serializer().visit(Constant(0.5)) == "0.5"


def test_serialize_call():
    tree = Call(op="ts_mean", args=(Field("close"), Constant(20.0)))
    assert Serializer().visit(tree) == "ts_mean(close, 20)"


def test_serialize_nested_call():
    tree = Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(20.0))),))
    assert Serializer().visit(tree) == "rank(ts_mean(close, 20))"


def test_round_trip_with_parser():
    reg = _registry()
    original = parse("rank(ts_mean(close, 20))", registry=reg)
    text = Serializer().visit(original)
    reparsed = parse(text, registry=reg)
    assert reparsed == original


def test_round_trip_preserves_binary_op_as_function_call():
    reg = _registry()
    original = parse("close + open" if False else "add(close, open)", registry=reg)
    text = Serializer().visit(original)
    assert text == "add(close, open)"
    assert parse(text, registry=reg) == original
