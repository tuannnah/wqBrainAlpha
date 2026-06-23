"""Test ComplexityVisitor: node count toàn cây (leaf + Call)."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field
from src.lang.visitors import ComplexityVisitor


def test_complexity_of_single_leaf_is_one():
    assert ComplexityVisitor().visit(Field("close")) == 1
    assert ComplexityVisitor().visit(Constant(5.0)) == 1


def test_complexity_counts_call_plus_children():
    tree = Call(op="rank", args=(Field("close"),))
    # rank + close = 2
    assert ComplexityVisitor().visit(tree) == 2


def test_complexity_of_nested_tree():
    # ts_mean(close, 20) -> ts_mean + close + 20 = 3 ; rank(...) -> rank + 3 = 4
    tree = Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(20.0))),))
    assert ComplexityVisitor().visit(tree) == 4


def test_complexity_of_binary_with_two_fields():
    tree = Call(op="add", args=(Field("close"), Field("open")))
    # add + close + open = 3
    assert ComplexityVisitor().visit(tree) == 3
