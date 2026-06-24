"""Test cây AST sealed hierarchy: Constant/Field/Call + NodeVisitor."""

from __future__ import annotations

import pytest

from src.lang.ast import Call, Constant, Field, Node, NodeVisitor


class _CountingVisitor:
    """Visitor tối giản để xác nhận dispatch accept() đúng phương thức."""

    def __init__(self) -> None:
        self.constants = 0
        self.fields = 0
        self.calls = 0

    def visit_constant(self, node: Constant) -> str:
        self.constants += 1
        return f"const:{node.value}"

    def visit_field(self, node: Field) -> str:
        self.fields += 1
        return f"field:{node.name}"

    def visit_call(self, node: Call) -> str:
        self.calls += 1
        return f"call:{node.op}"


def test_constant_is_frozen_and_hashable():
    c = Constant(1.5)
    assert c.value == 1.5
    with pytest.raises(AttributeError):
        c.value = 2.0  # type: ignore[misc]
    hash(c)  # không raise


def test_field_children_empty():
    f = Field("close")
    assert f.children() == ()
    assert f.name == "close"


def test_call_children_returns_args():
    a, b = Field("close"), Constant(5.0)
    call = Call(op="ts_mean", args=(a, b))
    assert call.children() == (a, b)
    assert call.op == "ts_mean"


def test_call_is_frozen_and_hashable():
    call = Call(op="rank", args=(Field("close"),))
    with pytest.raises(AttributeError):
        call.op = "ts_mean"  # type: ignore[misc]
    hash(call)  # không raise (args là tuple — hashable)


def test_accept_dispatches_to_correct_visit_method():
    v = _CountingVisitor()
    tree = Call(op="rank", args=(Field("close"), Constant(5.0)))
    result = tree.accept(v)
    assert result == "call:rank"
    assert v.calls == 1
    tree.args[0].accept(v)
    tree.args[1].accept(v)
    assert v.fields == 1 and v.constants == 1


def test_node_is_abstract_base():
    assert issubclass(Constant, Node)
    assert issubclass(Field, Node)
    assert issubclass(Call, Node)
    with pytest.raises(TypeError):
        Node()  # type: ignore[abstract]


def test_node_visitor_is_protocol_runtime_checkable_via_duck_typing():
    # NodeVisitor là Protocol thuần (không @runtime_checkable bắt buộc) —
    # xác nhận _CountingVisitor thỏa cấu trúc bằng cách dùng trực tiếp.
    v: NodeVisitor[str] = _CountingVisitor()
    assert v.visit_field(Field("open")) == "field:open"
