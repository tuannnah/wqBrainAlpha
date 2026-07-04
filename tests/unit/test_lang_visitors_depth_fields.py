"""Test DepthVisitor (đếm cả wrapper) và FieldCollector."""

from __future__ import annotations

from src.lang.ast import Call, Constant, Field
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, FieldCollector


def test_depth_of_leaf_is_one():
    assert DepthVisitor().visit(Field("close")) == 1
    assert DepthVisitor().visit(Constant(5.0)) == 1


def test_depth_of_single_call_is_two():
    tree = Call(op="rank", args=(Field("close"),))
    assert DepthVisitor().visit(tree) == 2


def test_depth_counts_wrapper_call():
    # rank(ts_mean(close, 20)) -> rank(1) -> ts_mean(2) -> close/20(3) => depth 3
    tree = Call(op="rank", args=(Call(op="ts_mean", args=(Field("close"), Constant(20.0))),))
    assert DepthVisitor().visit(tree) == 3


def test_depth_takes_max_over_multiple_children():
    # add(close, ts_mean(close,20)) -> add(1) -> [close(2), ts_mean(2)->[close,20](3)]
    tree = Call(op="add", args=(
        Field("close"),
        Call(op="ts_mean", args=(Field("close"), Constant(20.0))),
    ))
    assert DepthVisitor().visit(tree) == 3


def test_field_collector_single_field():
    tree = Call(op="rank", args=(Field("close"),))
    assert FieldCollector(default_registry()).visit(tree) == {"close"}


def test_field_collector_multiple_distinct_fields_deduped():
    tree = Call(op="add", args=(Field("close"), Call(op="ts_mean", args=(Field("close"), Constant(20.0)))))
    assert FieldCollector(default_registry()).visit(tree) == {"close"}


def test_field_collector_no_fields_for_constants_only():
    tree = Call(op="add", args=(Constant(1.0), Constant(2.0)))
    assert FieldCollector(default_registry()).visit(tree) == set()


def test_field_collector_two_distinct_fields():
    tree = Call(op="add", args=(Field("close"), Field("open")))
    assert FieldCollector(default_registry()).visit(tree) == {"close", "open"}


def test_field_collector_bo_qua_tham_so_group_cua_group_neutralize():
    """Bug thật: group_neutralize(x, sector) có tham số 2 là GROUP (tên nhóm), không phải
    field dữ liệu -- FieldCollector KHÔNG được coi 'sector' là field."""
    import src.operators_local  # noqa: F401  (side-effect: nạp group_neutralize vào registry)
    node = parse("group_neutralize(close, sector)")
    assert FieldCollector(default_registry()).visit(node) == {"close"}
