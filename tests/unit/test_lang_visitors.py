"""Test các visitor mới trên AST (OperatorCollector)."""

from __future__ import annotations

from src.lang.parser import parse_expression
from src.lang.visitors import OperatorCollector


def test_operator_collector_don_gian():
    node = parse_expression("rank(close)")
    assert OperatorCollector().visit(node) == {"rank"}


def test_operator_collector_long_nhau():
    node = parse_expression("rank(add(ts_delta(close, 5), open))")
    assert OperatorCollector().visit(node) == {"rank", "add", "ts_delta"}


def test_operator_collector_khong_co_operator():
    node = parse_expression("close")
    assert OperatorCollector().visit(node) == set()


def test_operator_collector_dem_operator_lap_lai_chi_1_lan():
    node = parse_expression("add(rank(close), rank(open))")
    assert OperatorCollector().visit(node) == {"add", "rank"}
