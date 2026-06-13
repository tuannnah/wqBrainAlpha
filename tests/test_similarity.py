"""Test tương đồng AST: canon + nhánh con chung lớn nhất + ratio (GĐ3: T3.2)."""

from __future__ import annotations

from src.decorrelation.similarity import (
    largest_common_subtree,
    similarity_ratio,
    subtree_canon,
)
from src.generation.ast_utils import parse_expression as P


def test_canon_field_va_so_la_generic():
    # field -> F, số -> N: đổi field/đổi window cho cùng canon.
    assert subtree_canon(P("rank(close)")) == subtree_canon(P("rank(volume)"))
    assert subtree_canon(P("ts_mean(close, 5)")) == subtree_canon(P("ts_mean(close, 120)"))


def test_ratio_giong_het_la_1():
    assert similarity_ratio("rank(ts_mean(close, 5))", "rank(ts_mean(close, 5))") == 1.0


def test_ratio_doi_field_van_trung():
    assert similarity_ratio("rank(close)", "rank(volume)") == 1.0


def test_ratio_doi_window_van_trung():
    assert similarity_ratio("ts_mean(close, 5)", "ts_mean(close, 120)") == 1.0


def test_ratio_operator_khac_nhau_la_0():
    assert similarity_ratio("rank(ts_mean(close, 5))", "rank(ts_delta(volume, 10))") == 0.0


def test_subtree_long_nhau_tinh_la_trung():
    # ts_mean(close,5) là nhánh con của rank(ts_mean(close,5)).
    assert similarity_ratio("rank(ts_mean(close, 5))", "ts_mean(close, 5)") == 1.0


def test_largest_common_subtree_dung_kich_thuoc():
    a = P("rank(ts_mean(close, 5))")
    b = P("ts_mean(volume, 20)")
    # nhánh chung lớn nhất = ts_mean(F, N): 3 node.
    assert largest_common_subtree(a, b) == 3


def test_ratio_nhan_ca_chuoi_va_node():
    a = P("rank(close)")
    assert similarity_ratio(a, "rank(close)") == 1.0
