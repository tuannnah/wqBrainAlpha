"""Test tương đồng AST: canon + nhánh con chung lớn nhất + ratio (GĐ3: T3.2)."""

from __future__ import annotations

from src.decorrelation.similarity import (
    avoid_subtree_canons,
    largest_common_subtree,
    similarity_ratio,
    subtree_canon,
)
from src.generation.ast_utils import parse_expression as P


def test_canon_mac_dinh_field_blind():
    # Mặc định (field_aware=False): field->F, đổi field/window đều cùng canon.
    assert subtree_canon(P("rank(close)")) == subtree_canon(P("rank(volume)"))
    assert subtree_canon(P("ts_mean(close, 5)")) == subtree_canon(P("ts_mean(close, 120)"))


def test_canon_field_aware_giu_ten_field():
    # field_aware=True: chỉ số -> N; field GIỮ tên (đổi field -> canon khác).
    assert subtree_canon(P("ts_mean(close, 5)"), field_aware=True) == subtree_canon(
        P("ts_mean(close, 120)"), field_aware=True
    )
    assert subtree_canon(P("rank(close)"), field_aware=True) != subtree_canon(
        P("rank(volume)"), field_aware=True
    )


def test_ratio_giong_het_la_1():
    assert similarity_ratio("rank(ts_mean(close, 5))", "rank(ts_mean(close, 5))") == 1.0


def test_ratio_mac_dinh_doi_field_van_trung():
    # Mặc định field-blind: đổi field vẫn trùng (phục vụ khử-trùng-cấu-trúc).
    assert similarity_ratio("rank(close)", "rank(volume)") == 1.0


def test_ratio_field_aware_doi_field_la_khac():
    # field_aware: field khác (dù cùng cấu trúc) -> không còn trùng (fix mù dataset).
    assert similarity_ratio("rank(close)", "rank(volume)", field_aware=True) == 0.0


def test_ratio_field_aware_dataset_thay_the_khong_trung():
    # field_aware: cùng khung ts_delta(...,N) nhưng field dataset khác -> không trùng.
    assert (
        similarity_ratio(
            "ts_delta(actual_eps_value, 5)", "ts_delta(close, 5)", field_aware=True
        )
        == 0.0
    )


def test_ratio_doi_window_van_trung():
    assert similarity_ratio("ts_mean(close, 5)", "ts_mean(close, 120)") == 1.0


def test_ratio_operator_khac_nhau_la_0():
    assert similarity_ratio("rank(ts_mean(close, 5))", "rank(ts_delta(volume, 10))") == 0.0


def test_subtree_long_nhau_tinh_la_trung():
    # ts_mean(close,5) là nhánh con của rank(ts_mean(close,5)).
    assert similarity_ratio("rank(ts_mean(close, 5))", "ts_mean(close, 5)") == 1.0


def test_largest_common_subtree_dung_kich_thuoc():
    a = P("rank(ts_mean(close, 5))")
    b = P("ts_mean(close, 20)")
    # nhánh chung lớn nhất = ts_mean(close, N): 3 node (cùng field, window gộp).
    assert largest_common_subtree(a, b) == 3


def test_ratio_nhan_ca_chuoi_va_node():
    a = P("rank(close)")
    assert similarity_ratio(a, "rank(close)") == 1.0


def test_avoid_subtree_canons_gop_pass_va_fail():
    """Gộp bộ khung phổ biến trong pass + bộ khung lặp trong failures (>=2 lần)."""
    passed = ["rank(ts_mean(close, 5))"] * 3  # canon phổ biến trong pass
    failed = ["rank(ts_delta(volume, 10))", "rank(ts_delta(open, 20))"]  # ts_delta lặp 2
    avoid = avoid_subtree_canons(passed, failed, passed_min=3, failed_min=2)
    assert "rank(ts_mean(F,N))" in avoid  # từ pass
    assert "ts_delta(F,N)" in avoid       # từ fail (lặp >= 2)


def test_avoid_subtree_canons_fail_don_le_khong_tinh():
    """Cấu trúc chỉ xuất hiện 1 failure (failed_min=2) -> không bị đưa vào avoid."""
    avoid = avoid_subtree_canons([], ["rank(ts_corr(close, volume, 20))"], failed_min=2)
    assert avoid == set()
