"""Test thống kê nhánh con phổ biến để tránh lặp (GĐ3: T3.6)."""

from __future__ import annotations

from src.decorrelation.similarity import common_subtrees


def test_dem_subtree_xuat_hien_nhieu_alpha():
    # ts_mean(F, N) xuất hiện ở cả 3 alpha (đổi field/window vẫn cùng canon).
    exprs = [
        "rank(ts_mean(close, 5))",
        "ts_mean(volume, 20)",
        "rank(ts_mean(open, 10))",
    ]
    result = dict(common_subtrees(exprs))
    assert result.get("ts_mean(F,N)") == 3


def test_chi_giu_subtree_dat_min_count():
    # ts_delta chỉ ở 1 alpha -> bị loại khi min_count=2.
    exprs = [
        "rank(ts_mean(close, 5))",
        "ts_mean(volume, 20)",
        "ts_delta(close, 3)",
    ]
    canons = {c for c, _ in common_subtrees(exprs, min_count=2)}
    assert "ts_mean(F,N)" in canons
    assert "ts_delta(F,N)" not in canons


def test_sort_giam_theo_so_lan():
    exprs = [
        "rank(ts_mean(close, 5))",
        "rank(ts_mean(volume, 20))",
        "ts_delta(close, 3)",
    ]
    result = common_subtrees(exprs, min_count=1)
    counts = [n for _, n in result]
    assert counts == sorted(counts, reverse=True)


def test_top_n_gioi_han_so_luong():
    exprs = [
        "rank(ts_mean(close, 5))",
        "ts_delta(volume, 3)",
        "ts_corr(close, volume, 10)",
    ]
    result = common_subtrees(exprs, min_count=1, top_n=2)
    assert len(result) <= 2


def test_bo_qua_bieu_thuc_parse_loi():
    exprs = ["rank(ts_mean(close, 5))", "bad ))(", "ts_mean(volume, 20)"]
    result = dict(common_subtrees(exprs, min_count=1))
    assert result.get("ts_mean(F,N)") == 2


def test_tap_rong_tra_danh_sach_rong():
    assert common_subtrees([]) == []
