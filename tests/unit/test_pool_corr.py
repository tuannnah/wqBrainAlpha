"""Test PoolCorrelation.max_corr: Pearson |rho| align trên dates chung, pool rỗng ->
(0.0, None), bỏ qua alpha không đủ overlap/variance để so sánh."""

from __future__ import annotations

import numpy as np

from src.backtest.pool_corr import PoolCorrelation


def _dates(start: str, n: int) -> np.ndarray:
    return (np.datetime64(start) + np.arange(n)).astype("datetime64[D]")


def test_empty_pool_returns_zero_and_none():
    pc = PoolCorrelation(pool={})
    candidate = np.array([0.01, -0.02, 0.03])
    rho, worst_id = pc.max_corr(candidate, _dates("2021-01-01", 3))
    assert rho == 0.0
    assert worst_id is None


def test_identical_series_gives_rho_one():
    pnl = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
    dates = _dates("2021-01-01", 5)
    pc = PoolCorrelation(pool={1: (dates, pnl.copy())})
    rho, worst_id = pc.max_corr(pnl.copy(), dates)
    assert np.isclose(rho, 1.0, atol=1e-9)
    assert worst_id == 1


def test_sign_flipped_series_gives_rho_minus_one_abs_one():
    pnl = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
    dates = _dates("2021-01-01", 5)
    pc = PoolCorrelation(pool={1: (dates, pnl.copy())})
    rho, worst_id = pc.max_corr(-pnl.copy(), dates)
    # Pearson(x, -x) = -1 -> |rho| = 1
    assert np.isclose(rho, 1.0, atol=1e-9)
    assert worst_id == 1


def test_independent_series_gives_low_rho():
    rng = np.random.default_rng(42)
    pool_pnl = rng.normal(size=2000)
    candidate_pnl = rng.normal(size=2000)  # độc lập (seed khác draw)
    dates = _dates("2021-01-01", 2000)
    pc = PoolCorrelation(pool={1: (dates, pool_pnl)})
    rho, worst_id = pc.max_corr(candidate_pnl, dates)
    assert rho < 0.10  # độc lập -> rho gần 0, ngưỡng lỏng tránh flaky
    assert worst_id == 1


def test_picks_worst_alpha_id_as_max_abs_rho_across_pool():
    dates = _dates("2021-01-01", 5)
    base = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
    rng = np.random.default_rng(7)
    pool = {
        1: (dates, rng.normal(size=5)),       # độc lập, |rho| thấp
        2: (dates, base.copy()),              # giống hệt candidate -> |rho|=1
        3: (dates, rng.normal(size=5)),       # độc lập, |rho| thấp
    }
    pc = PoolCorrelation(pool=pool)
    rho, worst_id = pc.max_corr(base.copy(), dates)
    assert np.isclose(rho, 1.0, atol=1e-9)
    assert worst_id == 2


def test_partial_date_overlap_aligns_on_intersection_only():
    # Pool alpha có lịch sử dài hơn candidate; chỉ 3 ngày cuối trùng nhau.
    pool_dates = _dates("2021-01-01", 6)
    pool_pnl = np.array([100.0, 100.0, 100.0, 0.01, -0.02, 0.03])  # 3 đầu là nhiễu lớn
    candidate_dates = _dates("2021-01-04", 3)  # trùng 3 ngày cuối của pool
    candidate_pnl = np.array([0.01, -0.02, 0.03])  # giống hệt phần overlap

    pc = PoolCorrelation(pool={9: (pool_dates, pool_pnl)})
    rho, worst_id = pc.max_corr(candidate_pnl, candidate_dates)
    assert np.isclose(rho, 1.0, atol=1e-9)
    assert worst_id == 9


def test_no_date_overlap_is_skipped_not_zero_forced():
    pool_dates = _dates("2020-01-01", 3)
    pool_pnl = np.array([0.01, -0.02, 0.03])
    candidate_dates = _dates("2025-01-01", 3)  # không trùng ngày nào
    candidate_pnl = np.array([0.05, 0.05, 0.05])

    pc = PoolCorrelation(pool={5: (pool_dates, pool_pnl)})
    rho, worst_id = pc.max_corr(candidate_pnl, candidate_dates)
    assert rho == 0.0
    assert worst_id is None  # không đủ overlap -> bỏ qua, không phải "so sánh ra 0"


def test_zero_variance_pool_series_is_skipped():
    dates = _dates("2021-01-01", 5)
    flat_pnl = np.full(5, 0.02)  # std = 0 -> Pearson không xác định
    candidate_pnl = np.array([0.01, -0.02, 0.03, 0.01, -0.01])

    pc = PoolCorrelation(pool={3: (dates, flat_pnl)})
    rho, worst_id = pc.max_corr(candidate_pnl, dates)
    assert rho == 0.0
    assert worst_id is None
