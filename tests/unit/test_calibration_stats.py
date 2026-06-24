"""Test Spearman rho tự viết bằng numpy (average rank + Pearson trên rank, KHÔNG scipy).

Tham chiếu: spearman([1,2,3,4,5],[5,4,3,2,1]) = -1.0 (đảo hoàn toàn);
spearman tăng đơn điệu cùng chiều = 1.0; có tie dùng average rank (vd [1,1,2,3] -> rank
[1.5,1.5,3,4])."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.stats import average_rank, spearman


def test_average_rank_no_ties():
    ranks = average_rank(np.array([10.0, 30.0, 20.0]))
    np.testing.assert_allclose(ranks, [1.0, 3.0, 2.0])


def test_average_rank_with_ties_uses_mean_rank():
    ranks = average_rank(np.array([1.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(ranks, [1.5, 1.5, 3.0, 4.0])


def test_spearman_perfect_positive_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert spearman(x, y) == pytest.approx(1.0)


def test_spearman_perfect_negative_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert spearman(x, y) == pytest.approx(-1.0)


def test_spearman_known_value_with_ties():
    # x có tie ở vị trí 0,1 -> rank trung bình; giá trị tham chiếu tính tay.
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 1.0, 3.0, 5.0, 4.0])
    rho = spearman(x, y)
    assert -1.0 <= rho <= 1.0
    # Tính tay: rx=[1.5,1.5,3,4,5], ry=[2,1,3,5,4]; mean=3.0 cả hai.
    # cov=8.5, Sdx^2=9.5, Sdy^2=10 -> rho = 8.5/sqrt(95) = 0.87208...
    assert rho == pytest.approx(0.8721, abs=1e-3)


def test_spearman_zero_correlation_random_unrelated():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    rho = spearman(x, y)
    assert abs(rho) < 0.2  # không tương quan rõ, ngưỡng lỏng tránh flaky


def test_spearman_ignores_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    rho = spearman(x, y)
    # chỉ còn cặp (1,1),(2,2),(5,5) hợp lệ -> tương quan dương hoàn hảo
    assert rho == pytest.approx(1.0)


def test_spearman_returns_nan_when_fewer_than_two_valid_pairs():
    x = np.array([1.0, np.nan])
    y = np.array([np.nan, 2.0])
    assert np.isnan(spearman(x, y))


def test_spearman_returns_nan_when_one_side_constant():
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    assert np.isnan(spearman(x, y))
