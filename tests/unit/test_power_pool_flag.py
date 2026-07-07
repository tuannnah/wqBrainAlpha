"""Test đỏ->xanh cho helper is_power_pool (Task 3): cờ Power Pool eligibility
(Sharpe>=1.0, <=8 operator, <=3 field trừ grouping, self_corr None hoặc <=0.5)."""

from __future__ import annotations

import src.operators_local  # noqa: F401
from src.app.closed_loop_adapters import is_power_pool
from src.lang.registry import default_registry


def test_power_pool_dat_khi_don_gian_va_sharpe_du():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 1.2, 0.3, reg) is True


def test_power_pool_khong_dat_khi_sharpe_thap():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 0.8, 0.3, reg) is False


def test_power_pool_khong_dat_khi_self_corr_cao():
    reg = default_registry()
    assert is_power_pool("rank(ts_delta(close, 5))", 1.5, 0.6, reg) is False


def test_power_pool_khong_dat_khi_qua_nhieu_field():
    reg = default_registry()
    # 4 field khác nhau > 3
    expr = "add(add(close, open), add(high, low))"
    assert is_power_pool(expr, 1.5, 0.1, reg) is False
