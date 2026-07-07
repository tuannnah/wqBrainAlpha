"""Các operator local KHÔNG được phun RuntimeWarning (overflow/invalid/empty-slice) khi
gặp NaN, inf hoặc giá trị lớn — vì log cảnh báo này (a) làm nhiễu log chạy dài, che lỗi
thật, và (b) báo hiệu inf/overflow lọt vào chỉ số local. Giá trị NaN vẫn phải propagate
đúng (sentinel 'thiếu'); ta chỉ chặn TIẾNG ỒN của cảnh báo, không đổi ngữ nghĩa.
"""

from __future__ import annotations

import types
import warnings

import numpy as np

from src.operators_local.arithmetic import multiply
from src.operators_local.group import group_neutralize
from src.operators_local.timeseries import (
    ts_corr,
    ts_decay_linear,
    ts_delta,
    ts_mean,
    ts_std,
    ts_std_dev,
)


def _runtime_warnings(fn):
    """Chạy fn(), trả danh sách RuntimeWarning phát ra (numpy FP warning + empty-slice)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all="warn"):
            fn()
    return [w for w in caught if issubclass(w.category, RuntimeWarning)]


def _nasty_panel() -> np.ndarray:
    """Panel (12×3) trộn NaN, inf và giá trị đủ lớn để x*x tràn float64 (>1e154)."""
    x = np.array(
        [
            [1.0, np.nan, 2.0],
            [np.inf, 3.0, np.nan],
            [1e200, 4.0, 5.0],
            [2.0, np.nan, 6.0],
            [np.nan, np.nan, np.nan],  # hàng toàn NaN -> empty-slice tiềm tàng
            [3.0, 7.0, -np.inf],
            [1e200, 8.0, 9.0],
            [4.0, np.nan, 10.0],
            [5.0, 11.0, np.nan],
            [np.nan, 12.0, 13.0],
            [6.0, 14.0, 15.0],
            [7.0, np.nan, 16.0],
        ],
        dtype=np.float64,
    )
    return x


def test_timeseries_ops_khong_phun_runtime_warning() -> None:
    x = _nasty_panel()
    y = _nasty_panel()[::-1].copy()
    for name, call in [
        ("ts_mean", lambda: ts_mean(None, x, 3)),
        ("ts_std", lambda: ts_std(None, x, 3)),
        ("ts_std_dev", lambda: ts_std_dev(None, x, 3)),
        ("ts_delta", lambda: ts_delta(None, x, 2)),
        ("ts_corr", lambda: ts_corr(None, x, y, 4)),
        ("ts_decay_linear", lambda: ts_decay_linear(None, x, 3)),
    ]:
        ws = _runtime_warnings(call)
        assert not ws, f"{name} phun RuntimeWarning: {[str(w.message) for w in ws]}"


def test_arithmetic_multiply_khong_phun_runtime_warning() -> None:
    x = _nasty_panel()
    ws = _runtime_warnings(lambda: multiply(None, x, x))
    assert not ws, f"multiply phun RuntimeWarning: {[str(w.message) for w in ws]}"


def test_group_neutralize_khong_phun_runtime_warning() -> None:
    x = _nasty_panel()
    groups = np.array([[0, 0, 1]] * x.shape[0], dtype=np.float64)
    ctx = types.SimpleNamespace(data=types.SimpleNamespace(groups={"sector": groups}))
    ws = _runtime_warnings(lambda: group_neutralize(ctx, x, "sector"))
    assert not ws, f"group_neutralize phun RuntimeWarning: {[str(w.message) for w in ws]}"
