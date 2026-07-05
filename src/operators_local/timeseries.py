"""Operator time-series: trailing window [t-d+1, t] (đúng d quan sát, kể cả t); thiếu đủ
lịch sử -> NaN. KHÔNG bao giờ đọc rows > t (no-look-ahead, B6/Global Constraints)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


def _window_slice(t: int, d: int) -> slice | None:
    start = t - d + 1
    if start < 0:
        return None
    return slice(start, t + 1)


@register(name="ts_mean", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_mean(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        with np.errstate(invalid="ignore"):
            out[t] = np.nanmean(x[win], axis=0)
    return out


@register(name="ts_std", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_std(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        with np.errstate(invalid="ignore"):
            out[t] = np.nanstd(x[win], axis=0)
    return out


@register(name="ts_sum", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_sum(ctx: EvalContext, x: Panel, d: int) -> Panel:
    """Tổng x trong d ngày gần nhất (trailing [t-d+1, t]) — khớp mô tả thật WQ Brain
    "Sum values of x for the past d days."."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        with np.errstate(invalid="ignore"):
            out[t] = np.nansum(x[win], axis=0)
    return out


@register(name="ts_std_dev", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_std_dev(ctx: EvalContext, x: Panel, d: int) -> Panel:
    """Độ lệch chuẩn của x trong d ngày gần nhất — khớp mô tả thật WQ Brain "Calculates
    the standard deviation of a data series x over the past d days, measuring how much
    the values deviate from their mean during that period." (cùng cách tính với ts_std,
    đăng ký tên riêng để khớp đúng tên operator thật trên platform)."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        with np.errstate(invalid="ignore"):
            out[t] = np.nanstd(x[win], axis=0)
    return out


@register(name="ts_delay", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False,
          gp_usable=False)
def ts_delay(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    if d < x.shape[0]:
        out[d:] = x[: x.shape[0] - d]
    return out


@register(name="ts_delta", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_delta(ctx: EvalContext, x: Panel, d: int) -> Panel:
    # annotate rõ kiểu biến trung gian: decorator @register làm mất signature cụ thể
    # của ts_delay (mypy thấy Callable[..., Any]) -> gọi trực tiếp trong biểu thức trừ
    # sẽ suy luận ra Any; gán qua biến đã khai kiểu Panel để giữ mypy --strict sạch.
    delayed: Panel = ts_delay(ctx, x, d)
    return x - delayed


@register(name="ts_rank", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=True, commutative=False)
def ts_rank(ctx: EvalContext, x: Panel, d: int) -> Panel:
    # Vectorize theo cột (thay vòng lồng Python T*assets cũ). Với mỗi t: đếm số quan
    # sát hợp lệ trong cửa sổ <= giá trị hiện tại rồi chuẩn hóa [0,1]. NaN so sánh ra
    # False nên tự loại; ô hiện tại NaN hoặc cửa sổ rỗng -> NaN.
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    with np.errstate(invalid="ignore"):
        for t in range(x.shape[0]):
            win = _window_slice(t, d)
            if win is None:
                continue
            window = x[win]
            valid = ~np.isnan(window)
            n_valid = valid.sum(axis=0)
            cur = x[t]  # giá trị hiện tại mỗi cột
            count = (valid & (window <= cur)).sum(axis=0)  # gồm cả chính nó
            denom = np.where(n_valid > 1, n_valid - 1, 1)
            ok = (n_valid > 0) & ~np.isnan(cur)
            out[t] = np.where(ok, (count - 1) / denom, np.nan)
    return out


@register(name="ts_zscore", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_zscore(ctx: EvalContext, x: Panel, d: int) -> Panel:
    mean: Panel = ts_mean(ctx, x, d)
    std: Panel = ts_std(ctx, x, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: Panel = (x - mean) / std
    out[std == 0.0] = np.nan
    return out


@register(name="ts_corr", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.PANEL, ArgKind.WINDOW), bounded=True,
          commutative=False)
def ts_corr(ctx: EvalContext, x: Panel, y: Panel, d: int) -> Panel:
    # Vectorize theo cột: mỗi t xử lý cả hàng cổ phiếu cùng lúc (thay 2 vòng lồng
    # Python cũ ~O(T*assets) call numpy nhỏ -> chậm hàng trăm lần). Pearson tính bằng
    # công thức centered hai-lượt (mean rồi độ lệch) để khớp np.corrcoef về số học.
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(x.shape[0]):
            win = _window_slice(t, d)
            if win is None:
                continue
            wx, wy = x[win], y[win]
            valid = ~np.isnan(wx) & ~np.isnan(wy)  # loại NaN theo CẶP
            n = valid.sum(axis=0)
            n_safe = np.where(n > 0, n, 1)  # tránh chia 0 khi cột rỗng
            mx = np.where(valid, wx, 0.0).sum(axis=0) / n_safe
            my = np.where(valid, wy, 0.0).sum(axis=0) / n_safe
            dx = np.where(valid, wx - mx, 0.0)
            dy = np.where(valid, wy - my, 0.0)
            cov = (dx * dy).sum(axis=0)
            vx = (dx * dx).sum(axis=0)
            vy = (dy * dy).sum(axis=0)
            corr = cov / np.sqrt(vx * vy)
            # valid<2 hoặc một biến hằng (std=0 -> vx/vy=0) -> NaN, khớp bản gốc.
            ok = (n >= 2) & (vx > 0.0) & (vy > 0.0)
            out[t] = np.where(ok, corr, np.nan)
    return out


@register(name="ts_decay_linear", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False,
          gp_usable=False)
def ts_decay_linear(ctx: EvalContext, x: Panel, d: int) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    weights = np.arange(1, d + 1, dtype=np.float64)  # xa nhất=1 ... gần nhất(t)=d
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        window = x[win]
        for col in range(x.shape[1]):
            series = window[:, col]
            valid = ~np.isnan(series)
            if not np.any(valid):
                continue
            w = weights[valid]
            out[t, col] = float(np.sum(series[valid] * w) / np.sum(w))
    return out


@register(name="ts_backfill", category=OpCategory.TIME_SERIES,
          signature=(ArgKind.PANEL, ArgKind.WINDOW), bounded=False, commutative=False)
def ts_backfill(ctx: EvalContext, x: Panel, d: int) -> Panel:
    """Lấp NaN bằng giá trị hợp lệ gần nhất trong d hàng trước (rows <= t); quá d hàng
    không tìm thấy giá trị hợp lệ -> giữ NaN."""
    out = x.copy()
    d = int(d)
    for col in range(x.shape[1]):
        last_valid_row = -1
        for t in range(x.shape[0]):
            if not np.isnan(x[t, col]):
                last_valid_row = t
            elif last_valid_row >= 0 and (t - last_valid_row) <= d:
                out[t, col] = x[last_valid_row, col]
    return out
