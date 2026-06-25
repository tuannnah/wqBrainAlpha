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
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        window = x[win]
        for col in range(x.shape[1]):
            series = window[:, col]
            valid = ~np.isnan(series)
            n_valid = int(valid.sum())
            if n_valid == 0 or np.isnan(x[t, col]):
                continue
            vals = series[valid]
            denom = n_valid - 1 if n_valid > 1 else 1
            out[t, col] = float(np.sum(vals <= x[t, col]) - 1) / denom
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
    out = np.full_like(x, np.nan, dtype=np.float64)
    d = int(d)
    for t in range(x.shape[0]):
        win = _window_slice(t, d)
        if win is None:
            continue
        wx, wy = x[win], y[win]
        for col in range(x.shape[1]):
            sx, sy = wx[:, col], wy[:, col]
            valid = ~np.isnan(sx) & ~np.isnan(sy)
            if int(valid.sum()) < 2:
                continue
            sxv, syv = sx[valid], sy[valid]
            if np.std(sxv) == 0.0 or np.std(syv) == 0.0:
                continue
            out[t, col] = float(np.corrcoef(sxv, syv)[0, 1])
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
