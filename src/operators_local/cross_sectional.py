"""Operator cross-sectional: rank/winsorize/scale/zscore — per-row (mỗi ngày t), chỉ trên
cell in-universe (NaN tự loại nhờ nan-aware reducer numpy khi input panel có NaN ngoài
universe)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


def _row_mean_std(x: Panel):
    """(valid, n, mean, std) per-row nan-aware, tính bằng tổng có mask (không dùng
    np.nanmean/np.nanstd -> không phát RuntimeWarning với hàng toàn NaN). std ddof=0
    two-pass (mean rồi độ lệch) nên chính xác, không cancellation."""
    valid = ~np.isnan(x)
    n = valid.sum(axis=1, keepdims=True)
    n_safe = np.where(n > 0, n, 1)
    mean = np.where(valid, x, 0.0).sum(axis=1, keepdims=True) / n_safe
    dev = np.where(valid, x - mean, 0.0)
    std = np.sqrt((dev * dev).sum(axis=1, keepdims=True) / n_safe)
    return valid, n, mean, std


@register(name="rank", category=OpCategory.CROSS_SECTIONAL,
          signature=(ArgKind.PANEL,), bounded=True, commutative=False)
def rank(ctx: EvalContext, x: Panel) -> Panel:
    # Vectorize per-row (bỏ vòng t): đẩy NaN xuống cuối bằng +inf, hạng 0-based qua
    # argsort-của-argsort (stable, giữ nguyên xử lý ties theo thứ tự xuất hiện), chuẩn
    # hóa [0,1] theo n_valid-1 mỗi hàng; ô NaN -> NaN.
    valid = ~np.isnan(x)
    n_valid = valid.sum(axis=1, keepdims=True)
    filled = np.where(valid, x, np.inf)
    order = np.argsort(filled, axis=1, kind="stable")
    rank_pos = np.argsort(order, axis=1).astype(np.float64)  # hạng 0-based
    denom = np.where(n_valid > 1, n_valid - 1, 1)
    with np.errstate(invalid="ignore"):
        out = np.where(valid, rank_pos / denom, np.nan)
    return out


@register(name="winsorize", category=OpCategory.CROSS_SECTIONAL,
          signature=(ArgKind.PANEL, ArgKind.SCALAR), bounded=False, commutative=False)
def winsorize(ctx: EvalContext, x: Panel, std_count: float) -> Panel:
    # Vectorize per-row: clip vào [mean±k*std] tính trên mỗi hàng (nan-aware). Hàng có
    # <2 quan sát hoặc std=0 giữ nguyên; ô NaN giữ NaN.
    out = x.copy()
    valid, n_valid, mean, std = _row_mean_std(x)
    lo, hi = mean - std_count * std, mean + std_count * std
    clipped = np.clip(x, lo, hi)
    apply = valid & (n_valid >= 2) & (std > 0.0)
    return np.where(apply, clipped, out)


@register(name="zscore", category=OpCategory.CROSS_SECTIONAL,
          signature=(ArgKind.PANEL,), bounded=False, commutative=False)
def zscore(ctx: EvalContext, x: Panel) -> Panel:
    # Vectorize per-row: (x-mean)/std mỗi hàng (nan-aware). Hàng <2 quan sát hoặc std=0
    # -> NaN; ô NaN -> NaN.
    valid, n_valid, mean, std = _row_mean_std(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - mean) / std
    ok = valid & (n_valid >= 2) & (std > 0.0)
    return np.where(ok, z, np.nan)


@register(name="scale", category=OpCategory.SCALING,
          signature=(ArgKind.PANEL,), bounded=False, gp_usable=False, commutative=False)
def scale(ctx: EvalContext, x: Panel) -> Panel:
    """Rescale per-row để tổng |giá trị| trong-universe = 1 (rank/sign-preserving,
    wrapper config — không tham gia core GP search, B5)."""
    valid = ~np.isnan(x)
    total = np.where(valid, np.abs(x), 0.0).sum(axis=1, keepdims=True)  # Σ|x| in-universe
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = x / total
    ok = valid & (total > 0.0)
    return np.where(ok, scaled, np.nan)
