"""Operator cross-sectional: rank/winsorize/scale/zscore — per-row (mỗi ngày t), chỉ trên
cell in-universe (NaN tự loại nhờ nan-aware reducer numpy khi input panel có NaN ngoài
universe)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="rank", category=OpCategory.CROSS_SECTIONAL,
          signature=(ArgKind.PANEL,), bounded=True, commutative=False)
def rank(ctx: EvalContext, x: Panel) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        valid = ~np.isnan(row)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        order = np.argsort(row[valid], kind="stable")
        ranks = np.empty(n_valid, dtype=np.float64)
        denom = n_valid - 1 if n_valid > 1 else 1
        ranks[order] = np.arange(n_valid, dtype=np.float64) / denom
        out[t][valid] = ranks
    return out


@register(name="winsorize", category=OpCategory.CROSS_SECTIONAL,
          signature=(ArgKind.PANEL, ArgKind.SCALAR), bounded=False, commutative=False)
def winsorize(ctx: EvalContext, x: Panel, std_count: float) -> Panel:
    out = x.copy()
    for t in range(x.shape[0]):
        row = out[t]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        mean = float(np.mean(row[valid]))
        std = float(np.std(row[valid]))
        if std == 0.0:
            continue
        lo, hi = mean - std_count * std, mean + std_count * std
        row[valid] = np.clip(row[valid], lo, hi)
    return out


@register(name="zscore", category=OpCategory.CROSS_SECTIONAL,
          signature=(ArgKind.PANEL,), bounded=False, commutative=False)
def zscore(ctx: EvalContext, x: Panel) -> Panel:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        mean = float(np.mean(row[valid]))
        std = float(np.std(row[valid]))
        if std == 0.0:
            continue
        out[t][valid] = (row[valid] - mean) / std
    return out


@register(name="scale", category=OpCategory.SCALING,
          signature=(ArgKind.PANEL,), bounded=False, gp_usable=False, commutative=False)
def scale(ctx: EvalContext, x: Panel) -> Panel:
    """Rescale per-row để tổng |giá trị| trong-universe = 1 (rank/sign-preserving,
    wrapper config — không tham gia core GP search, B5)."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        valid = ~np.isnan(row)
        if valid.sum() == 0:
            continue
        total = float(np.sum(np.abs(row[valid])))
        if total == 0.0:
            continue
        out[t][valid] = row[valid] / total
    return out
