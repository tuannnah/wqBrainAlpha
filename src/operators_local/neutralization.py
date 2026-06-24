"""regression_neut/vector_neut — 2 op DUY NHẤT trong MiniBrain giảm self-correlation
(B5). Mỗi op hoạt động per-row (cross-sectional), chỉ trên cell in-universe."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="regression_neut", category=OpCategory.NEUTRALIZATION,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False)
def regression_neut(ctx: EvalContext, y: Panel, x: Panel) -> Panel:
    """Residual cross-sectional per-row của y hồi quy OLS (với intercept) theo x."""
    out = np.full_like(y, np.nan, dtype=np.float64)
    for t in range(y.shape[0]):
        yr, xr = y[t], x[t]
        valid = ~np.isnan(yr) & ~np.isnan(xr)
        n_valid = int(valid.sum())
        if n_valid < 2:
            continue
        xv, yv = xr[valid], yr[valid]
        if np.std(xv) == 0.0:
            out[t][valid] = yv - float(np.mean(yv))
            continue
        design = np.column_stack([np.ones(n_valid), xv])
        coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
        out[t][valid] = yv - design @ coef
    return out


@register(name="vector_neut", category=OpCategory.NEUTRALIZATION,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False)
def vector_neut(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    """Trừ phần chiếu của x lên y mỗi hàng: x - (x.y / y.y) * y, chỉ trên in-universe."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        xr, yr = x[t], y[t]
        valid = ~np.isnan(xr) & ~np.isnan(yr)
        if not np.any(valid):
            continue
        xv, yv = xr[valid], yr[valid]
        denom = float(np.dot(yv, yv))
        if denom == 0.0:
            out[t][valid] = xv
            continue
        proj_coef = float(np.dot(xv, yv)) / denom
        out[t][valid] = xv - proj_coef * yv
    return out
