"""regression_neut/vector_neut — 2 op DUY NHẤT trong MiniBrain giảm self-correlation
(B5). Mỗi op hoạt động per-row (cross-sectional), chỉ trên cell in-universe."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="regression_neut", category=OpCategory.NEUTRALIZATION,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False,
          gp_usable=False)
def regression_neut(ctx: EvalContext, y: Panel, x: Panel) -> Panel:
    """Residual cross-sectional per-row của y hồi quy OLS (với intercept) theo x."""
    # Vectorize per-row (bỏ vòng t + lstsq): OLS 1 biến + intercept có nghiệm đóng
    # b=Σ(dx·dy)/Σ(dx²), residual=(y-ȳ)-b·(x-x̄). std(x)=0 -> b=0 -> residual=y-ȳ (khớp
    # nhánh gốc). <2 quan sát in-universe -> NaN.
    valid = ~np.isnan(y) & ~np.isnan(x)
    n = valid.sum(axis=1, keepdims=True)
    n_safe = np.where(n > 0, n, 1)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    mx = xv.sum(axis=1, keepdims=True) / n_safe
    my = yv.sum(axis=1, keepdims=True) / n_safe
    dx = np.where(valid, x - mx, 0.0)
    dy = np.where(valid, y - my, 0.0)
    sxx = (dx * dx).sum(axis=1, keepdims=True)
    sxy = (dx * dy).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(sxx > 0.0, sxy / sxx, 0.0)
    return np.where(valid & (n >= 2), dy - b * dx, np.nan)


@register(name="vector_neut", category=OpCategory.NEUTRALIZATION,
          signature=(ArgKind.PANEL, ArgKind.PANEL), bounded=False, commutative=False,
          gp_usable=False)
def vector_neut(ctx: EvalContext, x: Panel, y: Panel) -> Panel:
    """Trừ phần chiếu của x lên y mỗi hàng: x - (x.y / y.y) * y, chỉ trên in-universe."""
    # Vectorize per-row (bỏ vòng t): proj_coef = Σ(x·y)/Σ(y·y) trên in-universe mỗi hàng;
    # denom=0 -> giữ nguyên x; ô ngoài in-universe -> NaN.
    valid = ~np.isnan(x) & ~np.isnan(y)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    denom = (yv * yv).sum(axis=1, keepdims=True)
    num = (xv * yv).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        proj = np.where(denom > 0.0, num / denom, 0.0)  # denom=0 -> proj=0 -> out=x
    return np.where(valid, x - proj * y, np.nan)
