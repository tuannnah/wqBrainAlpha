"""trade_when/hump — conditioning lever (B5: trade_when là nguồn edge chính qua gating;
hump giảm turnover, không nên áp lên alpha có turnover nhanh là bản chất). Cả hai
carry-forward chỉ dùng giá trị tại rows <= t (no-look-ahead)."""

from __future__ import annotations

import numpy as np

from src.engine.evaluator import EvalContext
from src.lang.registry import ArgKind, OpCategory, register
from src.local_types import Panel


@register(name="trade_when", category=OpCategory.CONDITIONAL,
          signature=(ArgKind.PANEL, ArgKind.PANEL, ArgKind.PANEL), bounded=False,
          commutative=False)
def trade_when(ctx: EvalContext, trigger: Panel, alpha: Panel, exit_cond: Panel) -> Panel:
    out = np.full_like(alpha, np.nan, dtype=np.float64)
    last_valid = np.full(alpha.shape[1], np.nan, dtype=np.float64)
    for t in range(alpha.shape[0]):
        trig_t, exit_t, alpha_t = trigger[t], exit_cond[t], alpha[t]
        take_new = trig_t > 0
        carry = (~take_new) & (exit_t > 0)
        out[t][take_new] = alpha_t[take_new]
        out[t][carry] = last_valid[carry]
        # còn lại (không trigger, không carry) giữ NaN mặc định
        has_val = ~np.isnan(out[t])
        last_valid = np.where(has_val, out[t], last_valid)
    return out


@register(name="hump", category=OpCategory.CONDITIONAL,
          signature=(ArgKind.PANEL, ArgKind.SCALAR), bounded=False, commutative=False)
def hump(ctx: EvalContext, x: Panel, thr: float) -> Panel:
    # Bỏ vòng col (giữ vòng t vì carry-forward tuần tự): mỗi bước t cập nhật `last` cho
    # cả hàng cổ phiếu cùng lúc. Ô hiện tại NaN -> giữ NaN, last không đổi; ô có giá trị
    # -> cập nhật last khi vượt ngưỡng thr rồi ghi last.
    out = x.copy()
    last = np.full(x.shape[1], np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        for t in range(x.shape[0]):
            cur = x[t]
            valid = ~np.isnan(cur)
            trigger = valid & (np.isnan(last) | (np.abs(cur - last) >= thr))
            last = np.where(trigger, cur, last)
            out[t] = np.where(valid, last, out[t])
    return out
