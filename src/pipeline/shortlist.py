"""Rank + decorrelate candidate đã có metrics/pnl → short-list cuối để sim Brain.

Đây là bước tổng hợp của pipeline: nhận candidate đã qua score_one, xếp hạng theo fitness,
rồi loại tuần tự candidate tương quan PnL quá cao với cái ĐÃ CHỌN (decorrelate nội bộ) VÀ với
pool đã pass (decorrelate pool-aware) — đúng nguyên tắc B9: PnL self-corr là nguyên nhân
reject hàng đầu, không phải AST-hash dedup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.backtest.metrics_local import AlphaMetrics
from src.backtest.pool_corr import PoolCorrelation
from src.local_types import Dates


@dataclass(frozen=True, slots=True)
class ShortlistCandidate:
    """Một candidate đã backtest xong: expr + metrics + PnL hằng ngày (để tính tương quan)."""

    expr: str
    metrics: AlphaMetrics
    pnl: npt.NDArray[np.float64]
    dates: Dates


def _pairwise_abs_rho(
    pnl_a: npt.NDArray[np.float64], dates_a: Dates,
    pnl_b: npt.NDArray[np.float64], dates_b: Dates,
) -> float | None:
    """Pearson |rho| trên giao ngày chung; None nếu thiếu điểm/phương sai bằng 0 (giống
    PoolCorrelation._pairwise_rho Phase 6) — KHÔNG bịa rho=0 giả."""
    # BẮT BUỘC sort theo dates trước searchsorted: giữ ngang hàng với
    # PoolCorrelation._pairwise_rho Phase 6 (pool_corr.py:68-73).
    ord_a = np.argsort(dates_a)
    dates_a = dates_a[ord_a]
    pnl_a = pnl_a[ord_a]
    ord_b = np.argsort(dates_b)
    dates_b = dates_b[ord_b]
    pnl_b = pnl_b[ord_b]
    common = np.intersect1d(dates_a, dates_b)
    if common.size < 2:
        return None
    idx_a = np.searchsorted(dates_a, common)
    idx_b = np.searchsorted(dates_b, common)
    x = pnl_a[idx_a]
    y = pnl_b[idx_b]
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return None
    x = x[finite]
    y = y[finite]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    rho = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(rho):
        return None
    return abs(rho)


def build_shortlist(
    candidates: list[ShortlistCandidate],
    top_k: int,
    max_corr: float,
    pool_corr: PoolCorrelation | None = None,
) -> list[ShortlistCandidate]:
    """Xếp hạng `candidates` theo `metrics.fitness` giảm dần, rồi quét tuần tự: giữ candidate
    nếu max|rho| với MỌI candidate đã giữ trước đó VÀ với pool (qua `pool_corr.max_corr` nếu
    có) đều < `max_corr`. Dừng khi đủ `top_k` hoặc hết. Không sửa đổi danh sách đầu vào."""
    ranked = sorted(candidates, key=lambda c: c.metrics.fitness, reverse=True)
    kept: list[ShortlistCandidate] = []
    for cand in ranked:
        if len(kept) >= top_k:
            break
        if pool_corr is not None:
            pool_rho, _worst = pool_corr.max_corr(cand.pnl, cand.dates)
            if pool_rho >= max_corr:
                continue
        too_correlated = False
        for chosen in kept:
            rho = _pairwise_abs_rho(cand.pnl, cand.dates, chosen.pnl, chosen.dates)
            if rho is not None and rho >= max_corr:
                too_correlated = True
                break
        if not too_correlated:
            kept.append(cand)
    return kept
