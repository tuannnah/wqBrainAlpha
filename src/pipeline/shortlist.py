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
from src.backtest.pool_corr import PoolCorrelation, pairwise_abs_rho
from src.local_types import Dates


@dataclass(frozen=True, slots=True)
class ShortlistCandidate:
    """Một candidate đã backtest xong: expr + metrics + PnL hằng ngày (để tính tương quan)."""

    expr: str
    metrics: AlphaMetrics
    pnl: npt.NDArray[np.float64]
    dates: Dates


# Tương quan PnL đôi một dùng chung với combiner/pool_corr (một chỗ duy nhất).
_pairwise_abs_rho = pairwise_abs_rho


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
