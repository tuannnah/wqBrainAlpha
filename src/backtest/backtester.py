"""Backtester: weights (đã delay bởi PortfolioBuilder) + returns -> daily PnL + equity.

Delay-1 KHÔNG được áp lại ở đây — `weights` truyền vào `run` là đầu ra của
`PortfolioBuilder.build` (đã dịch `cfg.delay` dòng). Công thức: pnl_t = nansum(w_t * ret_t)
theo trục asset, chỉ trên cell in-universe (an toàn double-mask dù caller đã mask).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.data.market_panel import MarketData
from src.local_types import Panel


@dataclass(frozen=True, slots=True)
class BacktestResult:
    daily_pnl: npt.NDArray[np.float64]  # (T,)
    equity_curve: npt.NDArray[np.float64]  # (T,)
    weights: Panel  # (T, N), đã delay


class Backtester:
    """Chạy backtest delay-1 (hoặc delay tuỳ ý đã áp sẵn trong `weights`)."""

    def run(self, weights: Panel, data: MarketData) -> BacktestResult:
        masked_weights = np.where(data.universe, weights, np.nan)
        contrib = masked_weights * data.returns
        with np.errstate(invalid="ignore"):
            daily_pnl = np.nansum(contrib, axis=1)
        # Ngày toàn-NaN (vd do delay ở đầu chuỗi) -> nansum trả 0.0 (đúng ngữ nghĩa numpy),
        # giữ nguyên — không có pnl phát sinh là hợp lý cho ngày chưa có weight.
        equity_curve = np.cumsum(daily_pnl)
        return BacktestResult(
            daily_pnl=daily_pnl, equity_curve=equity_curve, weights=weights,
        )
