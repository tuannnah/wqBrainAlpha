"""AlphaMetrics + MetricsCalculator — đo lường BacktestResult (B8 master spec).

fitness dùng TURNOVER_FLOOR từ config/thresholds.py (Gap #7/R9: ngưỡng chỉ ở MỘT nơi,
không hardcode ở call site). annual_return dùng annualized simple return (KHÔNG CAGR —
đúng sửa Gap #7). per_year_sharpe (Task 4.2) dùng data.years() — regime robustness
first-class, không phải số phụ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.thresholds import TURNOVER_FLOOR
from src.backtest.backtester import BacktestResult
from src.data.market_panel import MarketData


@dataclass(frozen=True, slots=True)
class AlphaMetrics:
    sharpe: float
    annual_return: float
    turnover: float
    max_drawdown: float
    fitness: float
    per_year_sharpe: dict[int, float]
    weight_concentration: float


class MetricsCalculator:
    """Tính AlphaMetrics từ BacktestResult + MarketData. Stateless, an toàn dùng lại."""

    PERIODS_PER_YEAR: int = 252

    def compute(self, bt: BacktestResult, data: MarketData) -> AlphaMetrics:
        sharpe = self._sharpe(bt.daily_pnl)
        annual_return = self._annual_return(bt.daily_pnl)
        turnover = self._turnover(bt.weights)
        max_drawdown = self._max_drawdown(bt.equity_curve)
        fitness = sharpe * np.sqrt(abs(annual_return) / max(turnover, TURNOVER_FLOOR))
        per_year_sharpe = self._per_year_sharpe(bt.daily_pnl, data)
        weight_concentration = self._weight_concentration(bt.weights)
        return AlphaMetrics(
            sharpe=sharpe, annual_return=annual_return, turnover=turnover,
            max_drawdown=max_drawdown, fitness=float(fitness),
            per_year_sharpe=per_year_sharpe, weight_concentration=weight_concentration,
        )

    def _sharpe(self, daily_pnl: np.ndarray) -> float:
        valid = daily_pnl[np.isfinite(daily_pnl)]
        if valid.size < 2:
            return 0.0
        std = valid.std(ddof=0)
        if std == 0.0:
            return 0.0
        return float(valid.mean() / std * np.sqrt(self.PERIODS_PER_YEAR))

    def _annual_return(self, daily_pnl: np.ndarray) -> float:
        valid = daily_pnl[np.isfinite(daily_pnl)]
        if valid.size == 0:
            return 0.0
        return float(valid.mean() * self.PERIODS_PER_YEAR)

    def _turnover(self, weights: np.ndarray) -> float:
        if weights.shape[0] < 2:
            return 0.0
        prev = weights[:-1]
        curr = weights[1:]
        diff = np.abs(curr - prev)
        both_nan = np.isnan(prev) & np.isnan(curr)
        diff = np.where(both_nan, 0.0, diff)
        with np.errstate(invalid="ignore"):
            per_day = np.nansum(diff, axis=1)
        valid_rows = ~np.all(np.isnan(prev) | np.isnan(curr), axis=1)
        if not valid_rows.any():
            return 0.0
        return float(per_day[valid_rows].mean())

    def _max_drawdown(self, equity_curve: np.ndarray) -> float:
        if equity_curve.size == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        return float(np.max(drawdown))

    def _per_year_sharpe(self, daily_pnl: np.ndarray, data: MarketData) -> dict[int, float]:
        out: dict[int, float] = {}
        for year, sl in data.years().items():
            out[year] = self._sharpe(daily_pnl[sl])
        return out

    def _weight_concentration(self, weights: np.ndarray) -> float:
        if weights.size == 0:
            return 0.0
        gross = np.nansum(np.abs(weights), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            max_abs = np.nanmax(np.abs(weights), axis=1)
            share = np.where(gross > 0, max_abs / gross, 0.0)
        finite_share = share[np.isfinite(share)]
        if finite_share.size == 0:
            return 0.0
        return float(np.max(finite_share))
