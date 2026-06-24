"""Test AlphaMetrics + MetricsCalculator.compute: sharpe/annual_return/turnover/
max_drawdown/fitness trên BacktestResult biết trước, tính tay đối chiếu."""

from __future__ import annotations

import numpy as np
import pytest

from config.thresholds import TURNOVER_FLOOR
from src.backtest.backtester import BacktestResult
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.data.market_panel import MarketData


def _panel_3d_2n() -> MarketData:
    t, n = 3, 2
    dates = (np.datetime64("2021-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array(["A", "B"], dtype=np.str_)
    universe = np.ones((t, n), dtype=bool)
    returns = np.zeros((t, n))
    groups = {"sector": np.zeros((t, n), dtype=np.int64)}
    return MarketData(dates=dates, assets=assets, fields={}, universe=universe,
                      returns=returns, groups=groups)


def test_sharpe_matches_hand_calculation():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.01, -0.005, 0.02])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((3, 2)))
    m = MetricsCalculator().compute(bt, data)
    expected = daily_pnl.mean() / daily_pnl.std(ddof=0) * np.sqrt(252)
    assert isinstance(m, AlphaMetrics)
    assert np.isclose(m.sharpe, expected)


def test_sharpe_zero_when_std_is_zero():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.01, 0.01, 0.01])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((3, 2)))
    m = MetricsCalculator().compute(bt, data)
    assert m.sharpe == 0.0


def test_annual_return_is_mean_pnl_times_252():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.001, 0.002, 0.0])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((3, 2)))
    m = MetricsCalculator().compute(bt, data)
    assert np.isclose(m.annual_return, daily_pnl.mean() * 252)


def test_turnover_matches_hand_calculation():
    data = _panel_3d_2n()
    daily_pnl = np.zeros(3)
    weights = np.array([[0.5, -0.5], [0.3, -0.3], [0.5, -0.5]])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    # |w1-w0| = |0.3-0.5|+|-0.3+0.5| = 0.4 ; |w2-w1| = |0.5-0.3|+|-0.5+0.3| = 0.4
    expected_turnover = np.mean([0.4, 0.4])
    assert np.isclose(m.turnover, expected_turnover)


def test_turnover_counts_one_sided_nan_as_entering_universe():
    # Mã B NaN ở ngày 0 (chưa vào universe) -> có weight 0.3 ở ngày 1 (vừa vào).
    # Hàng [0]->[1] KHÔNG all-NaN-một-phía (mã A có weight cả 2 ngày) nên valid_rows
    # vẫn giữ hàng này; turnover ngày đó phải tính |0.3 - 0| = 0.3 cho mã B, không bị
    # np.nansum làm rớt.
    data = _panel_3d_2n()
    daily_pnl = np.zeros(3)
    weights = np.array([
        [0.5, np.nan],   # B chưa vào universe
        [0.5, 0.3],      # B vừa vào universe với weight 0.3
        [0.5, 0.3],      # giữ nguyên -> turnover ngày này = 0
    ])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    # day0->day1: |0.5-0.5| + |0.3-0| = 0.3 ; day1->day2: |0.5-0.5| + |0.3-0.3| = 0.0
    expected_turnover = np.mean([0.3, 0.0])
    assert np.isclose(m.turnover, expected_turnover)


def test_max_drawdown_matches_hand_calculation():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.10, -0.20, 0.05])  # equity: 0.10, -0.10, -0.05
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((3, 2)))
    m = MetricsCalculator().compute(bt, data)
    # running_max: 0.10, 0.10, 0.10 ; drawdown: 0, 0.20, 0.15 -> max = 0.20
    assert np.isclose(m.max_drawdown, 0.20)


def test_fitness_uses_turnover_floor_from_config_not_hardcoded():
    data = _panel_3d_2n()
    daily_pnl = np.array([0.01, 0.01, 0.01])
    weights = np.zeros((3, 2))  # turnover = 0.0 -> phải dùng floor
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    assert m.turnover == 0.0
    expected_fitness = m.sharpe * np.sqrt(abs(m.annual_return) / TURNOVER_FLOOR)
    assert np.isclose(m.fitness, expected_fitness)


def test_alpha_metrics_is_frozen():
    m = AlphaMetrics(sharpe=1.0, annual_return=0.1, turnover=0.1, max_drawdown=0.05,
                     fitness=1.0, per_year_sharpe={}, weight_concentration=0.1)
    with pytest.raises(AttributeError):
        m.sharpe = 2.0  # type: ignore[misc]


def _two_year_panel() -> MarketData:
    """4 ngày: 2 ngày năm 2021, 2 ngày năm 2022."""
    t, n = 4, 2
    dates = np.array(
        ["2021-12-30", "2021-12-31", "2022-01-01", "2022-01-02"], dtype="datetime64[D]"
    )
    assets = np.array(["A", "B"], dtype=np.str_)
    universe = np.ones((t, n), dtype=bool)
    returns = np.zeros((t, n))
    groups = {"sector": np.zeros((t, n), dtype=np.int64)}
    return MarketData(dates=dates, assets=assets, fields={}, universe=universe,
                      returns=returns, groups=groups)


def test_per_year_sharpe_splits_by_data_years():
    data = _two_year_panel()
    daily_pnl = np.array([0.01, 0.01, -0.02, -0.02])  # 2021 toàn lãi, 2022 toàn lỗ
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=np.zeros((4, 2)))
    m = MetricsCalculator().compute(bt, data)
    years = data.years()
    assert set(m.per_year_sharpe) == set(years)
    for year, sl in years.items():
        expected = MetricsCalculator()._sharpe(daily_pnl[sl])
        assert np.isclose(m.per_year_sharpe[year], expected)
    # 2021 toàn lãi đều -> std=0 -> sharpe=0.0 theo quy ước; 2022 cũng vậy
    assert m.per_year_sharpe[2021] == 0.0
    assert m.per_year_sharpe[2022] == 0.0


def test_weight_concentration_is_worst_day_max_name_share():
    data = _two_year_panel()
    daily_pnl = np.zeros(4)
    # ngày 0: cân bằng 50/50 ; ngày 2: mã A chiếm 90% book -> concentration phải bắt ngày này
    weights = np.array([
        [0.5, -0.5],
        [0.5, -0.5],
        [0.9, -0.1],
        [0.5, -0.5],
    ])
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    assert np.isclose(m.weight_concentration, 0.9)


def test_weight_concentration_zero_when_all_weights_nan():
    data = _two_year_panel()
    daily_pnl = np.zeros(4)
    weights = np.full((4, 2), np.nan)
    bt = BacktestResult(daily_pnl=daily_pnl, equity_curve=np.cumsum(daily_pnl),
                        weights=weights)
    m = MetricsCalculator().compute(bt, data)
    assert m.weight_concentration == 0.0
