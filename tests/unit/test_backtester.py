"""Test Backtester.run: weights (đã delay) + returns -> daily_pnl + equity_curve."""

from __future__ import annotations

import numpy as np

from src.backtest.backtester import Backtester, BacktestResult
from src.data.market_panel import MarketData


def _panel_with_known_returns() -> MarketData:
    t, n = 3, 2
    dates = (np.datetime64("2021-01-01") + np.arange(t)).astype("datetime64[D]")
    assets = np.array(["A", "B"], dtype=np.str_)
    universe = np.ones((t, n), dtype=bool)
    returns = np.array([[0.01, -0.02], [0.02, 0.01], [-0.01, 0.0]])
    groups = {"sector": np.zeros((t, n), dtype=np.int64)}
    return MarketData(dates=dates, assets=assets, fields={}, universe=universe,
                      returns=returns, groups=groups)


def test_pnl_is_dot_product_of_weights_and_returns_same_row():
    data = _panel_with_known_returns()
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    result = Backtester().run(weights, data)
    expected_pnl = np.array([
        0.5 * 0.01 + (-0.5) * (-0.02),
        0.5 * 0.02 + (-0.5) * 0.01,
        0.5 * (-0.01) + (-0.5) * 0.0,
    ])
    np.testing.assert_allclose(result.daily_pnl, expected_pnl)


def test_equity_curve_is_cumsum_of_pnl():
    data = _panel_with_known_returns()
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    result = Backtester().run(weights, data)
    np.testing.assert_allclose(result.equity_curve, np.cumsum(result.daily_pnl))


def test_first_row_nan_weights_from_delay_give_zero_pnl_not_nan_propagated_to_equity():
    data = _panel_with_known_returns()
    weights = np.array([[np.nan, np.nan], [0.5, -0.5], [0.5, -0.5]])
    result = Backtester().run(weights, data)
    assert result.daily_pnl[0] == 0.0  # nansum trên toàn-NaN row -> 0.0, không NaN
    assert not np.isnan(result.equity_curve).any()


def test_result_stores_weights_passed_in():
    data = _panel_with_known_returns()
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    result = Backtester().run(weights, data)
    np.testing.assert_allclose(result.weights, weights)


def test_backtest_result_is_frozen():
    data = _panel_with_known_returns()
    weights = np.zeros((3, 2))
    result = Backtester().run(weights, data)
    import pytest
    with pytest.raises(AttributeError):
        result.daily_pnl = np.zeros(3)  # type: ignore[misc]


def test_out_of_universe_cells_excluded_from_pnl():
    data = _panel_with_known_returns()
    universe = data.universe.copy()
    universe[1, 0] = False  # mã A ngày 2 ngoài universe
    data2 = MarketData(dates=data.dates, assets=data.assets, fields=data.fields,
                        universe=universe, returns=data.returns, groups=data.groups)
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
    # weights[1,0] phải bị mask NaN bởi caller (PortfolioBuilder) trước khi tới Backtester;
    # Backtester tự mask lại theo universe để an toàn dù caller quên.
    result = Backtester().run(weights, data2)
    expected_day1 = -0.5 * 0.01  # chỉ còn cạnh B (asset A bị loại khỏi universe)
    assert np.isclose(result.daily_pnl[1], expected_day1)
