"""Test đỏ->xanh cho sub_universe_ok (Task 4): proxy robustness sub-universe local — alpha
phải giữ Sharpe khi giới hạn về nhóm mã thanh khoản nhất (xấp xỉ sub-universe test của Brain)."""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.sub_universe import sub_universe_ok
from src.data.market_panel import MarketData
from src.lang.parser import parse
from src.lang.registry import default_registry


def _panel(t=80, n=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.arange("2020-01-01", "2021-06-01", dtype="datetime64[D]")[:t].astype("datetime64[ns]")
    close = 100 + np.cumsum(rng.normal(0, 1, (t, n)), axis=0)
    fields = {k: close.copy() for k in ("close", "open", "high", "low", "vwap")}
    fields["volume"] = np.abs(rng.normal(1e6, 2e5, (t, n)))
    return MarketData(
        dates=dates, assets=np.array([f"S{i}" for i in range(n)]), fields=fields,
        universe=np.ones((t, n), dtype=bool),
        returns=np.vstack([np.zeros((1, n)), np.diff(close, axis=0) / close[:-1]]),
        groups={"sector": (np.arange(n) % 3).reshape(1, n).repeat(t, axis=0)},
    )


def test_sub_universe_ok_tra_bool_khong_sap():
    data = _panel()
    node = parse("rank(ts_delta(close, 5))")
    cfg = PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)
    out = sub_universe_ok(node, cfg, data, default_registry(), full_sharpe=1.0, frac=0.5)
    assert isinstance(out, bool)


def test_sub_universe_full_sharpe_khong_duong_thi_pass():
    data = _panel()
    node = parse("rank(ts_delta(close, 5))")
    cfg = PortfolioConfig(neutralization=Neutralization.MARKET, decay=4, truncation=0.08)
    assert sub_universe_ok(node, cfg, data, default_registry(), full_sharpe=-0.5, frac=0.5) is True
