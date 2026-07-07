"""PortfolioBuilder KHÔNG phun RuntimeWarning khi neutralize MARKET/group trên panel có hàng
toàn-NaN (prefix warm-up của decay/lookback) hoặc inf. MARKET giờ là neutralization mặc định
của closed-loop nên đây là đường nóng — log phải sạch (không 'Mean of empty slice'/'invalid')."""

from __future__ import annotations

import warnings

import numpy as np

from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData


def _panel_with_nan_prefix(t=20, n=6) -> MarketData:
    rng = np.random.default_rng(0)
    dates = np.arange("2020-01-01", "2020-02-01", dtype="datetime64[D]")[:t].astype("datetime64[ns]")
    close = 100 + np.cumsum(rng.normal(0, 1, (t, n)), axis=0)
    fields = {k: close.copy() for k in ("close", "open", "high", "low", "vwap")}
    fields["volume"] = np.abs(rng.normal(1e6, 1e5, (t, n)))
    return MarketData(
        dates=dates, assets=np.array([f"S{i}" for i in range(n)]), fields=fields,
        universe=np.ones((t, n), dtype=bool),
        returns=np.vstack([np.zeros((1, n)), np.diff(close, axis=0) / close[:-1]]),
        groups={"sector": (np.arange(n) % 2).reshape(1, n).repeat(t, axis=0)},
    )


def _runtime_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all="warn"):
            fn()
    return [w for w in caught if issubclass(w.category, RuntimeWarning)]


def _signal_with_nan_prefix(data: MarketData) -> np.ndarray:
    sig = np.array(data.field("close"), dtype=np.float64)
    sig[:3, :] = np.nan          # hàng toàn-NaN (warm-up)
    sig[5, 0] = np.inf           # 1 ô inf để thử nhánh subtract
    return sig


def test_neutralize_market_khong_phun_warning():
    data = _panel_with_nan_prefix()
    sig = _signal_with_nan_prefix(data)
    cfg = PortfolioConfig(neutralization=Neutralization.MARKET, decay=0, truncation=0.08)
    ws = _runtime_warnings(lambda: PortfolioBuilder().build(sig, cfg, data))
    assert not ws, f"MARKET neutralize phun RuntimeWarning: {[str(w.message) for w in ws]}"


def test_neutralize_sector_khong_phun_warning():
    data = _panel_with_nan_prefix()
    sig = _signal_with_nan_prefix(data)
    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR, decay=0, truncation=0.08)
    ws = _runtime_warnings(lambda: PortfolioBuilder().build(sig, cfg, data))
    assert not ws, f"SECTOR neutralize phun RuntimeWarning: {[str(w.message) for w in ws]}"
