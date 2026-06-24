"""MVP end-to-end: parse(alpha viết tay) -> eval -> build portfolio -> backtest ->
Sharpe sơ bộ + equity curve, trên dữ liệu thật (fixture small_panel).

Đây là MILESTONE MVP của toàn dự án MiniBrain (Part E master spec): chứng minh đường ống
parse->eval->backtest chạy thông trên một alpha thật, không mock bất cứ thành phần nào.

Lưu ý field: fixture `small_panel` (Phase 0, tests/conftest.py) chỉ có
fields={"close", "volume"} — KHÔNG có "open". Alpha mẫu gốc của plan
(`rank(ts_mean(divide(subtract(close, open), open), 5))`) không parse/eval được trên
small_panel vì thiếu field "open". Đã đổi sang biểu thức tương đương dùng return 1 ngày
(ts_delay(close, 1)) làm "open" giả lập — biểu thức cuối dùng:

    rank(ts_mean(divide(subtract(close, ts_delay(close, 1)), ts_delay(close, 1)), 5))

Biểu thức này đã được verify parse+eval thành công thật trên small_panel (không toàn-NaN,
chỉ NaN ở vài ngày đầu do ts_delay/ts_mean chưa đủ lookback — đúng hành vi window operator).
"""

from __future__ import annotations

import numpy as np

import src.operators_local  # noqa: F401  side-effect: đăng ký 28 operator thật vào registry
from src.backtest.backtester import Backtester
from src.backtest.config import Neutralization, PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import parse
from src.lang.registry import default_registry


def _rough_sharpe(daily_pnl: np.ndarray) -> float:
    valid = daily_pnl[~np.isnan(daily_pnl)]
    if valid.std(ddof=0) == 0 or valid.size < 2:
        return 0.0
    return float(valid.mean() / valid.std(ddof=0) * np.sqrt(252))


def test_handwritten_alpha_runs_end_to_end_and_produces_equity_curve(
    small_panel: MarketData,
) -> None:
    expr = "rank(ts_mean(divide(subtract(close, ts_delay(close, 1)), ts_delay(close, 1)), 5))"
    node = parse(expr)
    ctx = EvalContext(data=small_panel, registry=default_registry(), cache=None)
    signal = Evaluator(ctx).evaluate(node)
    assert signal.shape == (len(small_panel.dates), len(small_panel.assets))

    cfg = PortfolioConfig(neutralization=Neutralization.SECTOR, decay=0,
                          truncation=0.10, scale_book=1.0, delay=1)
    weights = PortfolioBuilder().build(signal, cfg, small_panel)
    result = Backtester().run(weights, small_panel)

    assert result.equity_curve.shape == (len(small_panel.dates),)
    assert not np.isnan(result.equity_curve).any()
    sharpe = _rough_sharpe(result.daily_pnl)
    assert np.isfinite(sharpe)
    print(f"[MVP demo] equity_curve[-1]={result.equity_curve[-1]:.4f} sharpe~{sharpe:.3f}")
