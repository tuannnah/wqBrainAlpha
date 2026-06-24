"""score_local_gate — cổng local BẮT BUỘC trước khi đốt sim Brain (D9, gỡ đường cũ).

Phase 3 MVP: gate tối thiểu — expr phải parse được, eval ra signal không toàn-NaN, và
backtest sinh được ít nhất 1 ngày pnl hữu hạn. KHÔNG còn đủ Sharpe/turnover/concentration —
đó là việc của Phase 4 (MetricsCalculator + GateEvaluator), sẽ mở rộng hàm này khi có. Đây
là điểm DUY NHẤT src/llm được phép import từ tầng backtest (dependency rule một chiều).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import src.operators_local  # noqa: F401  side-effect: đăng ký 28 operator thật vào registry
from src.backtest.backtester import Backtester
from src.backtest.config import PortfolioConfig
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import ParseError, parse
from src.lang.registry import default_registry


@dataclass(frozen=True, slots=True)
class LocalGateVerdict:
    """Kết quả gate local: pass/fail kèm lý do (để ghi `record_failure`)."""

    passed: bool
    reason: str


def score_local_gate(expr: str, cfg: PortfolioConfig, data: MarketData) -> LocalGateVerdict:
    """Gate tối thiểu Phase 3: expr phải parse được, eval không toàn-NaN, backtest ra
    được ít nhất 1 ngày pnl hữu hạn. Đây là gate "evaluable" hẹp — chưa xét Sharpe/
    turnover/concentration (Phase 4 sẽ mở rộng bằng MetricsCalculator + GateEvaluator)."""
    try:
        node = parse(expr)
    except ParseError as exc:
        return LocalGateVerdict(False, f"parse lỗi: {exc}")

    ctx = EvalContext(data=data, registry=default_registry(), cache=None)
    try:
        signal = Evaluator(ctx).evaluate(node)
    except (KeyError, ValueError) as exc:
        return LocalGateVerdict(False, f"eval lỗi: {exc}")

    if np.all(np.isnan(signal)):
        return LocalGateVerdict(False, "signal toàn NaN — không có giá trị dùng được")

    weights = PortfolioBuilder().build(signal, cfg, data)
    result = Backtester().run(weights, data)
    if not np.isfinite(result.daily_pnl).any():
        return LocalGateVerdict(False, "không sinh được pnl hữu hạn")

    return LocalGateVerdict(True, "ok")
