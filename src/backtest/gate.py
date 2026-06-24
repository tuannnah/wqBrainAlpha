"""score_local_gate — cổng local BẮT BUỘC trước khi đốt sim Brain (D9, gỡ đường cũ).

Phase 4: gate giờ dùng MetricsCalculator (Task 4.1) + GateEvaluator (Task 4.3) thật —
xét đủ depth/fields_ok/self_corr/weight_concentration (hard gates) thay cho gate tối
thiểu Phase 3 (chỉ "parse được + eval không toàn-NaN + có pnl hữu hạn"). Đây là điểm DUY
NHẤT src/llm được phép import từ tầng backtest (dependency rule một chiều).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import src.operators_local  # noqa: F401  side-effect: đăng ký 28 operator thật vào registry
from src.backtest.backtester import Backtester
from src.backtest.config import PortfolioConfig
from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import MetricsCalculator
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import ParseError, parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, FieldCollector


@dataclass(frozen=True, slots=True)
class LocalGateVerdict:
    """Kết quả gate local: pass/fail kèm lý do (để ghi `record_failure`)."""

    passed: bool
    reason: str


def score_local_gate(
    expr: str, cfg: PortfolioConfig, data: MarketData, self_corr: float = 0.0,
) -> LocalGateVerdict:
    """Gate Phase 4: expr phải parse được, eval không toàn-NaN, backtest ra được ít nhất
    1 ngày pnl hữu hạn, RỒI áp GateEvaluator thật (depth/fields_ok/self_corr/weight_
    concentration hard gates) trên AlphaMetrics tính bởi MetricsCalculator.

    `self_corr` mặc định 0.0 vì `RefinementLoop` (3 vị trí gọi `local_gate_fn`) chưa
    truyền tham số này — `PoolCorrelation` (so sánh với pool alpha đã có) chỉ xuất hiện ở
    Phase 6, nên tạm coi self_corr = 0.0 (không chặn) để giữ tương thích ngược với loop."""
    try:
        node = parse(expr)
    except ParseError as exc:
        return LocalGateVerdict(False, f"parse lỗi: {exc}")

    fields = FieldCollector().visit(node)
    fields_ok = fields.issubset(set(data.fields.keys()))
    depth = DepthVisitor().visit(node)

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

    metrics = MetricsCalculator().compute(result, data)
    verdict = GateEvaluator().evaluate(
        metrics, self_corr=self_corr, depth=depth, fields_ok=fields_ok,
    )
    if not verdict.passed:
        return LocalGateVerdict(False, f"gate hard fail: {'; '.join(verdict.hard_failures)}")
    return LocalGateVerdict(True, "ok")
