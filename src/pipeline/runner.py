"""Lớp orchestration cuối của MiniBrain: score_one chấm 1 expr KHÔNG đốt sim; generate_many
drive GPEngine rồi rút short-list. Network-agnostic — nhận MarketData/GPEngine qua tham số
injected, test được bằng fake hoàn toàn. KHÔNG import src.llm/src.generation (dependency rule
B1); KHÔNG import cứng src.gp (generate_many dùng Protocol structural)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.backtest.backtester import Backtester
from src.backtest.config import PortfolioConfig
from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.metrics_local import AlphaMetrics, MetricsCalculator
from src.backtest.pool_corr import PoolCorrelation
from src.backtest.portfolio import PortfolioBuilder
from src.data.market_panel import MarketData
from src.engine.evaluator import EvalContext, Evaluator
from src.lang.parser import ParseError, parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, FieldCollector
from src.local_types import Dates

_EMPTY_METRICS = AlphaMetrics(
    sharpe=0.0, annual_return=0.0, turnover=0.0, max_drawdown=0.0,
    fitness=0.0, per_year_sharpe={}, weight_concentration=0.0,
)


@dataclass(frozen=True, slots=True)
class _ScoreResult:
    """Kết quả đầy đủ của một lần chấm: metrics + verdict + PnL/dates (PnL rỗng nếu fail
    trước khi backtest chạy). Dùng nội bộ để generate_many lấy PnL không phải backtest lại."""

    metrics: AlphaMetrics
    verdict: GateVerdict
    pnl: npt.NDArray[np.float64]
    dates: Dates


def _score_one_full(
    expr: str,
    cfg: PortfolioConfig,
    data: MarketData,
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
) -> _ScoreResult:
    """parse → eval → portfolio → backtest → metrics → pool_corr → gate. Thuần local, tất
    định với cùng (expr, cfg, data, pool). Lỗi parse/eval → metrics rỗng + verdict fail có lý
    do rõ ràng (KHÔNG silent, KHÔNG bịa metrics) và PnL rỗng."""
    empty_pnl: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    try:
        node = parse(expr)
    except ParseError as exc:
        return _ScoreResult(
            _EMPTY_METRICS,
            GateVerdict(passed=False, hard_failures=[f"parse lỗi: {exc}"]),
            empty_pnl, data.dates,
        )

    fields = FieldCollector().visit(node)
    fields_ok = bool(fields) and fields.issubset(data.field_names())
    depth = DepthVisitor().visit(node)

    ctx = EvalContext(data=data, registry=default_registry(), cache=None)
    try:
        signal = Evaluator(ctx).evaluate(node)
    except (KeyError, ValueError) as exc:
        return _ScoreResult(
            _EMPTY_METRICS,
            GateVerdict(passed=False, hard_failures=[f"eval lỗi: {exc}"]),
            empty_pnl, data.dates,
        )

    if bool(np.all(np.isnan(signal))):
        return _ScoreResult(
            _EMPTY_METRICS,
            GateVerdict(passed=False, hard_failures=["signal toàn NaN — không dùng được"]),
            empty_pnl, data.dates,
        )

    weights = PortfolioBuilder().build(signal, cfg, data)
    bt = Backtester().run(weights, data)
    metrics = MetricsCalculator().compute(bt, data)

    if pool:
        verdict = GateEvaluator().evaluate_with_pool(
            metrics, candidate_pnl=bt.daily_pnl, candidate_dates=data.dates,
            pool_corr=PoolCorrelation(pool=pool), depth=depth, fields_ok=fields_ok,
        )
    else:
        verdict = GateEvaluator().evaluate(
            metrics, self_corr=0.0, depth=depth, fields_ok=fields_ok,
        )
    return _ScoreResult(metrics, verdict, bt.daily_pnl, data.dates)


def score_one(
    expr: str,
    cfg: PortfolioConfig,
    data: MarketData,
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
) -> tuple[AlphaMetrics, GateVerdict]:
    """Chấm 1 expr local (không đốt sim). Trả (AlphaMetrics, GateVerdict). Xem
    `_score_one_full` cho ngữ nghĩa lỗi/pool đầy đủ."""
    res = _score_one_full(expr, cfg, data, pool)
    return res.metrics, res.verdict
