"""Lớp orchestration cuối của MiniBrain: score_one chấm 1 expr KHÔNG đốt sim; generate_many
drive GPEngine rồi rút short-list. Network-agnostic — nhận MarketData/GPEngine qua tham số
injected, test được bằng fake hoàn toàn. KHÔNG import src.llm/src.generation (dependency rule
B1); KHÔNG import cứng src.gp (generate_many dùng Protocol structural)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
from src.lang.visitors import DepthVisitor, FieldCollector, Serializer
from src.local_types import Dates
from src.pipeline.shortlist import ShortlistCandidate, build_shortlist

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


class _GPIndividualLike(Protocol):
    expr: object
    fitness: object | None


class _GPRunResultLike(Protocol):
    final_population: list[_GPIndividualLike]


class _RunsGP(Protocol):
    def run(self) -> _GPRunResultLike: ...


def generate_many(
    gp_engine: _RunsGP,
    cfg: PortfolioConfig,
    data: MarketData,
    top_k: int,
    max_corr: float,
    pool: dict[int, tuple[Dates, npt.NDArray[np.float64]]] | None = None,
) -> list[ShortlistCandidate]:
    """Chạy `gp_engine.run()` → final_population; với mỗi Individual đã eval (fitness không
    None), serialize AST → string, chấm lại qua `_score_one_full` (một nguồn AlphaMetrics +
    PnL duy nhất, KHÔNG backtest 2 lần), giữ cái pass gate, rồi `build_shortlist` top_k +
    decorrelate pool-aware. Individual fitness=None (chưa eval trong GP) bị bỏ qua."""
    result = gp_engine.run()
    serializer = Serializer()
    pool_corr = PoolCorrelation(pool=pool) if pool else None

    candidates: list[ShortlistCandidate] = []
    seen: set[str] = set()
    for ind in result.final_population:
        if ind.fitness is None:
            continue
        expr_str = serializer.visit(ind.expr)  # type: ignore[arg-type]
        if expr_str in seen:
            continue
        seen.add(expr_str)
        res = _score_one_full(expr_str, cfg, data, pool)
        if not res.verdict.passed:
            continue
        candidates.append(
            ShortlistCandidate(expr=expr_str, metrics=res.metrics, pnl=res.pnl, dates=res.dates)
        )
    return build_shortlist(candidates, top_k=top_k, max_corr=max_corr, pool_corr=pool_corr)
