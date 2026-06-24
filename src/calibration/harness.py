"""CalibrationHarness: re-score local từng BrainRecord, đo Spearman rho vs Brain (B10).

KHÔNG đốt sim mới — chỉ đọc lại expr đã có từ DB và gọi LocalScorer (injected) để tính lại
metric thuần local. `run()` không raise trên từng record lỗi; scorer trả None -> loại khỏi
mẫu (n giảm), không phình rho bằng cách loại các điểm dữ liệu khó.

`make_local_scorer` đóng gói pipeline local thật (parse -> eval -> portfolio -> backtest ->
metrics). Config dùng để re-score PHẢI khớp config Brain đã dùng để sim ground-truth thì
calibration mới hợp lệ — bộ ground-truth OHLCV của Phase 4.5 sim với
`neutralization=NONE, decay=0, truncation=0.0, delay=1`, nên scorer dùng đúng config đó
(không phải PortfolioConfig mặc định SECTOR/0.10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from config.thresholds import SELF_CORR_MAX
from src.calibration.loader import BrainRecord
from src.calibration.report import CalibrationReport
from src.calibration.stats import spearman

if TYPE_CHECKING:
    from src.data.market_panel import MarketData


@dataclass(frozen=True, slots=True)
class LocalScore:
    sharpe: float
    fitness: float
    self_corr: float | None
    per_year_sharpe: dict[int, float]


class LocalScorer(Protocol):
    def __call__(self, expr_string: str) -> LocalScore | None: ...


class CalibrationHarness:
    """Đo độ tin cậy của ranking local so với Brain trên tập alpha đã sim thật."""

    def __init__(self, scorer: LocalScorer) -> None:
        self._scorer = scorer

    def run(self, brain_records: list[BrainRecord]) -> CalibrationReport:
        local_sharpes: list[float] = []
        local_fitnesses: list[float] = []
        brain_sharpes: list[float] = []
        brain_fitnesses: list[float] = []
        corr_pairs: list[tuple[float, float]] = []
        year_sums: dict[int, list[float]] = {}

        for record in brain_records:
            score = self._scorer(record.expr_string)
            if score is None:
                continue
            local_sharpes.append(score.sharpe)
            local_fitnesses.append(score.fitness)
            brain_sharpes.append(record.brain_sharpe if record.brain_sharpe is not None else math.nan)
            brain_fitnesses.append(
                record.brain_fitness if record.brain_fitness is not None else math.nan
            )
            if score.self_corr is not None and record.brain_self_corr is not None:
                corr_pairs.append((score.self_corr, record.brain_self_corr))
            for year, value in score.per_year_sharpe.items():
                year_sums.setdefault(year, []).append(value)

        n = len(local_sharpes)
        if n == 0:
            return CalibrationReport(
                n=0, spearman_sharpe=math.nan, spearman_fitness=math.nan,
                self_corr_agreement=math.nan, decile_hit_rate=math.nan, by_year={},
            )

        rho_sharpe = spearman(
            np.array(local_sharpes, dtype=np.float64), np.array(brain_sharpes, dtype=np.float64)
        )
        rho_fitness = spearman(
            np.array(local_fitnesses, dtype=np.float64), np.array(brain_fitnesses, dtype=np.float64)
        )

        if corr_pairs:
            agree = sum(
                1
                for local_c, brain_c in corr_pairs
                if (local_c < SELF_CORR_MAX) == (brain_c < SELF_CORR_MAX)
            )
            self_corr_agreement = agree / len(corr_pairs)
        else:
            self_corr_agreement = math.nan

        decile_hit_rate = _decile_hit_rate(local_sharpes, brain_sharpes)
        by_year = {year: float(np.mean(values)) for year, values in year_sums.items()}

        return CalibrationReport(
            n=n, spearman_sharpe=rho_sharpe, spearman_fitness=rho_fitness,
            self_corr_agreement=self_corr_agreement, decile_hit_rate=decile_hit_rate,
            by_year=by_year,
        )


def _decile_hit_rate(local: list[float], brain: list[float]) -> float:
    """Tỉ lệ giao nhau giữa top-decile-Brain và top-decile-local (cùng kích thước)."""
    n = len(local)
    k = max(1, math.ceil(n / 10))
    brain_top = set(sorted(range(n), key=lambda i: brain[i], reverse=True)[:k])
    local_top = set(sorted(range(n), key=lambda i: local[i], reverse=True)[:k])
    return len(brain_top & local_top) / k


def make_local_scorer(data: "MarketData") -> LocalScorer:
    """Đóng gói pipeline local thật (Phase 1-4) thành một LocalScorer cho calibration.

    Config khớp ground-truth OHLCV Phase 4.5 (`neutralization=NONE, decay=0, truncation=0.0,
    scale_book=1.0, delay=1`) — đây là điều kiện để re-score local SO SÁNH ĐƯỢC với Brain
    Sharpe đã lưu. `self_corr` local để None (cần pool PnL Phase 6 mới đo được, chưa có).
    expr không re-score được (parse lỗi / field thiếu / eval lỗi) -> None (loại khỏi mẫu).
    """
    import src.operators_local  # noqa: F401  side-effect: đăng ký operator vào registry
    from src.backtest.backtester import Backtester
    from src.backtest.config import Neutralization, PortfolioConfig
    from src.backtest.metrics_local import MetricsCalculator
    from src.backtest.portfolio import PortfolioBuilder
    from src.engine.evaluator import EvalContext, Evaluator
    from src.lang.parser import ParseError, parse
    from src.lang.registry import default_registry

    cfg = PortfolioConfig(
        neutralization=Neutralization.NONE, decay=0, truncation=0.0, scale_book=1.0, delay=1,
    )
    registry = default_registry()
    builder = PortfolioBuilder()
    backtester = Backtester()
    calculator = MetricsCalculator()

    def _score(expr_string: str) -> LocalScore | None:
        try:
            node = parse(expr_string)
        except ParseError:
            return None
        try:
            ctx = EvalContext(data=data, registry=registry, cache=None)
            signal = Evaluator(ctx).evaluate(node)
        except (KeyError, ValueError):
            return None
        if not np.isfinite(signal).any():
            return None
        weights = builder.build(signal, cfg, data)
        bt = backtester.run(weights, data)
        metrics = calculator.compute(bt, data)
        return LocalScore(
            sharpe=metrics.sharpe, fitness=metrics.fitness, self_corr=None,
            per_year_sharpe=metrics.per_year_sharpe,
        )

    return _score
