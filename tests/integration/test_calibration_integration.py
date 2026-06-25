"""Integration Phase 4.5: make_local_scorer thật (parse->eval->portfolio->backtest->metrics)
trên fixture small_panel, nối vào CalibrationHarness end-to-end.

Smoke test: chứng minh harness chạy thông với scorer THẬT trên dữ liệu thật, KHÔNG mock.
Không assert giá trị Spearman cụ thể (3 alpha viết tay với brain_sharpe bịa chỉ để chứng minh
đường ống) — chỉ assert đúng shape/loại/hữu hạn của CalibrationReport.
"""

from __future__ import annotations

import math

from src.calibration.harness import CalibrationHarness, LocalScore, make_local_scorer
from src.calibration.loader import BrainRecord
from src.calibration.report import CalibrationReport
from src.data.market_panel import MarketData


def _records() -> list[BrainRecord]:
    # 3 expr parse được trên small_panel (fields close/volume); brain_sharpe bịa để chạy ống.
    return [
        BrainRecord("rank(close)", brain_sharpe=1.2, brain_fitness=0.9,
                    brain_turnover=0.3, brain_self_corr=None),
        BrainRecord("ts_mean(close, 5)", brain_sharpe=-0.4, brain_fitness=-0.2,
                    brain_turnover=0.2, brain_self_corr=None),
        BrainRecord("rank(divide(close, volume))", brain_sharpe=0.6, brain_fitness=0.5,
                    brain_turnover=0.5, brain_self_corr=None),
    ]


def test_make_local_scorer_scores_real_expression(small_panel: MarketData) -> None:
    scorer = make_local_scorer(small_panel)
    score = scorer("rank(close)")
    assert isinstance(score, LocalScore)
    assert math.isfinite(score.sharpe)
    assert math.isfinite(score.fitness)


def test_invalid_field_expression_returns_none(small_panel: MarketData) -> None:
    scorer = make_local_scorer(small_panel)
    # field 'open' không có trong small_panel -> eval lỗi -> None (loại khỏi mẫu, không raise)
    assert scorer("rank(open)") is None


def test_unparseable_expression_returns_none(small_panel: MarketData) -> None:
    scorer = make_local_scorer(small_panel)
    assert scorer("rank(") is None


def test_returns_is_queryable_as_field(small_panel: MarketData) -> None:
    # `returns` là field WQ hợp lệ; MarketData lưu nó ở .returns (không trong .fields).
    # make_local_scorer phải expose nó để expr tham chiếu `returns` re-score được (không None).
    scorer = make_local_scorer(small_panel)
    score = scorer("rank(returns)")
    assert score is not None
    assert math.isfinite(score.sharpe)


def test_harness_end_to_end_with_real_scorer(small_panel: MarketData) -> None:
    harness = CalibrationHarness(scorer=make_local_scorer(small_panel))
    report = harness.run(_records())
    assert isinstance(report, CalibrationReport)
    assert report.n == 3  # cả 3 expr re-score được local
    # n=3 -> spearman tính được (không NaN do thiếu điểm); giá trị nằm trong [-1, 1]
    assert math.isfinite(report.spearman_sharpe)
    assert -1.0 <= report.spearman_sharpe <= 1.0
    assert math.isfinite(report.decile_hit_rate)
    # không alpha nào có brain_self_corr -> self_corr_agreement = NaN
    assert math.isnan(report.self_corr_agreement)
