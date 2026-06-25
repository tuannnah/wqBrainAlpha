"""Test FitnessVector: deflated_sharpe haircut theo n_trials, from_metrics map đúng
AlphaMetrics + corr penalty + turnover band, hướng tối ưu nhất quán (max sharpe/min năm
tệ nhất, min mọi penalty)."""

from __future__ import annotations

import math

import pytest

from config.thresholds import TURNOVER_BAND
from src.backtest.metrics_local import AlphaMetrics
from src.gp.fitness_vec import FitnessVector, deflated_sharpe, from_metrics


def _metrics(**overrides) -> AlphaMetrics:
    base = dict(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2, 2022: 0.5}, weight_concentration=0.05,
    )
    base.update(overrides)
    return AlphaMetrics(**base)


def test_deflated_sharpe_no_haircut_for_single_trial():
    assert deflated_sharpe(1.5, n_trials=1) == pytest.approx(1.5)
    assert deflated_sharpe(1.5, n_trials=0) == pytest.approx(1.5)


def test_deflated_sharpe_haircut_grows_with_trials():
    d10 = deflated_sharpe(1.5, n_trials=10)
    d1000 = deflated_sharpe(1.5, n_trials=1000)
    assert d10 < 1.5
    assert d1000 < d10  # nhiều lần thử hơn -> haircut nặng hơn


def test_deflated_sharpe_matches_formula():
    sharpe, n = 2.0, 100
    expected = sharpe - math.sqrt(2 * math.log(n)) / math.sqrt(252)
    assert deflated_sharpe(sharpe, n) == pytest.approx(expected)


def test_from_metrics_per_year_min_sharpe_is_worst_year():
    fv = from_metrics(_metrics(), complexity=10, pool_corr=0.1, pop_corr=0.05, n_trials=1)
    assert fv.per_year_min_sharpe == pytest.approx(0.5)


def test_from_metrics_empty_per_year_gives_zero():
    fv = from_metrics(
        _metrics(per_year_sharpe={}), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.per_year_min_sharpe == pytest.approx(0.0)


def test_from_metrics_turnover_inside_band_has_zero_penalty():
    mid = (TURNOVER_BAND[0] + TURNOVER_BAND[1]) / 2
    fv = from_metrics(
        _metrics(turnover=mid), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.0)


def test_from_metrics_turnover_below_band_penalized_by_distance():
    too_low = TURNOVER_BAND[0] - 0.05
    fv = from_metrics(
        _metrics(turnover=too_low), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.05, abs=1e-9)


def test_from_metrics_turnover_above_band_penalized_by_distance():
    too_high = TURNOVER_BAND[1] + 0.10
    fv = from_metrics(
        _metrics(turnover=too_high), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1,
    )
    assert fv.turnover_penalty == pytest.approx(0.10, abs=1e-9)


def test_from_metrics_passes_through_corr_penalties_unchanged():
    fv = from_metrics(_metrics(), complexity=10, pool_corr=0.42, pop_corr=0.31, n_trials=1)
    assert fv.pool_corr_penalty == pytest.approx(0.42)
    assert fv.pop_corr_penalty == pytest.approx(0.31)


def test_from_metrics_complexity_penalty_scales_with_node_count():
    fv_small = from_metrics(_metrics(), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1)
    fv_large = from_metrics(_metrics(), complexity=100, pool_corr=0.0, pop_corr=0.0, n_trials=1)
    assert fv_large.complexity_penalty > fv_small.complexity_penalty


def test_fitness_vector_is_frozen_and_hashable():
    fv = from_metrics(_metrics(), complexity=10, pool_corr=0.0, pop_corr=0.0, n_trials=1)
    assert isinstance(fv, FitnessVector)
    with pytest.raises(AttributeError):
        fv.sharpe_deflated = 99.0  # type: ignore[misc]
