# tests/unit/test_gates.py
"""Test GateVerdict + GateEvaluator: hard gates (depth/fields/self_corr/concentration)
tách bạch khỏi soft scores (sharpe/fitness/turnover_band/per_year_min)."""

from __future__ import annotations

from config.thresholds import MAX_DEPTH, SELF_CORR_MAX, TURNOVER_BAND, WEIGHT_CONCENTRATION_CAP
from src.backtest.gates import GateEvaluator, GateVerdict
from src.backtest.metrics_local import AlphaMetrics


def _good_metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2, 2022: 0.8},
        weight_concentration=WEIGHT_CONCENTRATION_CAP / 2,
    )


def test_all_pass_when_within_every_hard_limit():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert isinstance(verdict, GateVerdict)
    assert verdict.passed is True
    assert verdict.hard_failures == []


def test_depth_over_cap_is_hard_failure():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=MAX_DEPTH + 1, fields_ok=True)
    assert verdict.passed is False
    assert any("depth" in f for f in verdict.hard_failures)


def test_fields_not_ok_is_hard_failure():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=False)
    assert verdict.passed is False
    assert any("fields_ok" in f for f in verdict.hard_failures)


def test_self_corr_at_or_above_max_is_hard_failure():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=SELF_CORR_MAX, depth=3, fields_ok=True)
    assert verdict.passed is False
    assert any("self_corr" in f for f in verdict.hard_failures)


def test_self_corr_just_below_max_passes_that_gate():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(
        m, self_corr=SELF_CORR_MAX - 0.01, depth=3, fields_ok=True
    )
    assert not any("self_corr" in f for f in verdict.hard_failures)


def test_weight_concentration_over_cap_is_hard_failure():
    m = AlphaMetrics(
        sharpe=1.5, annual_return=0.2, turnover=0.3, max_drawdown=0.1, fitness=2.0,
        per_year_sharpe={}, weight_concentration=WEIGHT_CONCENTRATION_CAP + 0.01,
    )
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.passed is False
    assert any("weight_concentration" in f for f in verdict.hard_failures)


def test_multiple_hard_failures_all_recorded():
    m = AlphaMetrics(
        sharpe=0.0, annual_return=0.0, turnover=0.0, max_drawdown=0.0, fitness=0.0,
        per_year_sharpe={}, weight_concentration=1.0,
    )
    verdict = GateEvaluator().evaluate(
        m, self_corr=0.99, depth=MAX_DEPTH + 5, fields_ok=False
    )
    assert verdict.passed is False
    assert len(verdict.hard_failures) == 4  # depth + fields + self_corr + concentration


def test_soft_scores_contain_sharpe_fitness_turnover_band_per_year_min():
    m = _good_metrics()
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.soft_scores["sharpe"] == m.sharpe
    assert verdict.soft_scores["fitness"] == m.fitness
    assert verdict.soft_scores["turnover_band"] == 1.0  # 0.30 trong TURNOVER_BAND
    assert verdict.soft_scores["per_year_min"] == min(m.per_year_sharpe.values())


def test_turnover_band_score_negative_when_below_floor():
    m = AlphaMetrics(
        sharpe=1.0, annual_return=0.1, turnover=TURNOVER_BAND[0] - 0.005, max_drawdown=0.1,
        fitness=1.0, per_year_sharpe={2021: 0.5}, weight_concentration=0.05,
    )
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.soft_scores["turnover_band"] < 0.0


def test_turnover_band_score_negative_when_above_ceiling():
    m = AlphaMetrics(
        sharpe=1.0, annual_return=0.1, turnover=TURNOVER_BAND[1] + 0.05, max_drawdown=0.1,
        fitness=1.0, per_year_sharpe={2021: 0.5}, weight_concentration=0.05,
    )
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.soft_scores["turnover_band"] < 0.0


def test_per_year_min_zero_when_per_year_sharpe_empty():
    m = AlphaMetrics(
        sharpe=1.0, annual_return=0.1, turnover=0.3, max_drawdown=0.1, fitness=1.0,
        per_year_sharpe={}, weight_concentration=0.05,
    )
    verdict = GateEvaluator().evaluate(m, self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.soft_scores["per_year_min"] == 0.0


def test_gate_verdict_is_frozen():
    import pytest
    verdict = GateVerdict(passed=True, hard_failures=[], soft_scores={})
    with pytest.raises(AttributeError):
        verdict.passed = False  # type: ignore[misc]
