"""Test GateEvaluator.evaluate_with_pool: self_corr tinh THAT tu PoolCorrelation.max_corr,
khong phai gia tri truyen tay; hard gate SELF_CORR_MAX van ap dung dung nhu evaluate()."""

from __future__ import annotations

import numpy as np

from config.thresholds import SELF_CORR_MAX
from src.backtest.gates import GateEvaluator
from src.backtest.metrics_local import AlphaMetrics
from src.backtest.pool_corr import PoolCorrelation


def _dates(start: str, n: int) -> np.ndarray:
    return (np.datetime64(start) + np.arange(n)).astype("datetime64[D]")


def _good_metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.5, annual_return=0.20, turnover=0.30, max_drawdown=0.10,
        fitness=2.0, per_year_sharpe={2021: 1.2}, weight_concentration=0.05,
    )


def test_evaluate_with_pool_passes_when_pool_empty():
    verdict = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=np.array([0.01, -0.01, 0.02]),
        candidate_dates=_dates("2021-01-01", 3), pool_corr=PoolCorrelation(pool={}),
        depth=3, fields_ok=True,
    )
    assert verdict.passed is True
    assert verdict.hard_failures == []


def test_evaluate_with_pool_hard_fails_when_identical_to_pool_alpha():
    dates = _dates("2021-01-01", 10)
    pnl = np.linspace(0.01, 0.10, 10)
    pool_corr = PoolCorrelation(pool={1: (dates, pnl.copy())})
    verdict = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=pnl.copy(), candidate_dates=dates,
        pool_corr=pool_corr, depth=3, fields_ok=True,
    )
    assert verdict.passed is False
    assert any("self_corr" in f for f in verdict.hard_failures)


def test_evaluate_with_pool_uses_same_threshold_as_evaluate():
    dates = _dates("2021-01-01", 10)
    pnl = np.linspace(0.01, 0.10, 10)
    pool_corr = PoolCorrelation(pool={1: (dates, pnl.copy())})
    rho, _ = pool_corr.max_corr(pnl.copy(), dates)
    assert rho >= SELF_CORR_MAX  # identical series -> rho=1.0 >= 0.70

    verdict_pool = GateEvaluator().evaluate_with_pool(
        _good_metrics(), candidate_pnl=pnl.copy(), candidate_dates=dates,
        pool_corr=pool_corr, depth=3, fields_ok=True,
    )
    verdict_manual = GateEvaluator().evaluate(
        _good_metrics(), self_corr=rho, depth=3, fields_ok=True
    )
    assert verdict_pool.hard_failures == verdict_manual.hard_failures
    assert verdict_pool.soft_scores == verdict_manual.soft_scores


def test_evaluate_unchanged_signature_still_works():
    # evaluate() Phase 4 KHONG bi pha vo boi Task 6.2
    verdict = GateEvaluator().evaluate(_good_metrics(), self_corr=0.1, depth=3, fields_ok=True)
    assert verdict.passed is True
