"""Test evaluate_local trong src/scoring/filter.py: wrap GateEvaluator, không đụng
passes/blocking_dimensions (filter sim-Brain cũ)."""

from __future__ import annotations

from src.backtest.gates import GateVerdict
from src.backtest.metrics_local import AlphaMetrics
from src.scoring.filter import evaluate_local, passes  # noqa: F401  (passes vẫn import được, không bị xoá)


def _passing_metrics() -> AlphaMetrics:
    return AlphaMetrics(
        sharpe=1.5, annual_return=0.2, turnover=0.3, max_drawdown=0.1, fitness=2.0,
        per_year_sharpe={2021: 1.0}, weight_concentration=0.05,
    )


def test_evaluate_local_returns_gate_verdict():
    verdict = evaluate_local(_passing_metrics(), self_corr=0.1, depth=3, fields_ok=True)
    assert isinstance(verdict, GateVerdict)
    assert verdict.passed is True


def test_evaluate_local_hard_fail_propagates_reason():
    m = _passing_metrics()
    verdict = evaluate_local(m, self_corr=0.99, depth=3, fields_ok=True)
    assert verdict.passed is False
    assert any("self_corr" in r for r in verdict.hard_failures)


def test_legacy_passes_function_still_importable_and_unmodified_signature():
    # đảm bảo evaluate_local KHÔNG phá filter cũ — passes() vẫn nhận (source, thresholds)
    import inspect
    sig = inspect.signature(passes)
    assert list(sig.parameters) == ["source", "thresholds"]
