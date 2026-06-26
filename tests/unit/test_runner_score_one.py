"""Test score_one: parse→eval→backtest→metrics→gate trên small_panel, không mạng/sim Brain."""

from __future__ import annotations

import numpy as np

from src.backtest.config import PortfolioConfig
from src.backtest.gates import GateVerdict
from src.backtest.metrics_local import AlphaMetrics
from src.pipeline.runner import score_one


def test_valid_expression_returns_metrics_and_verdict(small_panel) -> None:  # noqa: ANN001
    metrics, verdict = score_one("close", PortfolioConfig(delay=1), small_panel)
    assert isinstance(metrics, AlphaMetrics)
    assert isinstance(verdict, GateVerdict)
    assert np.isfinite(metrics.sharpe)


def test_parse_error_returns_failing_verdict_not_exception(small_panel) -> None:  # noqa: ANN001
    metrics, verdict = score_one("not_a_real_op(close,", PortfolioConfig(), small_panel)
    assert verdict.passed is False
    assert any("parse" in f.lower() for f in verdict.hard_failures)
    assert metrics.sharpe == 0.0


def test_unknown_field_returns_failing_verdict(small_panel) -> None:  # noqa: ANN001
    metrics, verdict = score_one("totally_unknown_field_xyz", PortfolioConfig(), small_panel)
    assert verdict.passed is False


def test_pool_aware_metrics_unchanged_by_pool(small_panel) -> None:  # noqa: ANN001
    dates = small_panel.dates
    pool = {1: (dates, np.linspace(0.01, 0.10, len(dates)))}
    m_no, _ = score_one("close", PortfolioConfig(delay=1), small_panel, pool=None)
    m_pool, v_pool = score_one("close", PortfolioConfig(delay=1), small_panel, pool=pool)
    assert m_no == m_pool  # pool chỉ ảnh hưởng verdict.self_corr, không đổi AlphaMetrics
    assert isinstance(v_pool, GateVerdict)


def test_deterministic_same_inputs_same_output(small_panel) -> None:  # noqa: ANN001
    cfg = PortfolioConfig(delay=1)
    m1, v1 = score_one("close", cfg, small_panel)
    m2, v2 = score_one("close", cfg, small_panel)
    assert m1 == m2
    assert v1.passed == v2.passed
    assert v1.hard_failures == v2.hard_failures
