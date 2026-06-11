"""Test metrics normalize, scorer và filter."""

from __future__ import annotations

from src.scoring.filter import FilterThresholds, passes
from src.scoring.metrics import normalize
from src.scoring.scorer import score


def test_normalize_dien_default_khi_thieu():
    m = normalize({"sharpe": 1.5})
    assert m["sharpe"] == 1.5
    assert m["turnover"] == 0.5  # default
    assert m["drawdown"] == 1.0


def test_score_cong_thuc():
    m = {"sharpe": 2.0, "fitness": 1.0, "turnover": 0.3, "drawdown": 0.1}
    expected = 0.40 * 2.0 + 0.30 * 1.0 + 0.15 * (1 - 0.1) + 0.15 * (1 - 0.0)
    assert abs(score(m) - expected) < 1e-9


def test_filter_pass():
    m = {"sharpe": 1.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1}
    ok, reasons = passes(m)
    assert ok, reasons


def test_filter_fail_sharpe_thap():
    m = {"sharpe": 0.5, "fitness": 1.2, "turnover": 0.3, "drawdown": 0.1}
    ok, reasons = passes(m)
    assert not ok
    assert any("sharpe" in r for r in reasons)


def test_filter_custom_threshold():
    m = {"sharpe": 1.0, "fitness": 1.1, "turnover": 0.3, "drawdown": 0.1}
    ok, _ = passes(m, FilterThresholds(min_sharpe=0.9))
    assert ok
