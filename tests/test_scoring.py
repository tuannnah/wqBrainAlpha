"""Test metrics normalize, scorer và filter."""

from __future__ import annotations

import pytest

from src.scoring.filter import FilterThresholds, passes
from src.scoring.metrics import normalize, submit_score
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


def test_submit_score_cong_thuc_min_sharpe_fitness_chuan_hoa():
    # T4.1: submit_score = min(sharpe/SUBMIT_SHARPE_REF, fitness/SUBMIT_FITNESS_REF) —
    # công thức điểm-nộp dùng chung (combine_stage/closed_loop_adapters/calibration harness).
    from config.thresholds import SUBMIT_FITNESS_REF, SUBMIT_SHARPE_REF

    assert submit_score(1.25, 1.0) == pytest.approx(1.0)
    # trục fitness thấp hơn trục sharpe -> min chọn trục fitness (siết bởi trục yếu nhất).
    assert submit_score(2.5, 0.5) == pytest.approx(0.5 / SUBMIT_FITNESS_REF)
    assert submit_score(0.0, 1.0) == pytest.approx(0.0 / SUBMIT_SHARPE_REF)
