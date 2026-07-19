# tests/unit/test_calibration_report.py
"""Test CalibrationReport: dataclass frozen+slots, dùng để Task 4.5.4 lắp vào."""

from __future__ import annotations

import pytest

from src.calibration.report import CalibrationReport


def test_report_holds_all_fields():
    report = CalibrationReport(
        n=50, spearman_sharpe=0.62, spearman_fitness=0.55, spearman_submit_score=0.58,
        self_corr_agreement=0.80, decile_hit_rate=0.40,
        by_year={2022: 0.5, 2023: 0.6},
    )
    assert report.n == 50
    assert report.spearman_sharpe == pytest.approx(0.62)
    assert report.spearman_fitness == pytest.approx(0.55)
    assert report.spearman_submit_score == pytest.approx(0.58)
    assert report.self_corr_agreement == pytest.approx(0.80)
    assert report.decile_hit_rate == pytest.approx(0.40)
    assert report.by_year == {2022: 0.5, 2023: 0.6}
    assert report.by_family == {}  # mặc định rỗng (T4.2) khi không truyền


def test_report_is_frozen():
    report = CalibrationReport(
        n=1, spearman_sharpe=0.0, spearman_fitness=0.0, spearman_submit_score=0.0,
        self_corr_agreement=0.0, decile_hit_rate=0.0, by_year={},
    )
    with pytest.raises(AttributeError):
        report.n = 2  # type: ignore[misc]
