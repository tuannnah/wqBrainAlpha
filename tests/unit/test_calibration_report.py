# tests/unit/test_calibration_report.py
"""Test CalibrationReport: dataclass frozen+slots, dùng để Task 4.5.4 lắp vào."""

from __future__ import annotations

import math

import pytest

from config.thresholds import CALIBRATION_MIN_SAMPLE_N, CALIBRATION_RHO_BAR
from src.calibration.report import CalibrationReport, FamilyCalibration, calibration_warnings


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


def test_family_calibration_holds_fields_and_frozen():
    # T4.2: FamilyCalibration mang ρ + hệ số local->Brain ước lượng riêng MỘT họ.
    fc = FamilyCalibration(
        family="pv_reversal", n=12, spearman_sharpe=0.42, local_to_brain_ratio=1.35,
    )
    assert fc.family == "pv_reversal"
    assert fc.n == 12
    assert fc.spearman_sharpe == pytest.approx(0.42)
    assert fc.local_to_brain_ratio == pytest.approx(1.35)
    with pytest.raises(AttributeError):
        fc.n = 99  # type: ignore[misc]


def test_report_by_family_holds_dict_of_family_calibration():
    fc = FamilyCalibration(
        family="momentum", n=40, spearman_sharpe=0.6, local_to_brain_ratio=1.2,
    )
    report = CalibrationReport(
        n=40, spearman_sharpe=0.6, spearman_fitness=0.5, spearman_submit_score=0.55,
        self_corr_agreement=0.8, decile_hit_rate=0.4, by_year={},
        by_family={"momentum": fc},
    )
    assert report.by_family["momentum"] is fc


# --- T4.3: calibration_warnings — cảnh báo n nhỏ / ρ thấp, KHÔNG tự hạ ngưỡng gì ---


def _khoe_report(**overrides) -> CalibrationReport:
    """Report 'khoẻ' mặc định: n đủ lớn, mọi ρ >= bar, không family nào yếu — override từng
    field để dựng ca cảnh báo cụ thể."""
    base = dict(
        n=CALIBRATION_MIN_SAMPLE_N, spearman_sharpe=CALIBRATION_RHO_BAR + 0.1,
        spearman_fitness=CALIBRATION_RHO_BAR + 0.1,
        spearman_submit_score=CALIBRATION_RHO_BAR + 0.1,
        self_corr_agreement=0.9, decile_hit_rate=0.5, by_year={}, by_family={},
    )
    base.update(overrides)
    return CalibrationReport(**base)


def test_calibration_warnings_rong_khi_moi_thu_khoe():
    assert calibration_warnings(_khoe_report()) == []


def test_calibration_warnings_n_nho_tong():
    report = _khoe_report(n=CALIBRATION_MIN_SAMPLE_N - 1)
    warnings = calibration_warnings(report)
    assert any("chưa đáng tin" in w and f"n={report.n}" in w for w in warnings)


def test_calibration_warnings_rho_sharpe_thap_khong_bao_ha_nguong():
    report = _khoe_report(spearman_sharpe=CALIBRATION_RHO_BAR - 0.2)
    warnings = calibration_warnings(report)
    assert any(
        "spearman_sharpe" in w and "KHÔNG hạ" in w for w in warnings
    ), warnings


def test_calibration_warnings_rho_submit_score_thap_rieng():
    # spearman_sharpe khoẻ nhưng spearman_submit_score thấp -> vẫn cảnh báo riêng trục này
    # (đúng mục tiêu T4.1/T4.3: điểm-nộp có thể lệch dù Sharpe thô khớp hạng).
    report = _khoe_report(spearman_submit_score=CALIBRATION_RHO_BAR - 0.1)
    warnings = calibration_warnings(report)
    assert any("spearman_submit_score" in w and "KHÔNG hạ" in w for w in warnings)
    assert not any("spearman_sharpe=" in w for w in warnings)  # trục sharpe không bị nêu oan


def test_calibration_warnings_rho_nan_bao_khong_xac_dinh():
    report = _khoe_report(spearman_sharpe=math.nan)
    warnings = calibration_warnings(report)
    assert any("spearman_sharpe" in w and "NaN" in w for w in warnings)


def test_calibration_warnings_family_n_nho():
    fc = FamilyCalibration(
        family="pv_reversal", n=5, spearman_sharpe=0.9, local_to_brain_ratio=1.3,
    )
    report = _khoe_report(by_family={"pv_reversal": fc})
    warnings = calibration_warnings(report)
    assert any(
        "pv_reversal" in w and "n=5" in w and "chưa đáng tin" in w for w in warnings
    ), warnings


def test_calibration_warnings_family_rho_thap_du_mau():
    fc = FamilyCalibration(
        family="news_social", n=CALIBRATION_MIN_SAMPLE_N,
        spearman_sharpe=CALIBRATION_RHO_BAR - 0.3, local_to_brain_ratio=1.1,
    )
    report = _khoe_report(by_family={"news_social": fc})
    warnings = calibration_warnings(report)
    assert any(
        "news_social" in w and "KHÔNG hạ" in w for w in warnings
    ), warnings
