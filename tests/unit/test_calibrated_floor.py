"""Pha 4: pre-sim floor CALIBRATED thay đơn-ngưỡng cứng 0.5 (IMPROVEMENT_SPEC §4).

Thay vì hằng số 0.5 tuỳ ý, suy floor từ MỤC TIÊU Brain sharpe qua tỉ lệ hiệu chỉnh đo được
brain≈local×1.28 (winner local 1.23 -> Brain 1.57). Muốn Brain>=target thì local phải
>= target/1.28. Ghi được ngưỡng áp dụng để audit."""

from __future__ import annotations

import pytest

from config.thresholds import (
    CALIBRATION_LOCAL_TO_BRAIN,
    CALIBRATION_MIN_SAMPLE_N,
    calibrated_floor,
)


def test_ty_le_hieu_chinh_hop_ly():
    # brain ≈ local × 1.28 (đo trực tiếp) -> hằng số > 1.
    assert 1.1 < CALIBRATION_LOCAL_TO_BRAIN < 1.5


def test_floor_suy_tu_target_brain():
    # Muốn Brain >= 0.64 -> local floor = 0.64 / 1.28 = 0.5 (khớp floor cũ, nhưng nay derived).
    f = calibrated_floor(0.64)
    assert f == pytest.approx(0.5, abs=0.02)


def test_floor_cao_hon_khi_target_cao_hon():
    """Target Brain cao hơn -> floor local cao hơn (siết quota mạnh hơn)."""
    assert calibrated_floor(1.28) > calibrated_floor(0.64)


def test_floor_target_1_28_ra_1_0():
    assert calibrated_floor(1.28) == pytest.approx(1.0, abs=0.02)


# --- T4.2: family + family_coefficients (bảng hệ số per-family, dict truyền vào — KHÔNG
# wire vào gate production, chỉ hàm sẵn sàng + test) ---


def test_floor_dung_he_so_rieng_ho_khi_du_mau():
    coeffs = {"pv_reversal": (1.5, CALIBRATION_MIN_SAMPLE_N)}  # đủ mẫu (n == ngưỡng tối thiểu)
    f = calibrated_floor(0.75, family="pv_reversal", family_coefficients=coeffs)
    assert f == pytest.approx(0.75 / 1.5)  # dùng 1.5, KHÔNG phải 1.28 chung


def test_floor_fallback_khi_ho_chua_du_mau():
    coeffs = {"pv_reversal": (1.5, CALIBRATION_MIN_SAMPLE_N - 1)}  # thiếu 1 mẫu so với ngưỡng
    f = calibrated_floor(0.64, family="pv_reversal", family_coefficients=coeffs)
    assert f == pytest.approx(calibrated_floor(0.64))  # fallback hệ số chung 1.28


def test_floor_fallback_khi_ho_khong_co_trong_bang():
    coeffs = {"momentum": (1.2, 50)}
    f = calibrated_floor(0.64, family="pv_reversal", family_coefficients=coeffs)
    assert f == pytest.approx(calibrated_floor(0.64))


def test_floor_fallback_khi_khong_truyen_family():
    # Không truyền family/family_coefficients -> hành vi CŨ y hệt (gate production không đổi).
    assert calibrated_floor(0.64) == pytest.approx(0.64 / CALIBRATION_LOCAL_TO_BRAIN)


def test_floor_fallback_khi_he_so_ho_la_nan():
    import math

    coeffs = {"pv_reversal": (math.nan, 50)}  # đủ mẫu nhưng hệ số vô nghĩa (ratio không tính được)
    f = calibrated_floor(0.64, family="pv_reversal", family_coefficients=coeffs)
    assert f == pytest.approx(calibrated_floor(0.64))


def test_floor_fallback_khi_he_so_ho_la_0():
    # Review T4 Important #1: local_to_brain_ratio = median(brain/local) HOÀN TOÀN có thể ra
    # đúng 0.0 (đa số brain_sharpe=0 trong họ) — 0.0 vẫn "finite" nên guard cũ (chỉ check
    # isfinite) lọt qua, gây target/0.0 = ZeroDivisionError khi floor này được wire dùng thật.
    # Phải fallback hệ số chung, KHÔNG raise.
    coeffs = {"pv_reversal": (0.0, 50)}  # đủ mẫu nhưng hệ số = 0 (vô nghĩa cho phép chia)
    f = calibrated_floor(0.64, family="pv_reversal", family_coefficients=coeffs)
    assert f == pytest.approx(calibrated_floor(0.64))


def test_floor_fallback_khi_he_so_ho_am():
    # Hệ số local->Brain ÂM cũng vô nghĩa cho floor (Brain kỳ vọng ngược dấu local -> floor suy
    # ra không còn ý nghĩa "local càng cao Brain càng cao") -> fallback, không dùng hệ số âm.
    coeffs = {"pv_reversal": (-1.2, 50)}
    f = calibrated_floor(0.64, family="pv_reversal", family_coefficients=coeffs)
    assert f == pytest.approx(calibrated_floor(0.64))
