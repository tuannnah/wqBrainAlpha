"""Pha 4: pre-sim floor CALIBRATED thay đơn-ngưỡng cứng 0.5 (IMPROVEMENT_SPEC §4).

Thay vì hằng số 0.5 tuỳ ý, suy floor từ MỤC TIÊU Brain sharpe qua tỉ lệ hiệu chỉnh đo được
brain≈local×1.28 (winner local 1.23 -> Brain 1.57). Muốn Brain>=target thì local phải
>= target/1.28. Ghi được ngưỡng áp dụng để audit."""

from __future__ import annotations

import pytest

from config.thresholds import CALIBRATION_LOCAL_TO_BRAIN, calibrated_floor


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
