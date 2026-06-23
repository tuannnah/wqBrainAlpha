# tests/unit/test_thresholds.py
from config import thresholds


def test_threshold_values_present_and_sane():
    assert thresholds.MAX_DEPTH == 7
    assert thresholds.SELF_CORR_MAX == 0.70
    assert thresholds.TURNOVER_FLOOR == 0.125
    assert 0.0 < thresholds.WEIGHT_CONCENTRATION_CAP <= 1.0
    assert thresholds.CALIBRATION_RHO_BAR == 0.5
    lo, hi = thresholds.TURNOVER_BAND
    assert 0.0 <= lo < hi
