"""Pha 0: helper chẩn đoán — map reasons hard_filter -> mã fail_check chuẩn, và suy family
từ field/cấu trúc biểu thức (IMPROVEMENT_SPEC §3 Pha 0: fail_check, family)."""

from __future__ import annotations

from src.reporting.diagnostics import classify_family, fail_check_from_reasons


def test_fail_check_low_sharpe():
    assert fail_check_from_reasons(["sharpe 0.40 < 0.5"]) == "LOW_SHARPE"


def test_fail_check_low_fitness():
    assert fail_check_from_reasons(["fitness 0.71 <= 1.0"]) == "LOW_FITNESS"


def test_fail_check_turnover():
    assert fail_check_from_reasons(["turnover 0.85 ngoài [0.01, 0.70]"]) == "HIGH_TURNOVER"


def test_fail_check_uu_tien_sharpe_truoc_fitness():
    """Nhiều reason -> chọn cái đầu tiên theo thứ tự nghiêm trọng (sharpe trước fitness)."""
    assert fail_check_from_reasons(["sharpe 0.4 < 0.5", "fitness 0.7 <= 1.0"]) == "LOW_SHARPE"


def test_fail_check_rong():
    assert fail_check_from_reasons([]) == ""


def test_fail_check_khong_khop_giu_nguyen_raw():
    """Reason lạ -> trả UNKNOWN thay vì mất thông tin."""
    assert fail_check_from_reasons(["drawdown 0.5 >= 0.4"]) == "HIGH_DRAWDOWN"
    assert fail_check_from_reasons(["cái gì đó lạ"]) == "UNKNOWN"


def test_family_pv_reversal():
    """close/vwap/open + ts_mean -> họ pv_reversal (cụm đã bão hoà)."""
    assert classify_family("multiply(-1, ts_mean(subtract(close, vwap), 10))") == "pv_reversal"


def test_family_options_iv():
    assert classify_family("subtract(implied_volatility_mean_30, historical_volatility_30)") == "options_iv"


def test_family_news_social():
    assert classify_family("ts_delta(snt_social_value, 5)") == "news_social"


def test_family_momentum():
    assert classify_family("ts_delta(close, 60)") == "momentum"


def test_family_fundamental():
    assert classify_family("divide(ts_backfill(ebit, 60), assets)") == "fundamental"


def test_family_analyst():
    assert classify_family("ts_delta(snt1_d1_netearningsrevision, 20)") == "analyst"


def test_family_unknown():
    assert classify_family("rank(some_unknown_field)") == "other"
