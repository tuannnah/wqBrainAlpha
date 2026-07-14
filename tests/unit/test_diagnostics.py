"""Pha 0: helper chẩn đoán — map reasons hard_filter -> mã fail_check chuẩn, và suy family
từ field/cấu trúc biểu thức (IMPROVEMENT_SPEC §3 Pha 0: fail_check, family)."""

from __future__ import annotations

from src.reporting.diagnostics import (
    categorize_presim_reason,
    classify_family,
    fail_check_from_reasons,
)


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


def test_family_short_interest_theo_field_securities_lending():
    """Field securities-lending verify LIVE 14/07 (shortinterest3: loan_utilization_ratio,
    mean_loan_rate) phải phân loại short_interest — KHÔNG rơi vào 'fundamental' chỉ vì có
    ts_backfill (nhánh fundamental match substring 'ts_backfill' phía sau)."""
    assert classify_family(
        "multiply(-1, ts_mean(ts_backfill(loan_utilization_ratio, 5), 22))"
    ) == "short_interest"
    assert classify_family(
        "multiply(-1, ts_delta(ts_backfill(mean_loan_rate, 5), 22))"
    ) == "short_interest"


def test_family_short_interest_theo_si_surprise():
    assert classify_family(
        "multiply(-1, ts_backfill(short_interest_surprise_ratio, 66))"
    ) == "short_interest"


def test_family_unknown():
    assert classify_family("rank(some_unknown_field)") == "other"


# --- F1 (review final feature/frontier-seeds): classify_family phải biết category frontier
# (FRONTIER_CATEGORY_BY_FIELD trong src/generation/frontier_seeds.py) — trước đây 18/40 core
# frontier có ts_backfill rơi chung vào "fundamental" (rule substring "ts_backfill"), khiến
# family-budget (max_per_family=8, đóng họ khi 0 pass) đóng oan sau 8 core đầu, bỏ phí quota
# presim đã đốt cho các core frontier còn lại đội lốt "fundamental".


def test_family_frontier_insider_co_ts_backfill_khong_roi_vao_fundamental():
    assert classify_family(
        "ts_backfill(directional_indicator_score, 66)"
    ) == "frontier_insider"


def test_family_frontier_call_filing():
    assert classify_family(
        "divide(subtract(ts_backfill(count_positive_profitability_answer, 66), "
        "ts_backfill(count_negative_profitability_answer, 66)), "
        "add(1, ts_backfill(answer_chunk_count, 66)))"
    ) == "frontier_call_filing"


def test_family_frontier_fund_panel_khong_ts_backfill():
    """Category fund_panel là panel daily (cov~1.0) — KHÔNG dùng ts_backfill nên trước đây
    rơi vào "other"; vẫn phải nhận family frontier riêng để không lẫn với core "other" khác."""
    assert classify_family(
        "ts_delta(vec_avg(holder_account_total), 22)"
    ) == "frontier_fund_panel"


def test_family_frontier_khong_doi_ket_qua_fundamental_cu():
    """Expr fundamental CŨ (không field frontier) không được đổi family. Dùng revenue/assets
    thay vì operating_income: "operating_income" chứa substring "rating" -> khớp oan rule
    "analyst" (lỗi có sẵn từ trước, KHÔNG liên quan F1 — ngoài phạm vi finding này)."""
    assert classify_family(
        "divide(ts_backfill(revenue, 66), ts_backfill(assets, 66))"
    ) == "fundamental"


def test_family_frontier_khong_doi_ket_qua_pv_momentum_cu():
    """Expr price/volume cũ (không field frontier) không được đổi family."""
    assert classify_family("multiply(-1, ts_mean(subtract(close, vwap), 10))") == "pv_reversal"
    assert classify_family("ts_delta(close, 60)") == "momentum"


def test_family_frontier_short_period_van_giu_short_interest_cu():
    """Category short_period tái dùng đúng field short_interest_pred mà rule "short_interest"
    (dòng trên) đã test từ trước — KHÔNG đổi tên thành "frontier_short_period", giữ nguyên
    family cũ vì cùng một nghĩa nghiệp vụ (short-interest surprise/predicted change)."""
    assert classify_family(
        "multiply(-1, ts_backfill(short_interest_predicted_change, 22))"
    ) == "short_interest"


# --- Task 3 (spec C2): phân loại lý do pre-sim reject (PreFilter.check) -> mã ổn định ---


def test_categorize_presim_operator_invalid():
    assert categorize_presim_reason("Operator không tồn tại: fake_op") == "OPERATOR_INVALID"


def test_categorize_presim_field_invalid():
    assert categorize_presim_reason("Field/hằng không tồn tại: fake_field") == "FIELD_INVALID"


def test_categorize_presim_depth_do_sau():
    assert categorize_presim_reason("Độ sâu > 7") == "DEPTH"


def test_categorize_presim_depth_so_node():
    assert categorize_presim_reason("Số node > 30") == "DEPTH"


def test_categorize_presim_parse_loi():
    assert categorize_presim_reason("Parse lỗi: unexpected token") == "PARSE"


def test_categorize_presim_ngoac_khong_can_bang():
    assert categorize_presim_reason("Dấu ngoặc không cân bằng") == "PARSE"


def test_categorize_presim_la_khong_khop_tra_fallback():
    """Reason lạ (không khớp luật nào) -> PRESIM_REJECT thay vì mất thông tin."""
    assert categorize_presim_reason("lý do lạ chưa từng thấy") == "PRESIM_REJECT"


def test_fail_check_from_reasons_van_khop_luat_cu_sau_khi_them_presim():
    """Thêm luật presim KHÔNG được phá luật cũ (sharpe/fitness/turnover/drawdown)."""
    assert fail_check_from_reasons(["sharpe 0.40 < 0.5"]) == "LOW_SHARPE"
    assert fail_check_from_reasons(["fitness 0.71 <= 1.0"]) == "LOW_FITNESS"
    assert fail_check_from_reasons(["turnover 0.85 ngoài [0.01, 0.70]"]) == "HIGH_TURNOVER"
    assert fail_check_from_reasons(["drawdown 0.5 >= 0.4"]) == "HIGH_DRAWDOWN"
