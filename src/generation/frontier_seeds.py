"""Kho seed FRONTIER — core alt-data từ 12 dataset ít người đào (users < ~350).

Bối cảnh (spec 2026-07-14-frontier-seeds-design.md): 19 core tuyển chọn cũ đã bão hoà,
GP chỉ quanh quẩn price/volume yfinance → nguồn ý tưởng cạn. Account có quyền 299 dataset;
module này khai thác 12 dataset gần như chưa ai đào (self-corr kỳ vọng thấp — đúng chỗ dễ
qua gate nộp) với field ĐÃ VERIFY LIVE qua API /data-fields ngày 2026-07-14.

Nguyên tắc (skill WQ Brain):
- Hypothesis 4 phần ghi ở comment từng nhóm (quan sát → cơ chế → hiệu ứng → công thức).
- Field VECTOR bọc vec_avg/vec_sum trước khi vào operator MATRIX (pre-filter chặn nếu không).
- Field thưa (insider/call/filing/13F/kỳ short-interest) bắt buộc ts_backfill (22/66).
- Core trần depth ≤ 5 — chừa 2 mức cho tuner bọc (vector_neut/rank); neutralization/decay
  đi qua SETTINGS sim (đường _sim_direct), không bọc trong biểu thức.
- Sai dấu không chết: mini-sweep của _sim_direct tự flip dấu khi sharpe quá âm.

Điểm cắm giai đoạn B (spec): thêm `generate_frontier_cores(catalog)` trong module này và
nối kết quả vào cùng chỗ wire (closed_loop_adapters._gather_direct_cores) — không đổi kiến trúc.
"""

from __future__ import annotations

from src.lang.registry import ArgKind, OpCategory, register

# `vec_avg`/`vec_sum` là operator VECTOR-only của Brain thật — panel local (MiniBrain)
# không có kiểu dữ liệu VECTOR nên KHÔNG có/cần impl thật; mọi core dùng chúng đi THẲNG
# Brain sim (đường _sim_direct), Evaluator local không bao giờ gọi tới impl bên dưới. Đăng
# ký ở đây (không phải operators_local) để `src.lang.parser.parse` (strict) nhận diện được
# 2 tên operator này khi validate core frontier — nếu không đăng ký, parse() sẽ ném
# ParseError "operator không tồn tại trong registry" dù core hoàn toàn hợp lệ trên Brain.
# `gp_usable=False`: GP không được tự ý sinh 2 op này (không hiểu ngữ nghĩa VECTOR->MATRIX).
def _vec_reduce_khong_impl_local(*_args: object, **_kwargs: object) -> object:
    raise NotImplementedError(
        "vec_avg/vec_sum chỉ chạy trên Brain thật (VECTOR) — không evaluate được ở panel local"
    )


for _vec_op_name in ("vec_avg", "vec_sum"):
    register(
        name=_vec_op_name, category=OpCategory.SCALING,
        signature=(ArgKind.PANEL,), bounded=False, gp_usable=False,
    )(_vec_reduce_khong_impl_local)
del _vec_op_name

# ============================== insider_trx_matrix (M) =============================
# Quan sát: insider mua bằng tiền túi là tín hiệu thông tin mạnh (Lakonishok-Lee 2001);
# bán có thể chỉ là đa dạng hoá. Cơ chế: bất cân xứng thông tin nội bộ. Hiệu ứng: mua ròng
# → drift dương các tuần sau. Field sự kiện D0 (thưa) → ts_backfill 66.
_INSIDER_CORES = (
    "subtract(ts_backfill(total_buy_transaction_count, 66), "
    "ts_backfill(total_sell_transaction_count, 66))",
    "subtract(ts_backfill(total_top_buy_transaction_count, 66), "
    "ts_backfill(total_top_sell_transaction_count, 66))",
    "subtract(ts_backfill(usd_top_secondary_signal_value, 66), "
    "ts_backfill(usd_top_quaternary_signal_value, 66))",
    "subtract(ts_backfill(usd_secondary_signal_value, 66), "
    "ts_backfill(usd_quaternary_signal_value, 66))",
    "ts_backfill(directional_indicator_score, 66)",
)

# ================================= insiders3 (V) ===================================
# Quan sát: giọng điệu 8-K/10-Q (NLP) chứa thông tin chưa vào giá ngay (Loughran-McDonald).
# Cơ chế: nhà đầu tư đọc chậm văn bản dài. Hiệu ứng: tone dương → drift dương. Filing thưa
# → vec_avg trước (VECTOR), ts_backfill 66 sau.
_FILING_TONE_CORES = (
    "ts_backfill(subtract(vec_avg(insd3_8k_positive_score), "
    "vec_avg(insd3_8k_negative_score)), 66)",
    "ts_backfill(vec_avg(insd3_10q_tone_score), 66)",
)

# ========================== earningscall_sentiment (M) =============================
# Quan sát: phần Q&A (trả lời KHÔNG kịch bản) lộ tin thật hơn phần thuyết trình soạn sẵn
# (Larcker-Zakolyukina). Cơ chế: chi phí nói dối khi bị hỏi xoáy. Hiệu ứng: tone Q&A dương
# → drift dương; lo ngại leverage/valuation → âm. Call theo quý → ts_backfill 66; chuẩn hoá
# theo tổng chunk (+1 tránh chia 0).
_CALL_CORES = (
    "divide(subtract(ts_backfill(count_positive_profitability_answer, 66), "
    "ts_backfill(count_negative_profitability_answer, 66)), "
    "add(1, ts_backfill(answer_chunk_count, 66)))",
    "divide(subtract(ts_backfill(count_positive_cashflow_summary, 66), "
    "ts_backfill(count_negative_cashflow_summary, 66)), "
    "add(1, ts_backfill(summary_chunk_count, 66)))",
    "multiply(-1, divide(ts_backfill(count_negative_leverage_question, 66), "
    "add(1, ts_backfill(question_chunk_count, 66))))",
    "multiply(-1, divide(ts_backfill(count_negative_valuation_summary, 66), "
    "add(1, ts_backfill(summary_chunk_count, 66))))",
    # Trả lời thật lạc quan hơn kịch bản thuyết trình → tự tin thật (không phải spin).
    "subtract(ts_backfill(count_positive_profitability_answer, 66), "
    "ts_backfill(count_positive_profitability_presentation, 66))",
)

# ============================== filing_sentiment (V) ===============================
# Quan sát: mật độ từ tích cực/tiêu cực trong 10-K/10-Q dự báo return (Loughran-McDonald).
# Cơ chế: underreaction với văn bản. Hiệu ứng: net tone dương → drift dương.
_FILING_SENT_CORES = (
    "ts_backfill(divide(subtract(vec_sum(positive_sentiment_term_count_2), "
    "vec_sum(negative_sentiment_term_count_2)), "
    "add(1, vec_sum(total_word_quantity_2))), 66)",
    "ts_backfill(vec_avg(aggregate_sentiment_score_2), 66)",
)

# ============================ stock_search_trends (M) ==============================
# Quan sát: search Google tăng đột biến = chú ý bán lẻ (Da-Engelberg-Gao "In Search of
# Attention"). Cơ chế: retail chỉ MUA thứ họ chú ý → áp lực giá ngắn hạn rồi mean-revert.
# Hiệu ứng: spike ngắn/nền dài → dương ngắn hạn; chú ý cao kéo dài → overpricing, fade.
_ATTENTION_CORES = (
    "divide(search_interest_7d_corporate_name, "
    "add(0.01, search_interest_84d_corporate_name))",
    "subtract(search_interest_today_corporate_name, search_interest_28d_corporate_name)",
    "multiply(-1, ts_rank(search_interest_28d_corporate_name, 250))",
    # search_interest (V): điểm nowcast chuẩn hoá — bản VECTOR bổ sung của cùng hypothesis.
    "ts_backfill(vec_avg(relative_interest_score_4), 22)",
)

# =============================== order_flow_imb (M) ================================
# Quan sát: imbalance option theo NHÓM tay chơi (customer=retail, firm/pro=informed,
# broker-dealer=market-maker) — Pan-Poteshman 2006: order flow option chứa thông tin.
# Cơ chế: informed chọn option vì đòn bẩy; retail sai hệ thống. Hiệu ứng: theo firm/pro,
# fade retail/dealer. Field daily (M) → không cần backfill.
_OPTION_FLOW_CORES = (
    "multiply(-1, ts_mean(customer_vol_imbalance, 5))",
    "ts_mean(firm_vol_imbalance, 5)",
    "ts_mean(pro_customer_vol_imbalance_otm, 5)",
    "multiply(-1, ts_mean(broker_dealer_vol_imbalance, 5))",
)

# ============================ order_book_imbalance (V) =============================
# Quan sát: imbalance đọng lại sau phiên đấu giá đóng cửa (%ADV) chưa được hấp thụ.
# Cơ chế: cầu/cung dư chuyển sang phiên sau. Hiệu ứng: drift cùng chiều imbalance 1-5 ngày.
_MICROSTRUCTURE_CORES = (
    "vec_avg(auction_order_imbalance_pct_adv)",
    "ts_delta(vec_avg(auction_order_imbalance_pct_adv), 5)",
)

# ================================ expected_move (M) ================================
# Quan sát: bất đối xứng kỳ vọng lên/xuống từ option + expected move cao = cổ phiếu lottery
# (Bali-Cakici-Whitelaw). Cơ chế: preference lottery → overpricing vol cao; skew ẩn hướng.
# Hiệu ứng: skew lên > xuống → dương; expected move cao kéo dài → fade (low-vol anomaly).
_EXPECTED_MOVE_CORES = (
    "subtract(upward_lognorm_move_percent_3, downward_lognorm_move_percent_3)",
    "subtract(upward_lognorm_expected_change_7, downward_lognorm_expected_change_7)",
    "multiply(-1, ts_rank(straddle_move_percent_7, 250))",
    "multiply(-1, ts_delta(straddle_move_percent_7, 5))",
)

# ================================ institutions18 (M) ===============================
# Quan sát: breadth sở hữu tổ chức tăng dự báo return (Chen-Hong-Stein 2002); crowding
# pct_held cao = chật chỗ, dễ unwind. 13F theo quý → ts_backfill 66.
_OWNERSHIP_CORES = (
    "ts_delta(ts_backfill(inst18_instownership_num_held, 66), 66)",
    "subtract(ts_backfill(inst18_instownership_cur_holding, 66), "
    "ts_backfill(inst18_instownership_pre_holding, 66))",
    "multiply(-1, ts_rank(ts_backfill(inst18_fundownershipv2_pct_held, 66), 250))",
)

# ============================= fund_holdings_panel (V) =============================
# Quan sát: panel quỹ hằng ngày — độ tập trung nắm giữ (HHI/crowding score) cao = rủi ro
# unwind đồng loạt; breadth tài khoản tăng = dòng tiền mới. Hiệu ứng: fade crowding, theo
# breadth momentum. Field daily (V) cov ~1.0 → vec_avg, không cần backfill.
_FUND_PANEL_CORES = (
    "multiply(-1, vec_avg(holding_value_distribution_score))",
    "multiply(-1, ts_rank(vec_avg(herfindahl_index_holdings), 250))",
    "ts_delta(vec_avg(holder_account_total), 22)",
    "ts_delta(vec_avg(top_weighted_account_number), 22)",
)

# ============================== short_interest_pred (M) ============================
# Quan sát: short interest công bố 2 tuần/lần; model dự báo thay đổi + surprise so dự báo.
# Cơ chế: shorts là informed (Boehmer-Jones-Zhang). Hiệu ứng: dự báo tăng short / surprise
# dương → âm. Kỳ 2 tuần → ts_backfill 22. (Khác core cũ: cũ dùng surprise_ratio backfill 66.)
_SHORT_PERIOD_CORES = (
    "multiply(-1, ts_backfill(short_interest_predicted_change, 22))",
    "multiply(-1, subtract(ts_backfill(short_interest_surprise_ratio, 22), "
    "ts_backfill(prior_short_interest_surprise_ratio, 22)))",
    "multiply(-1, ts_backfill(short_interest_surprise_amount, 22))",
)

# ================================= us_short_sale (M) ===============================
# Quan sát: tỷ lệ khối lượng bán khống HẰNG NGÀY (Reg SHO) — shorts informed ở tần suất
# ngày (Diether-Lee-Werner). Hiệu ứng: tỷ lệ short cao/tăng → âm ngắn hạn. Daily, cov 1.0.
_SHORT_DAILY_CORES = (
    "multiply(-1, divide(ts_sum(executed_short_trade_share_count, 5), "
    "add(1, ts_sum(aggregate_executed_trade_share_count, 5))))",
    "ts_delta(divide(reported_short_sale_share_quantity, "
    "add(1, reported_total_trade_share_quantity)), 22)",
)

FRONTIER_CORES: tuple[str, ...] = (
    _INSIDER_CORES + _FILING_TONE_CORES + _CALL_CORES + _FILING_SENT_CORES
    + _ATTENTION_CORES + _OPTION_FLOW_CORES + _MICROSTRUCTURE_CORES
    + _EXPECTED_MOVE_CORES + _OWNERSHIP_CORES + _FUND_PANEL_CORES
    + _SHORT_PERIOD_CORES + _SHORT_DAILY_CORES
)

# Field → category (chọn neutralization + test bất biến). MỌI field đã verify live
# 2026-07-14 qua /data-fields USA/TOP3000/D1 (xem logs/verified_fields_20260714.json).
_CAT = {
    "insider": (
        "total_buy_transaction_count", "total_sell_transaction_count",
        "total_top_buy_transaction_count", "total_top_sell_transaction_count",
        "usd_top_secondary_signal_value", "usd_top_quaternary_signal_value",
        "usd_secondary_signal_value", "usd_quaternary_signal_value",
        "directional_indicator_score",
    ),
    "call_filing": (
        "insd3_8k_positive_score", "insd3_8k_negative_score", "insd3_10q_tone_score",
        "count_positive_profitability_answer", "count_negative_profitability_answer",
        "answer_chunk_count", "count_positive_cashflow_summary",
        "count_negative_cashflow_summary", "summary_chunk_count",
        "count_negative_leverage_question", "question_chunk_count",
        "count_negative_valuation_summary", "count_positive_profitability_presentation",
        "positive_sentiment_term_count_2", "negative_sentiment_term_count_2",
        "total_word_quantity_2", "aggregate_sentiment_score_2",
    ),
    "attention": (
        "search_interest_7d_corporate_name", "search_interest_84d_corporate_name",
        "search_interest_today_corporate_name", "search_interest_28d_corporate_name",
        "relative_interest_score_4",
    ),
    "option_flow": (
        "customer_vol_imbalance", "firm_vol_imbalance", "pro_customer_vol_imbalance_otm",
        "broker_dealer_vol_imbalance", "upward_lognorm_move_percent_3",
        "downward_lognorm_move_percent_3", "upward_lognorm_expected_change_7",
        "downward_lognorm_expected_change_7", "straddle_move_percent_7",
    ),
    "microstructure": ("auction_order_imbalance_pct_adv",),
    # ownership = 13F theo QUÝ (thưa, bắt buộc ts_backfill); fund_panel = panel quỹ HẰNG
    # NGÀY (không backfill) — tách 2 category để test "field thưa phải ts_backfill" đúng.
    "ownership": (
        "inst18_instownership_num_held", "inst18_instownership_cur_holding",
        "inst18_instownership_pre_holding", "inst18_fundownershipv2_pct_held",
    ),
    "fund_panel": (
        "holding_value_distribution_score", "herfindahl_index_holdings",
        "holder_account_total", "top_weighted_account_number",
    ),
    "short_period": (
        "short_interest_predicted_change", "short_interest_surprise_ratio",
        "prior_short_interest_surprise_ratio", "short_interest_surprise_amount",
    ),
    "short_daily": (
        "executed_short_trade_share_count", "aggregate_executed_trade_share_count",
        "reported_short_sale_share_quantity", "reported_total_trade_share_quantity",
    ),
}
FRONTIER_CATEGORY_BY_FIELD: dict[str, str] = {
    f: cat for cat, fields in _CAT.items() for f in fields
}
FRONTIER_FIELDS: frozenset[str] = frozenset(FRONTIER_CATEGORY_BY_FIELD)

# Field kiểu VECTOR (verify live) — bắt buộc vec_avg/vec_sum trước operator MATRIX.
FRONTIER_VECTOR_FIELDS: frozenset[str] = frozenset({
    "insd3_8k_positive_score", "insd3_8k_negative_score", "insd3_10q_tone_score",
    "positive_sentiment_term_count_2", "negative_sentiment_term_count_2",
    "total_word_quantity_2", "aggregate_sentiment_score_2",
    "relative_interest_score_4", "auction_order_imbalance_pct_adv",
    "holding_value_distribution_score", "herfindahl_index_holdings",
    "holder_account_total", "top_weighted_account_number",
})
