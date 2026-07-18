"""Kho seed FRONTIER — core alt-data từ 13 dataset ít người đào (users < ~350).

Bối cảnh (spec 2026-07-14-frontier-seeds-design.md): 19 core tuyển chọn cũ đã bão hoà,
GP chỉ quanh quẩn price/volume yfinance → nguồn ý tưởng cạn. Account có quyền 299 dataset;
module này khai thác 13 dataset gần như chưa ai đào (self-corr kỳ vọng thấp — đúng chỗ dễ
qua gate nộp) với field ĐÃ VERIFY LIVE qua API /data-fields ngày 2026-07-14. (stock_search_trends
và search_interest là HAI dataset id riêng biệt — cùng nhóm attention nên gộp chung 1 comment
section bên dưới, không phải 12 dataset như bản nháp đầu.)

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
# 2026-07-14 qua /data-fields USA/TOP3000/D1 (xem logs/verified_frontier_fields_20260714.json).
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

# Category → neutralization group-neut (đường non-theme, docs WQ neutralization.md).
_NEUT_BY_CATEGORY = {
    "microstructure": "MARKET",
    "option_flow": "SECTOR",
    "ownership": "INDUSTRY",
    "fund_panel": "INDUSTRY",
    "insider": "SUBINDUSTRY",
    "call_filing": "SUBINDUSTRY",
    "attention": "SUBINDUSTRY",
    "short_period": "SUBINDUSTRY",
    "short_daily": "SUBINDUSTRY",
}
# Category → risk-neutralization cho Power Pool Theme (chỉ cho risk-neut).
_PP_BY_CATEGORY = {
    "attention": "CROWDING",
    "call_filing": "CROWDING",
    "ownership": "SLOW",
    "fund_panel": "SLOW",
    "insider": "SLOW",
}


def _frontier_categories(expr: str, registry=None) -> "list[str]":
    """Category frontier xuất hiện trong expr, theo THỨ TỰ ưu tiên ổn định của
    _NEUT_BY_CATEGORY (microstructure trước — đặc thù nhất). Rỗng = không phải frontier.

    parse_expression (lenient, chỉ kiểm cú pháp) thay vì parse strict: trích field không
    cần operator-tồn-tại/arity, và tiến trình gọi (vd `submit --power-pool`) có thể chưa
    import module đăng ký operator local — bug live 2026-07-18: strict ném ParseError
    'ts_rank' làm fallback mô tả câm lặng."""
    from src.lang.parser import parse_expression
    from src.lang.registry import default_registry
    from src.lang.visitors import FieldCollector

    reg = registry or default_registry()
    fields = FieldCollector(reg).visit(parse_expression(expr))
    cats = {FRONTIER_CATEGORY_BY_FIELD[f] for f in fields if f in FRONTIER_CATEGORY_BY_FIELD}
    return [c for c in _NEUT_BY_CATEGORY if c in cats]


def frontier_neutralization(expr: str, registry=None) -> "str | None":
    """Neutralization group-neut cho expr dùng field frontier; None nếu không phải frontier
    (caller — alt_data_seeds.neutralization_for_expr — giữ heuristic prefix cũ)."""
    cats = _frontier_categories(expr, registry)
    return _NEUT_BY_CATEGORY[cats[0]] if cats else None


def frontier_pp_choice(expr: str, registry=None) -> "str | None":
    """Risk-neutralization Power Pool cho expr frontier; STATISTICAL nếu category không có
    map riêng; None nếu không phải frontier."""
    cats = _frontier_categories(expr, registry)
    if not cats:
        return None
    return _PP_BY_CATEGORY.get(cats[0], "STATISTICAL")


# =========================== Hypothesis cấu trúc theo category =========================
# Hypothesis 4 phần vốn ghi ở COMMENT từng nhóm core phía trên — dạng đó con người đọc
# được nhưng selector Power Pool không dựng được mô tả Idea/Rationale (bắt buộc >=100 ký
# tự khi nộp pure PP). Bằng chứng 2026-07-18: LLdLVX0a (Sharpe 1.08, khớp theme) bị skip
# "thiếu mô tả" vì alphas.hypothesis = '{}' trên MỌI alpha đường sim-thẳng/near-miss.
# Nội dung TIẾNG ANH vì mô tả được PATCH thẳng lên Brain khi nộp. Ánh xạ vào mẫu
# build_power_pool_description: observation+background -> "Idea", implementation_spec ->
# "Rationale for data used", economic_rationale -> "Rationale for operators used".
# Hướng (dấu) viết TRUNG TÍNH vì core trong 1 category có thể follow hoặc fade.
from src.llm.hypothesis import Hypothesis  # noqa: E402 - sau registry để tránh vòng import

FRONTIER_HYPOTHESES: dict[str, Hypothesis] = {
    "insider": Hypothesis(
        observation="Corporate insiders buying with their own money is a strong information"
        " signal, while insider selling is often mere diversification (Lakonishok-Lee 2001).",
        background="Insiders hold private information about firm prospects; open-market net"
        " buying reveals conviction that is incorporated into prices only gradually.",
        economic_rationale="Sparse event-based insider fields are ts_backfill-ed (66d) so the"
        " latest filing keeps informing the signal between events; differencing buy versus"
        " sell activity isolates the direction of insider conviction.",
        implementation_spec="insider_trx_matrix transaction counts and USD signal values"
        " (total/top buy and sell counts, signal value tiers) aggregated as buy-minus-sell"
        " differences over a 66-day backfill window.",
    ),
    "call_filing": Hypothesis(
        observation="The tone of unscripted earnings-call answers and of 10-K/10-Q filings"
        " predicts returns: lying is costly when pressed by analysts (Larcker-Zakolyukina)"
        " and investors underreact to long documents (Loughran-McDonald).",
        background="Textual tone diffuses slowly into prices because processing long"
        " disclosures is costly; Q&A answers are harder to stage than prepared remarks.",
        economic_rationale="Positive-minus-negative chunk counts normalized by total chunks"
        " (+1 to avoid division by zero) build a bounded tone ratio; ts_backfill keeps the"
        " quarterly signal alive between calls or filings.",
        implementation_spec="earningscall_sentiment Q&A/summary/presentation chunk counts,"
        " insiders3 8-K/10-Q tone scores and filing_sentiment term counts, backfilled 66"
        " days to bridge quarterly reporting gaps.",
    ),
    "attention": Hypothesis(
        observation="Spikes in Google search interest for a company mark bursts of retail"
        " attention (Da-Engelberg-Gao, 'In Search of Attention').",
        background="Retail investors predominantly buy stocks that catch their attention,"
        " creating short-horizon price pressure that subsequently mean-reverts; persistently"
        " high attention marks overpricing.",
        economic_rationale="Ratios and differences of short versus long search-interest"
        " windows isolate attention SHOCKS from the attention level; ts_rank bounds the"
        " signal against extreme spikes.",
        implementation_spec="stock_search_trends corporate-name search interest over"
        " today/7d/28d/84d windows plus the search_interest normalized relative-interest"
        " score (vector field averaged via vec_avg).",
    ),
    "option_flow": Hypothesis(
        observation="Signed option volume split by account type (customer=retail,"
        " firm/professional=informed, broker-dealer=liquidity provider) carries distinct"
        " information about future stock returns (Pan-Poteshman 2006).",
        background="Informed traders prefer options for leverage; retail flow is"
        " systematically biased, and flow absorbed by liquidity providers marks short-term"
        " extremes that revert. Option-implied expected-move asymmetry carries the same"
        " forward-looking information (Bali-Cakici-Whitelaw lottery preference).",
        economic_rationale="ts_mean over one week smooths daily imbalance noise; rank or"
        " ts_rank converts flow into a bounded cross-sectional or temporal signal, and sign"
        " selection follows informed flow while fading retail/dealer-absorbed flow.",
        implementation_spec="order_flow_imb account-type volume imbalances"
        " (customer/firm/pro/broker-dealer) and expected_move upward-versus-downward"
        " lognormal move percentages and straddle-implied moves.",
    ),
    "microstructure": Hypothesis(
        observation="Order imbalance left unabsorbed at the closing auction (as %ADV)"
        " carries over into subsequent sessions.",
        background="Residual auction demand or supply cannot be fully absorbed at the"
        " close, so the imbalance predicts short-horizon drift in the same direction.",
        economic_rationale="vec_avg aggregates the auction imbalance vector into one daily"
        " cross-sectional value; ts_delta captures fresh imbalance shocks rather than the"
        " standing level.",
        implementation_spec="order_book_imbalance auction_order_imbalance_pct_adv, the"
        " closing-auction order imbalance normalized by average daily volume.",
    ),
    "ownership": Hypothesis(
        observation="Rising institutional ownership breadth predicts returns"
        " (Chen-Hong-Stein 2002), while very crowded institutional positions are prone to"
        " synchronized unwinds.",
        background="Breadth expansion reveals accumulating informed demand; crowding"
        " concentration raises fire-sale risk when funds de-lever together.",
        economic_rationale="Quarterly 13F fields are ts_backfill-ed (66d) to stay defined"
        " between filings; ts_delta captures ownership changes and ts_rank fades the most"
        " crowded names on a bounded scale.",
        implementation_spec="institutions18 holder counts, current-versus-prior holdings"
        " and percent-held from quarterly 13F filings, backfilled 66 days.",
    ),
    "fund_panel": Hypothesis(
        observation="A daily panel of fund holdings shows concentration (Herfindahl,"
        " crowding scores) and account-breadth changes in near real time.",
        background="High holding concentration means a few funds dominate the register and"
        " can trigger correlated unwinds; growing account breadth signals fresh inflows.",
        economic_rationale="vec_avg reduces per-fund vectors to one daily value; fading"
        " concentration (multiply -1, ts_rank) and following breadth growth (ts_delta)"
        " express the two mechanisms on a bounded scale.",
        implementation_spec="fund_holdings_panel daily holding-value distribution score,"
        " Herfindahl index of holdings, total holder accounts and top-weighted account"
        " numbers (vector fields).",
    ),
    "short_period": Hypothesis(
        observation="Published short interest arrives bi-weekly; model-predicted changes"
        " and surprises versus prediction lead the published number.",
        background="Short sellers are informed traders (Boehmer-Jones-Zhang): predicted"
        " short-interest builds and positive surprises precede underperformance.",
        economic_rationale="ts_backfill(22) bridges the bi-weekly reporting cycle;"
        " differencing surprise versus prior surprise isolates NEW short-pressure"
        " information, and multiply(-1) takes the side against rising short pressure.",
        implementation_spec="short_interest_pred predicted change, surprise ratio, prior"
        " surprise ratio and surprise amount around the bi-weekly short-interest cycle.",
    ),
    "short_daily": Hypothesis(
        observation="The daily fraction of trading volume executed as short sales (Reg SHO"
        " data) measures short-side pressure at daily frequency (Diether-Lee-Werner).",
        background="Daily short-sale ratios reveal informed short pressure much faster than"
        " bi-weekly short interest; rising ratios precede underperformance.",
        economic_rationale="Normalizing short volume by total volume (+1 to avoid division"
        " by zero) makes the pressure comparable across stocks; ts_sum/ts_delta windows"
        " capture pressure build-up versus its change.",
        implementation_spec="us_short_sale executed and reported short-sale share counts"
        " against aggregate/total trade share quantities, over 5-day sums or 22-day"
        " changes.",
    ),
}


def frontier_hypothesis(expr: str, registry=None) -> "Hypothesis | None":
    """Hypothesis cấu trúc của category frontier chứa field trong expr (ưu tiên theo thứ
    tự _NEUT_BY_CATEGORY như frontier_neutralization); None nếu expr không dùng field
    frontier. Nguồn dữ liệu cho selector Power Pool dựng mô tả khi alpha thiếu hypothesis
    riêng (đường sim-thẳng/near-miss ghi '{}')."""
    cats = _frontier_categories(expr, registry)
    return FRONTIER_HYPOTHESES.get(cats[0]) if cats else None
