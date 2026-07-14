# Kho seed frontier — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nạp ~40 core alt-data mới (field đã verify live 2026-07-14) từ 12 dataset ít người đào vào đường sim-thẳng Brain sẵn có, kèm mapping neutralization và điểm cắm cho generator giai đoạn B.

**Architecture:** Module mới `src/generation/frontier_seeds.py` chứa cores + metadata field; hook mapping neutralization vào `alt_data_seeds`; wire một chỗ trong `build_closed_loop` qua helper `_gather_direct_cores`. Không đổi engine — mọi cơ chế (sim-thẳng, mini-sweep, saturation skip, field-guard) dùng lại nguyên.

**Tech Stack:** Python 3.12, pytest, sqlite (catalog `data_fields`), FASTEXPR parser nội bộ (`src/lang`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-frontier-seeds-design.md`.
- TDD bắt buộc: test trước, xem fail, code sau. Mỗi task 1 commit, message tiếng Việt.
- Code/comment tiếng Việt, giữ nguyên thuật ngữ kỹ thuật; giữ đúng dấu tiếng Việt.
- KHÔNG bịa field: chỉ dùng field trong danh sách FRONTIER_FIELDS của kế hoạch này
  (đã verify qua API `/data-fields` ngày 2026-07-14, lưu chứng cứ ở
  `logs/verified_fields_20260714.json` — Task 4 sẽ ghi file này).
- Field VECTOR bắt buộc bọc `vec_avg`/`vec_sum` TRƯỚC khi vào operator MATRIX
  (pre-filter simulator sẽ chặn `ts_backfill(vector_field, d)` — bằng chứng log 08:19 hôm nay).
- Field thưa (insider/call/filing/13F) bắt buộc `ts_backfill` (22 = biweekly/monthly, 66 = quarterly).
- Core trần depth ≤ 5 (MAX_DEPTH=7, chừa 2 mức cho tuner bọc vector_neut/rank).
- Chạy test: `venv\Scripts\python.exe -m pytest <file> -q` (Windows PowerShell, cwd = repo root).
- Test postgres `tests/test_db_postgres.py::test_make_engine_postgres_backend` fail sẵn do thiếu
  psycopg — không phải lỗi của thay đổi này, deselect khi chạy full suite.

---

### Task 1: Module `frontier_seeds.py` — cores + metadata

**Files:**
- Create: `src/generation/frontier_seeds.py`
- Test: `tests/test_frontier_seeds.py`

**Interfaces (Produces):**
- `FRONTIER_CORES: tuple[str, ...]` — ~40 core FASTEXPR trần.
- `FRONTIER_FIELDS: frozenset[str]` — mọi field alt-data dùng trong cores (đã verify live).
- `FRONTIER_VECTOR_FIELDS: frozenset[str]` — tập con field kiểu VECTOR.
- `FRONTIER_CATEGORY_BY_FIELD: dict[str, str]` — field → category
  (`insider | call_filing | attention | option_flow | microstructure | ownership | short`).

- [ ] **Step 1: Viết test fail**

```python
"""Kho seed frontier: core từ 12 dataset ít người đào, field verify live 2026-07-14.

Bất biến: parse được, depth trần ≤ 5 (chừa 2 mức cho wrapper tuner), field VECTOR phải
qua vec_avg/vec_sum, field thưa phải ts_backfill, mọi field nằm trong FRONTIER_FIELDS,
không trùng core với kho seed cũ (alt_data/fundamental/hypothesis)."""

from __future__ import annotations

from src.generation.alt_data_seeds import ALT_DATA_CORES
from src.generation.frontier_seeds import (
    FRONTIER_CATEGORY_BY_FIELD,
    FRONTIER_CORES,
    FRONTIER_FIELDS,
    FRONTIER_VECTOR_FIELDS,
)
from src.generation.fundamental_seeds import FUNDAMENTAL_CORES
from src.generation.hypothesis_seeds import HYPOTHESIS_CORES
from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import DepthVisitor, FieldCollector


def test_moi_core_parse_va_depth_tran_toi_da_5() -> None:
    for core in FRONTIER_CORES:
        node = parse(core)  # không được ném
        assert DepthVisitor().visit(node) <= 5, core


def test_so_luong_va_khong_trung_kho_cu() -> None:
    assert len(FRONTIER_CORES) >= 35
    assert len(set(FRONTIER_CORES)) == len(FRONTIER_CORES)
    cu = set(ALT_DATA_CORES) | set(FUNDAMENTAL_CORES) | set(HYPOTHESIS_CORES)
    assert not (set(FRONTIER_CORES) & cu)


def test_moi_field_trong_core_da_verify() -> None:
    reg = default_registry()
    for core in FRONTIER_CORES:
        for f in FieldCollector(reg).visit(parse(core)):
            assert f in FRONTIER_FIELDS, f"field chưa verify: {f} trong {core}"


def test_field_vector_phai_boc_vec_avg_hoac_vec_sum() -> None:
    for core in FRONTIER_CORES:
        for f in FRONTIER_VECTOR_FIELDS:
            if f in core:
                assert f"vec_avg({f})" in core or f"vec_sum({f})" in core, core


def test_field_thua_phai_ts_backfill() -> None:
    # Field sự kiện/quý (insider, call, filing, 13F, short-interest kỳ, search VECTOR):
    # thiếu ts_backfill là tín hiệu gần như toàn NaN (cardinal rule #3).
    sparse_cat = {"insider", "call_filing", "ownership", "short_period"}
    for core in FRONTIER_CORES:
        for f, cat in FRONTIER_CATEGORY_BY_FIELD.items():
            if f in core and cat in sparse_cat:
                assert "ts_backfill(" in core, f"thiếu ts_backfill cho {f}: {core}"


def test_moi_field_co_category() -> None:
    assert FRONTIER_FIELDS == set(FRONTIER_CATEGORY_BY_FIELD)
```

- [ ] **Step 2: Chạy test, xác nhận FAIL đúng lý do**

Run: `venv\Scripts\python.exe -m pytest tests\test_frontier_seeds.py -q`
Expected: FAIL/ERROR với `ModuleNotFoundError: No module named 'src.generation.frontier_seeds'`

- [ ] **Step 3: Viết `src/generation/frontier_seeds.py`**

Toàn bộ field dưới đây đã verify qua API `/data-fields` (USA/TOP3000/D1) ngày 2026-07-14.
Ghi chú kiểu field: (M) = MATRIX, (V) = VECTOR — VECTOR phải qua vec_avg/vec_sum.

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `venv\Scripts\python.exe -m pytest tests\test_frontier_seeds.py -q`
Expected: PASS toàn bộ. Nếu test depth fail ở core nào → đơn giản hoá đúng core đó
(bỏ một tầng transform), KHÔNG nới ngưỡng test.

- [ ] **Step 5: Commit**

```powershell
git add src/generation/frontier_seeds.py tests/test_frontier_seeds.py
git commit -m "feat(seeds): kho frontier ~40 core từ 12 dataset ít người đào (field verify live)"
```

---

### Task 2: Mapping neutralization cho field frontier

**Files:**
- Modify: `src/generation/frontier_seeds.py` (thêm 2 hàm cuối file)
- Modify: `src/generation/alt_data_seeds.py:78-94` (`neutralization_for_expr`) và `:106-125` (`pp_neutralization_for_expr`)
- Test: `tests/test_frontier_seeds.py` (thêm test)

**Interfaces:**
- Consumes: `FRONTIER_CATEGORY_BY_FIELD` (Task 1).
- Produces: `frontier_neutralization(expr, registry=None) -> str | None`;
  `frontier_pp_choice(expr, registry=None) -> str | None`.
  `None` = không có field frontier → caller giữ hành vi cũ.

- [ ] **Step 1: Viết test fail** (thêm vào `tests/test_frontier_seeds.py`)

```python
def test_frontier_neutralization_theo_category() -> None:
    from src.generation.frontier_seeds import frontier_neutralization

    # microstructure → MARKET; option_flow → SECTOR; ownership → INDUSTRY;
    # insider/call_filing/attention/short → SUBINDUSTRY; không field frontier → None.
    assert frontier_neutralization("vec_avg(auction_order_imbalance_pct_adv)") == "MARKET"
    assert frontier_neutralization("ts_mean(firm_vol_imbalance, 5)") == "SECTOR"
    assert frontier_neutralization(
        "ts_delta(ts_backfill(inst18_instownership_num_held, 66), 66)"
    ) == "INDUSTRY"
    assert frontier_neutralization(
        "ts_backfill(directional_indicator_score, 66)"
    ) == "SUBINDUSTRY"
    assert frontier_neutralization("rank(ts_delta(close, 5))") is None


def test_neutralization_for_expr_uu_tien_frontier() -> None:
    from src.generation.alt_data_seeds import neutralization_for_expr

    # Field frontier quyết định trước các heuristic prefix cũ.
    assert neutralization_for_expr("ts_mean(firm_vol_imbalance, 5)") == "SECTOR"
    # Hành vi cũ giữ nguyên khi không có field frontier.
    assert neutralization_for_expr(
        "multiply(-1, ts_mean(snt_social_value, 5))"
    ) == "SUBINDUSTRY"


def test_pp_choice_frontier() -> None:
    from src.generation.frontier_seeds import frontier_pp_choice

    # attention/call_filing → CROWDING; ownership/insider → SLOW; còn lại → STATISTICAL.
    assert frontier_pp_choice(
        "subtract(search_interest_today_corporate_name, search_interest_28d_corporate_name)"
    ) == "CROWDING"
    assert frontier_pp_choice(
        "ts_backfill(directional_indicator_score, 66)"
    ) == "SLOW"
    assert frontier_pp_choice("ts_mean(firm_vol_imbalance, 5)") == "STATISTICAL"
    assert frontier_pp_choice("rank(ts_delta(close, 5))") is None
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv\Scripts\python.exe -m pytest tests\test_frontier_seeds.py -q`
Expected: FAIL với `ImportError: cannot import name 'frontier_neutralization'`

- [ ] **Step 3: Thêm 2 hàm vào cuối `frontier_seeds.py`**

```python
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
    _NEUT_BY_CATEGORY (microstructure trước — đặc thù nhất). Rỗng = không phải frontier."""
    from src.lang.parser import parse
    from src.lang.registry import default_registry
    from src.lang.visitors import FieldCollector

    reg = registry or default_registry()
    fields = FieldCollector(reg).visit(parse(expr))
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
```

Sửa `alt_data_seeds.neutralization_for_expr` — thêm 3 dòng ĐẦU hàm (trước `reg = ...`):

```python
    # Field frontier (kho seed 2026-07-14) quyết định trước heuristic prefix cũ —
    # tránh va chạm tên (vd field earnings-call bắt đầu bằng "count_" không match prefix nào).
    from src.generation.frontier_seeds import frontier_neutralization

    _fn = frontier_neutralization(expr, registry)
    if _fn is not None:
        return _fn
```

Sửa `alt_data_seeds.pp_neutralization_for_expr` — thêm tương tự ĐẦU hàm (trước `reg = ...`):

```python
    from src.generation.frontier_seeds import frontier_pp_choice

    _fp = frontier_pp_choice(expr, registry)
    if _fp is not None:
        choice = _fp
        if not allowed:
            return "STATISTICAL"
        return choice if choice in allowed else sorted(allowed)[0]
```

(Import đặt TRONG hàm để tránh vòng import: `frontier_seeds` không import `alt_data_seeds`,
nhưng giữ quy ước module seed độc lập.)

- [ ] **Step 4: Chạy test, xác nhận PASS + không vỡ test cũ**

Run: `venv\Scripts\python.exe -m pytest tests\test_frontier_seeds.py tests\test_alt_data_seeds.py -q`
(nếu không có `tests\test_alt_data_seeds.py` thì chạy `-k alt_data` để tìm test liên quan)
Expected: PASS toàn bộ.

- [ ] **Step 5: Commit**

```powershell
git add src/generation/frontier_seeds.py src/generation/alt_data_seeds.py tests/test_frontier_seeds.py
git commit -m "feat(seeds): mapping neutralization theo category frontier (group-neut + Power Pool)"
```

---

### Task 3: Wire vào build_closed_loop qua helper `_gather_direct_cores`

**Files:**
- Modify: `src/app/closed_loop_adapters.py:1165-1173` (đoạn gom `direct_cores` trong `build_closed_loop`)
- Test: `tests/test_frontier_wire.py`

**Interfaces:**
- Produces: `_gather_direct_cores(include_alt_data, include_fundamental, include_hypothesis, include_frontier) -> tuple[str, ...]` (module-level, test được không cần dựng cả loop); tham số mới `include_frontier: bool = True` trên `build_closed_loop`.

- [ ] **Step 1: Viết test fail**

```python
"""Wire kho frontier vào build_closed_loop: cores frontier phải nằm trong direct_cores
(đường sim-thẳng AltDataIdeaSource) và tắt được qua cờ include_frontier."""

from __future__ import annotations

import inspect

from src.app.closed_loop_adapters import _gather_direct_cores, build_closed_loop
from src.generation.alt_data_seeds import ALT_DATA_CORES
from src.generation.frontier_seeds import FRONTIER_CORES


def test_gather_gom_du_cac_kho_theo_co() -> None:
    all_on = _gather_direct_cores(True, True, True, True)
    assert set(FRONTIER_CORES) <= set(all_on)
    assert set(ALT_DATA_CORES) <= set(all_on)
    # Tắt frontier -> không còn core frontier, kho cũ giữ nguyên.
    no_frontier = _gather_direct_cores(True, True, True, False)
    assert not (set(FRONTIER_CORES) & set(no_frontier))
    assert set(ALT_DATA_CORES) <= set(no_frontier)


def test_build_closed_loop_co_tham_so_include_frontier_mac_dinh_bat() -> None:
    sig = inspect.signature(build_closed_loop)
    assert "include_frontier" in sig.parameters
    assert sig.parameters["include_frontier"].default is True
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `venv\Scripts\python.exe -m pytest tests\test_frontier_wire.py -q`
Expected: FAIL với `ImportError: cannot import name '_gather_direct_cores'`

- [ ] **Step 3: Sửa `closed_loop_adapters.py`**

Thêm import ở đầu file (cạnh import FUNDAMENTAL_CORES/HYPOTHESIS_CORES hiện có, ~dòng 32):

```python
from src.generation.frontier_seeds import FRONTIER_CORES
```

Thêm helper module-level (đặt ngay TRÊN `build_closed_loop`):

```python
def _gather_direct_cores(
    include_alt_data: bool, include_fundamental: bool,
    include_hypothesis: bool, include_frontier: bool,
) -> tuple[str, ...]:
    """Gom core đường sim-thẳng theo cờ — tách hàm để test wire không cần dựng cả loop.
    Frontier đặt SAU kho cũ: kho cũ đã bão hoà (saturation skip bỏ qua nhanh), frontier
    là nguồn mới chiếm phần lớn batch đầu."""
    cores: tuple[str, ...] = ()
    if include_alt_data:
        cores += ALT_DATA_CORES
    if include_fundamental:
        cores += FUNDAMENTAL_CORES
    if include_hypothesis:
        cores += HYPOTHESIS_CORES
    if include_frontier:
        cores += FRONTIER_CORES
    return cores
```

Trong `build_closed_loop`: thêm tham số `include_frontier: bool = True` vào chữ ký
(cạnh `include_alt_data`/`include_hypothesis` hiện có), rồi THAY khối gom cores
(hiện tại dòng 1165-1173):

```python
    direct_cores: tuple[str, ...] = _gather_direct_cores(
        include_alt_data, include_fundamental, include_hypothesis, include_frontier,
    )
```

(Giữ nguyên comment khối cũ phía trên đoạn này; field-guard `known_fields` phía
AltDataIdeaSource tự lọc field chưa cache — không cần thêm guard mới.)

- [ ] **Step 4: Chạy test wire + toàn bộ test adapters, xác nhận PASS**

Run: `venv\Scripts\python.exe -m pytest tests\test_frontier_wire.py -q`
Run: `venv\Scripts\python.exe -m pytest tests -q -k "adapter or closed_loop" `
Expected: PASS toàn bộ.

- [ ] **Step 5: Commit**

```powershell
git add src/app/closed_loop_adapters.py tests/test_frontier_wire.py
git commit -m "feat(closed-loop): wire kho frontier vào direct_cores (cờ include_frontier, mặc định bật)"
```

---

### Task 4: Script verify field với catalog DB + nghiệm thu suite

**Files:**
- Create: `tools/verify_frontier_fields.py`
- Test: không test riêng (script vận hành, logic verify là 3 dòng SQL) — nghiệm thu bằng chạy thật.

**Interfaces:**
- Consumes: `FRONTIER_FIELDS`, `FRONTIER_CORES` (Task 1); DB catalog `data_fields` (bảng `id`).
- Produces: exit code 0 nếu 100% field có trong catalog; in danh sách field THIẾU nếu không;
  ghi bằng chứng ra `logs/verified_fields_<YYYYMMDD>.json`.

- [ ] **Step 1: Viết script**

```python
# -*- coding: utf-8 -*-
"""Đối chiếu 100% field của FRONTIER_CORES với catalog DB thật (bảng data_fields).

Chạy TRƯỚC khi merge / sau khi load-fields:  venv\\Scripts\\python.exe tools\\verify_frontier_fields.py
Exit 0 = đủ hết; exit 1 = có field thiếu (in danh sách — cấm merge cho tới khi xử lý).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.frontier_seeds import FRONTIER_CORES, FRONTIER_FIELDS  # noqa: E402
from src.storage.db import make_engine  # noqa: E402


def main() -> int:
    engine = make_engine()
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, type, dataset_id FROM data_fields")).fetchall()
    catalog = {r[0]: {"type": r[1], "dataset": r[2]} for r in rows}
    thieu = sorted(f for f in FRONTIER_FIELDS if f not in catalog)
    co = {f: catalog[f] for f in sorted(FRONTIER_FIELDS & set(catalog))}
    out = {
        "ngay": date.today().isoformat(), "n_cores": len(FRONTIER_CORES),
        "n_fields": len(FRONTIER_FIELDS), "thieu": thieu, "co": co,
    }
    dest = Path("logs") / f"verified_fields_{date.today().strftime('%Y%m%d')}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Field frontier: {len(FRONTIER_FIELDS)} | thiếu trong catalog: {len(thieu)}")
    for f in thieu:
        print("  THIẾU:", f)
    print("Bằng chứng:", dest)
    return 1 if thieu else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Chạy script với DB thật, xác nhận 0 thiếu**

Run: `venv\Scripts\python.exe tools\verify_frontier_fields.py`
Expected: `thiếu trong catalog: 0`, exit 0. (Tiền đề: DB `wq_alpha_tuananhpo13_gmail_com.db`
đã load-fields — xác nhận 2026-07-14: catalog có 85.612 field, 12/12 field mẫu CÓ.)
Nếu có field thiếu → chạy menu 2 (tải fields) rồi chạy lại; vẫn thiếu → sửa/bỏ core dùng
field đó (cấm giữ field chưa verify).

- [ ] **Step 3: Chạy toàn bộ suite**

Run: `venv\Scripts\python.exe -m pytest -q --deselect tests/test_db_postgres.py::test_make_engine_postgres_backend`
Expected: PASS toàn bộ (baseline 2026-07-14: 1460 passed + 3 test reseed; cộng thêm test mới).

- [ ] **Step 4: Commit**

```powershell
git add tools/verify_frontier_fields.py logs/verified_fields_*.json
git commit -m "chore(seeds): script verify field frontier với catalog DB + bằng chứng verify"
```

---

## Nghiệm thu cuối (USER chạy — không tự động)

1. `venv\Scripts\python.exe tools\verify_frontier_fields.py` → 0 thiếu.
2. Chạy menu 5: batch đầu phải thấy core frontier được phục vụ (log AltDataIdeaSource,
   không bị field-guard chặn hàng loạt), có sim Brain thật từ ≥ 5 dataset mới.
3. Sau vài phiên: so tỷ lệ sim Sharpe > 0.5 nhóm frontier vs nhóm GP
   (baseline 2026-07-14: GP max 0.54, đa số < 0.2).
