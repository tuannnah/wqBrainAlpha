"""Alt-data seed cores — đường ĐI THẲNG tới Brain sim (bỏ qua chấm local).

Bối cảnh: panel local (`market_yf`) CHỈ có price/volume → mọi alpha dùng dataset thay thế
không chấm/tune local được và bị refiner mặc định giết ở `local_floor`. Nhưng đúng theo docs
WQ ("alpha tốt sống trong GAP/GATE/RESIDUAL, không phải LEVEL") và feedback độ-độc-đáo, đòn
bẩy chất lượng lớn nhất là MỞ RỘNG khỏi họ price/volume đã bão hòa (pool bị mine một họ VWAP
reversal → alpha mới trùng self-corr). Module này cung cấp các core alt-data trên field ĐÃ
XÁC MINH LIVE (get_datafields USA/TOP3000/D1, 2026-07-09) để đưa thẳng lên Brain sim.

Nguyên tắc:
- CHỈ field đã verify live: option8 (`implied_volatility_*`, `historical_volatility_*`),
  socialmedia8 (`snt_social_value`, `snt_social_volume`). KHÔNG dùng field chưa xác nhận
  (opt6_*/pcr_*/snt1_* — dataset option6/sentiment1 KHÔNG có cho account này) — cardinal rule #1.
- Cấu trúc GAP (hiệu hai chuỗi đồng họ) + reversal/attention-scaling, KHÔNG phải LEVEL.
- `ts_backfill` cho field option sparse (coverage ~0.97) — cardinal rule #3.
- Neutralization CHỌN THEO CATEGORY dataset (docs `advanced-topics/neutralization.md`):
  price/volume/option → MARKET/SECTOR; fundamental/analyst → INDUSTRY;
  news/social/sentiment → SUBINDUSTRY.
"""

from __future__ import annotations

from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import FieldCollector

# Mỗi core: signal core THUẦN (không wrapper neutralization/decay — Brain áp qua settings).
# Kèm giả thuyết kinh tế 1 dòng (nền tảng học thuật) — bắt buộc để nộp/Power Pool mô tả được.
# XEN KẼ option8 ↔ socialmedia8 để phiên ngắn (--max-ideas nhỏ) vẫn chạm CẢ HAI nguồn.
ALT_DATA_CORES: tuple[str, ...] = (
    # [option8] Skew put-call (fear): put IV > call IV = phòng thủ downside đắt → mean-revert, fade.
    "multiply(-1, subtract(ts_backfill(implied_volatility_put_30, 22), "
    "ts_backfill(implied_volatility_call_30, 22)))",
    # [socialmedia8] Fade mức sentiment xã hội (hype bán lẻ mean-revert).
    "multiply(-1, ts_mean(snt_social_value, 5))",
    # [option8] Variance risk premium: IV ẩn >> vol thực = bảo hiểm đắt → thường overpriced, fade.
    "multiply(-1, subtract(ts_backfill(implied_volatility_mean_30, 22), "
    "ts_backfill(historical_volatility_30, 22)))",
    # [socialmedia8] Fade THAY ĐỔI sentiment gần đây (overextension ngắn hạn).
    "multiply(-1, ts_delta(snt_social_value, 5))",
    # [option8] Độ dốc term-structure IV: 90d − 30d; contango (dương) ổn định hơn backwardation.
    "subtract(ts_backfill(implied_volatility_mean_90, 22), "
    "ts_backfill(implied_volatility_mean_30, 22))",
    # [socialmedia8] Sentiment fade khuếch đại theo chú ý (tweet-volume rank cao) — GATE xấp xỉ.
    "multiply(-1, multiply(ts_mean(snt_social_value, 5), ts_rank(snt_social_volume, 22)))",
)

# Tiền tố field → category dataset. Dùng để chọn neutralization (docs WQ).
_OPTION_PREFIXES = ("implied_volatility", "historical_volatility", "opt", "pcr")
_SOCIAL_PREFIXES = ("snt_", "snt1", "scl", "nws", "event_")
_SOCIAL_SUBSTR = ("sentiment", "social", "novelty", "buzz")
_ANALYST_PREFIXES = ("anl", "est", "fnd", "is_", "bs_", "cf_")
# Field fundamental (income/balance/cashflow) verify LIVE trên fundamental6 — neutralize theo
# INDUSTRY (docs WQ). Nhận diện qua tên field chuẩn hoá + tiền tố fnd6_.
_FUNDAMENTAL_FIELDS = frozenset({
    "assets", "cashflow_op", "revenue", "sales", "operating_income", "operating_expense",
    "return_assets", "sales_growth", "cash", "cash_st", "inventory", "sales_ps",
})
_FUNDAMENTAL_PREFIXES = ("fnd6_", "cashflow_")


def _is_fundamental(f: str) -> bool:
    return f in _FUNDAMENTAL_FIELDS or f.startswith(_FUNDAMENTAL_PREFIXES)


def _is_option(f: str) -> bool:
    return f.startswith(_OPTION_PREFIXES)


def _is_social(f: str) -> bool:
    return f.startswith(_SOCIAL_PREFIXES) or any(s in f for s in _SOCIAL_SUBSTR)


def _is_analyst(f: str) -> bool:
    return f.startswith(_ANALYST_PREFIXES)


def neutralization_for_expr(expr: str, registry=None) -> str:
    """Chọn neutralization theo category dataset của field alt-data trong biểu thức.

    Ưu tiên: option → SECTOR; news/social/sentiment → SUBINDUSTRY; analyst/fundamental →
    INDUSTRY; mặc định SUBINDUSTRY (WQ default, an toàn). Dùng ở nhánh sim-thẳng của refiner
    để Brain neutralize đúng nhóm cho từng nguồn dữ liệu (không hardcode một giá trị)."""
    reg = registry or default_registry()
    fields = FieldCollector(reg).visit(parse(expr))
    if any(_is_option(f) for f in fields):
        return "SECTOR"
    if any(_is_social(f) for f in fields):
        return "SUBINDUSTRY"
    if any(_is_analyst(f) for f in fields):
        return "INDUSTRY"
    if any(_is_fundamental(f) for f in fields):
        return "INDUSTRY"
    return "SUBINDUSTRY"


# Map category dataset -> neutralization RỦI RO ưu tiên (Power Pool Theme chỉ cho risk-neut).
# Khác `neutralization_for_expr` (group-neut cho đường non-PP).
_PP_CATEGORY_DEFAULT = {
    "option": "STATISTICAL",
    "social": "CROWDING",
    "analyst": "SLOW",
}


def pp_neutralization_for_expr(expr: str, allowed: frozenset[str], registry=None) -> str:
    """Chọn 1 neutralization RỦI RO cho biểu thức alt-data theo category dataset, GIAO với tập
    `allowed` của theme. option→STATISTICAL, social/sentiment→CROWDING, analyst/fundamental→SLOW,
    price-derived/mặc định→STATISTICAL. Lựa chọn không thuộc `allowed` ->
    phần tử đầu (sorted, ổn định) của `allowed`. `allowed` rỗng -> STATISTICAL (an toàn chung)."""
    reg = registry or default_registry()
    fields = FieldCollector(reg).visit(parse(expr))
    if any(_is_option(f) for f in fields):
        choice = _PP_CATEGORY_DEFAULT["option"]
    elif any(_is_social(f) for f in fields):
        choice = _PP_CATEGORY_DEFAULT["social"]
    elif any(_is_analyst(f) for f in fields):
        choice = _PP_CATEGORY_DEFAULT["analyst"]
    else:
        choice = "STATISTICAL"
    if not allowed:
        return "STATISTICAL"
    if choice in allowed:
        return choice
    return sorted(allowed)[0]


def pp_neut_candidates(
    expr: str, allowed: frozenset[str], registry=None, sweep: bool = False
) -> list[str]:
    """Danh sách neutralization để refiner sim. Mặc định 1× (chỉ lựa chọn map theo category);
    `sweep=True` -> toàn bộ `allowed` (sorted ổn định) để quét con giữ config tốt nhất."""
    if sweep and allowed:
        return sorted(allowed)
    return [pp_neutralization_for_expr(expr, allowed, registry)]
