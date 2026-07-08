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
ALT_DATA_CORES: tuple[str, ...] = (
    # --- option8: bề mặt vol ẩn (crash-fear / variance risk premium / term-structure) ---
    # Skew put-call (fear): put IV cao hơn call IV = phòng thủ downside đắt → mean-revert, fade.
    "multiply(-1, subtract(ts_backfill(implied_volatility_put_30, 22), "
    "ts_backfill(implied_volatility_call_30, 22)))",
    # Variance risk premium: IV ẩn >> vol thực = bảo hiểm đắt → thường overpriced, fade.
    "multiply(-1, subtract(ts_backfill(implied_volatility_mean_30, 22), "
    "ts_backfill(historical_volatility_30, 22)))",
    # Độ dốc term-structure IV: 90d − 30d; contango (dương) ổn định hơn backwardation (stress).
    "subtract(ts_backfill(implied_volatility_mean_90, 22), "
    "ts_backfill(implied_volatility_mean_30, 22))",
    # --- socialmedia8: chú ý/tin đồn bán lẻ (attention-driven mispricing) ---
    # Fade mức sentiment xã hội (hype bán lẻ mean-revert).
    "multiply(-1, ts_mean(snt_social_value, 5))",
    # Fade THAY ĐỔI sentiment gần đây (overextension ngắn hạn).
    "multiply(-1, ts_delta(snt_social_value, 5))",
    # Sentiment fade khuếch đại theo chú ý (tweet-volume rank cao) — xấp xỉ GATE bằng scaling.
    "multiply(-1, multiply(ts_mean(snt_social_value, 5), ts_rank(snt_social_volume, 22)))",
)

# Tiền tố field → category dataset. Dùng để chọn neutralization (docs WQ).
_OPTION_PREFIXES = ("implied_volatility", "historical_volatility", "opt", "pcr")
_SOCIAL_PREFIXES = ("snt_", "snt1", "scl", "nws", "event_")
_SOCIAL_SUBSTR = ("sentiment", "social", "novelty", "buzz")
_ANALYST_PREFIXES = ("anl", "est", "fnd", "is_", "bs_", "cf_")


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
    return "SUBINDUSTRY"
