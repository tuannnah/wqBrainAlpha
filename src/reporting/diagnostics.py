"""Helper chẩn đoán Pha 0 (IMPROVEMENT_SPEC §3): chuẩn hoá lý do thất bại thành mã fail_check
ổn định, và suy nhãn family (họ nhân tố) từ field/cấu trúc biểu thức.

- fail_check_from_reasons: reasons của hard_filter là free-text ("sharpe 0.40 < 0.5"); map về
  mã cố định (LOW_SHARPE/LOW_FITNESS/HIGH_TURNOVER/HIGH_DRAWDOWN/UNKNOWN) để phân bố tự động.
- classify_family: nhãn để family-aware budget (Pha 2) + phân bố summary. Suy heuristic từ tên
  field xuất hiện trong chuỗi (không cần parse) — đủ tốt để nhóm; refiner có thể override."""

from __future__ import annotations

# Thứ tự = độ nghiêm trọng: reason đầu khớp được chọn (sharpe trước fitness trước turnover).
_FAIL_CHECK_RULES: tuple[tuple[str, str], ...] = (
    ("sharpe", "LOW_SHARPE"),
    ("fitness", "LOW_FITNESS"),
    ("turnover", "HIGH_TURNOVER"),
    ("drawdown", "HIGH_DRAWDOWN"),
)


def fail_check_from_reasons(reasons: list[str]) -> str:
    """Reasons hard_filter -> 1 mã fail_check (theo thứ tự nghiêm trọng). Rỗng -> "";
    không khớp luật nào -> UNKNOWN (đừng mất thông tin thất bại)."""
    if not reasons:
        return ""
    for keyword, code in _FAIL_CHECK_RULES:
        for r in reasons:
            if keyword in r.lower():
                return code
    return "UNKNOWN"


# Nhận diện family theo dấu hiệu field (khớp substring, ưu tiên từ đặc trưng nhất trước).
def classify_family(expr: str) -> str:
    """Suy họ nhân tố từ biểu thức (heuristic substring). Trả một trong:
    options_iv / news_social / analyst / fundamental / pv_reversal / momentum / other."""
    e = expr.lower()

    def has(*subs: str) -> bool:
        return any(s in e for s in subs)

    if has("implied_volatility", "historical_volatility", "_iv", "put_", "call_"):
        return "options_iv"
    if has("snt_social", "social_value", "social_volume", "news", "buzz"):
        return "news_social"
    if has("earningsrevision", "netearnings", "analyst", "estimate", "rating", "recommendation"):
        return "analyst"
    if has("ebit", "assets", "cashflow", "book", "revenue", "dividend", "ts_backfill"):
        return "fundamental"
    # PV: có vwap/open (reversal intraday) hoặc close+ts_mean (mean-reversion) => pv_reversal.
    if has("vwap") or ("open" in e and "ts_mean" in e) or ("close" in e and "ts_mean" in e):
        return "pv_reversal"
    # Directional trên giá/khối lượng => momentum.
    if has("close", "volume", "returns", "high", "low") and "ts_delta" in e:
        return "momentum"
    if has("close", "volume", "returns", "high", "low"):
        return "pv_reversal"
    return "other"
