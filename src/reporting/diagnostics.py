"""Helper chẩn đoán Pha 0 (IMPROVEMENT_SPEC §3): chuẩn hoá lý do thất bại thành mã fail_check
ổn định, và suy nhãn family (họ nhân tố) từ field/cấu trúc biểu thức.

- fail_check_from_reasons: reasons của hard_filter là free-text ("sharpe 0.40 < 0.5"); map về
  mã cố định (LOW_SHARPE/LOW_FITNESS/HIGH_TURNOVER/HIGH_DRAWDOWN/UNKNOWN) để phân bố tự động.
  Cùng bảng luật cũng phủ reason tiền-kiểm (PreFilter.check) — Task 3 (spec C2): "Operator
  không tồn tại"/"Field/hằng không tồn tại"/"Độ sâu >"/"Số node >"/"Parse lỗi"/"ngoặc".
- categorize_presim_reason: bọc fail_check_from_reasons riêng cho reason pre-sim reject (chưa
  chạm Brain) -> OPERATOR_INVALID/FIELD_INVALID/DEPTH/PARSE; không khớp -> PRESIM_REJECT.
- classify_family: nhãn để family-aware budget (Pha 2) + phân bố summary. Suy heuristic từ tên
  field xuất hiện trong chuỗi (không cần parse) — đủ tốt để nhóm; refiner có thể override."""

from __future__ import annotations

# Field frontier (spec 2026-07-14-frontier-seeds-design.md) — import module-level: an toàn vì
# frontier_seeds KHÔNG import ngược diagnostics (không vòng import). Dùng để classify_family
# nhận diện category frontier trước khi rơi vào các rule catch-all bên dưới (xem finding F1).
from src.generation.frontier_seeds import FRONTIER_CATEGORY_BY_FIELD

# Thứ tự = độ nghiêm trọng: reason đầu khớp được chọn (sharpe trước fitness trước turnover).
_FAIL_CHECK_RULES: tuple[tuple[str, str], ...] = (
    ("sharpe", "LOW_SHARPE"),
    ("fitness", "LOW_FITNESS"),
    ("turnover", "HIGH_TURNOVER"),
    ("drawdown", "HIGH_DRAWDOWN"),
    # Reason tiền-kiểm (PreFilter.check, CHƯA chạm Brain) — Task 3 (spec C2): phân biệt bug
    # operator/field bịa khỏi sim thật rớt sharpe thấp (đừng gộp cả 2 vào LOW_SHARPE/UNKNOWN).
    ("operator", "OPERATOR_INVALID"),
    ("field", "FIELD_INVALID"),
    ("độ sâu", "DEPTH"),
    ("số node", "DEPTH"),
    ("parse", "PARSE"),
    ("ngoặc", "PARSE"),
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


def categorize_presim_reason(reason: str) -> str:
    """Phân loại reason pre-sim reject (`PreFilter.check`, `SimulationResult.presim_reason`)
    thành mã ổn định: OPERATOR_INVALID/FIELD_INVALID/DEPTH/PARSE. Dùng chung bảng luật với
    `fail_check_from_reasons` (không nhân đôi) — không khớp luật nào -> PRESIM_REJECT (đừng
    mất thông tin, khác UNKNOWN vốn dành cho reason sim thật)."""
    code = fail_check_from_reasons([reason])
    return code if code and code != "UNKNOWN" else "PRESIM_REJECT"


# Nhận diện family theo dấu hiệu field (khớp substring, ưu tiên từ đặc trưng nhất trước).
def classify_family(expr: str) -> str:
    """Suy họ nhân tố từ biểu thức (heuristic substring). Trả một trong:
    options_iv / news_social / earnings_drift / analyst_revision / short_interest /
    value_quality / analyst / fundamental / pv_reversal / momentum / other."""
    e = expr.lower()

    def has(*subs: str) -> bool:
        return any(s in e for s in subs)

    if has("implied_volatility", "historical_volatility", "_iv", "put_", "call_"):
        return "options_iv"
    if has("snt_social", "social_value", "social_volume", "news", "buzz"):
        return "news_social"
    # Family MỚI (hypothesis_seeds, Task hàng đợi RC1/RC2): kiểm TRƯỚC nhánh analyst/
    # fundamental CHUNG bên dưới để không bị nuốt vào bucket cũ — family-budget/saturation
    # cần nhãn RIÊNG mới coi các họ này là orthogonal (không lẫn pv_reversal/analyst/fundamental).
    if has("eps_value"):
        # Earnings surprise (actual vs consensus) -> PEAD/earnings-drift, KHÁC analyst_revision
        # (revision thuần, không có "actual") — kiểm trước để "eps_value" không rơi nhầm nhánh dưới.
        return "earnings_drift"
    if has("afv4_eps_mean", "afv4_cfps_mean"):
        return "analyst_revision"
    # Field verify LIVE 2026-07-14 (logs/verified_fields_20260714.json): securities lending
    # (shortinterest3: loan_utilization_ratio/mean_loan_rate) + SI surprise (short_interest_pred).
    # Kiểm TRƯỚC nhánh fundamental bên dưới — nếu không, ts_backfill trong core sẽ nuốt các
    # field cho vay chứng khoán vào bucket "fundamental" sai. Giữ days_to_cover/shares_short
    # trong luật để alpha lịch sử (nếu có) vẫn phân loại đúng họ.
    if has("days_to_cover", "shares_short", "short_interest", "loan_utilization", "loan_rate"):
        return "short_interest"
    # F1 (review final feature/frontier-seeds): 40 core FRONTIER (frontier_seeds.py, field verify
    # live 2026-07-14) — 18/40 core có ts_backfill sẽ rơi chung vào "fundamental" ở rule substring
    # "ts_backfill" bên dưới nếu không chặn TRƯỚC ở đây; family-budget (max_per_family=8, đóng họ
    # khi 0 pass — closed_loop.py:426-431) sẽ đóng oan sau 8 core đầu, bỏ phí quota presim đã đốt
    # cho các core frontier còn lại đội lốt "fundamental". Tra field trực tiếp trong
    # FRONTIER_CATEGORY_BY_FIELD (khớp SUBSTRING tên field trong expr, cùng phong cách `has()`
    # phía trên — hàm này vốn không parse AST) — mỗi category trả họ riêng, tiền tố "frontier_"
    # để không va tên họ cũ (vd "short_interest"). ĐẶT SAU rule short_interest ngay trên: category
    # "short_period" tái dùng field short_interest_pred TRÙNG field rule short_interest đã test từ
    # trước — giữ nguyên family cũ cho các field đó (cùng ý nghĩa nghiệp vụ), KHÔNG đổi tên thành
    # "frontier_short_period".
    for field, cat in FRONTIER_CATEGORY_BY_FIELD.items():
        if field in e:
            return f"frontier_{cat}"
    if has("operating_income") and has("sales_growth"):
        # Conditioning quality x growth (2 field fundamental cùng lúc) -> khác hẳn 1 core
        # single-ratio của FUNDAMENTAL_CORES (chỉ 1 trong 2 field này, không cả hai).
        return "value_quality"
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
