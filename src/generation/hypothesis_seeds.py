"""Hypothesis-driven seed cores cho các HỌ NHÂN TỐ MỚI — mở khỏi 5 cụm đã có (pv_reversal,
momentum, fundamental value/quality đơn-ratio, option IV skew, social sentiment) để bớt trùng
self-corr khi pool đã đông (RC1/RC2 fix idea-generator).

QUAN TRỌNG (cardinal rule #1, không có phiên auth để verify live tại thời điểm viết module
này): field analyst4/short-interest dưới đây là TÊN TỐT NHẤT SUY RA từ quy ước đặt tên đã
verify trong `src/generation/families.py` (anl4_afv4_*, anl4_af_*) và tài liệu dataset
`shortinterest30` (`docs/worldquantbrain/docs/advanced-topics/fast-d1-documentation.md`) —
CHƯA xác nhận live cho account này. Đây chính xác là lý do field-validity guard ở
`closed_loop_adapters.build_closed_loop(known_fields=...)` tồn tại: nếu field sai, core bị
LỌC BỎ + log trước khi chạm Brain (không đốt quota/sai dấu như bug cũ). Field fundamental
(operating_income/assets/sales_growth) ĐÃ verify live (khớp `fundamental_seeds.FUNDAMENTAL_FIELDS`).

Mỗi core: signal THUẦN (không group_neutralize/scale/decay — Brain áp qua sim settings,
xem `neutralization_for_expr`). Cấu trúc GAP/RATIO/CONDITIONING, không LEVEL đơn thuần.
"""

from __future__ import annotations

from src.lang.parser import parse
from src.lang.registry import default_registry
from src.lang.visitors import FieldCollector

# --- Family "analyst_revision": EPS/CFPS consensus estimate revision momentum -----------------
# Chan-Jegadeesh-Lakonishok (1996) "Momentum Strategies": điều chỉnh dự báo đồng thuận của
# analyst khuếch tán chậm vào giá -> thay đổi ước tính gần đây dự báo lợi suất tiếp theo.
# Field anl4_afv4_eps_mean / anl4_afv4_cfps_mean: quy ước đặt tên analyst4 (families.py:99).
_ANALYST_REVISION_CORES: tuple[str, ...] = (
    # Thay đổi ước tính EPS đồng thuận ~1 tháng (revision momentum), long khi ước tính tăng.
    "ts_delta(ts_backfill(anl4_afv4_eps_mean, 66), 20)",
    # Thay đổi ước tính dòng tiền/cổ phần đồng thuận ~1 tháng — cùng cơ chế, field khác (đa
    # dạng nguồn trong cùng họ để giảm trùng self-corr chéo).
    "ts_delta(ts_backfill(anl4_afv4_cfps_mean, 66), 20)",
)

# --- Family "short_interest": bán khống / days-to-cover ---------------------------------------
# Asquith-Pathak-Ritter (2005) "Short interest, institutional ownership and stock returns":
# short-seller là nhà đầu tư CÓ THÔNG TIN -> short interest/days-to-cover cao dự báo lợi suất
# THẤP hơn -> fade (dấu âm). Boehmer-Jones-Zhang (2008): THAY ĐỔI short interest cũng có tín
# hiệu (short-selling gia tăng gần đây dự báo underperform mạnh hơn mức tĩnh).
_SHORT_INTEREST_CORES: tuple[str, ...] = (
    # Mức days-to-cover cao (nhiều ngày mới đóng hết vị thế short) -> fade.
    "multiply(-1, ts_backfill(days_to_cover, 66))",
    # Thay đổi số cổ phần bị short ~1 tháng -> short interest TĂNG nhanh -> fade mạnh hơn.
    "multiply(-1, ts_delta(ts_backfill(shares_short, 66), 20))",
)

# --- Family "earnings_drift": PEAD (post-earnings-announcement drift) qua surprise -----------
# Bernard-Thomas (1989/1990): thị trường phản ứng KHÔNG ĐỦ với earnings surprise, giá tiếp tục
# trôi (drift) theo hướng surprise trong ~60 ngày giao dịch sau công bố. Surprise đo bằng GAP
# giữa EPS thực tế (anl4_af_eps_value) và ước tính đồng thuận (anl4_afv4_eps_mean) — KHÁC
# analyst_revision (đo THAY ĐỔI ước tính, không so với thực tế).
_EARNINGS_DRIFT_CORES: tuple[str, ...] = (
    # Surprise GAP thô: EPS thực tế trừ đồng thuận, long khi vượt kỳ vọng.
    "subtract(ts_backfill(anl4_af_eps_value, 66), ts_backfill(anl4_afv4_eps_mean, 66))",
    # Surprise làm mượt ~40 phiên (~2 tháng) để bắt đúng cửa sổ drift Bernard-Thomas thay vì
    # nhiễu 1 phiên công bố.
    "ts_mean(subtract(ts_backfill(anl4_af_eps_value, 66), "
    "ts_backfill(anl4_afv4_eps_mean, 66)), 40)",
)

# --- Family "value_quality": profitability CONDITIONED theo tăng trưởng (không LEVEL đơn) ----
# Novy-Marx (2013) quality + Asness-Frazzini-Pedersen (2019) "Quality Minus Junk": phần bù
# quality mạnh nhất ở nhóm công ty KHÔNG tăng trưởng nóng (tăng trưởng nóng thường đi kèm định
# giá cao/rủi ro đảo ngược) -> CONDITIONING (nhân rank) thay vì 1 ratio đơn (khác hẳn cấu trúc
# FUNDAMENTAL_CORES hiện có — mỗi core ở đó chỉ 1 field, đây là tương tác 2 field).
_VALUE_QUALITY_CORES: tuple[str, ...] = (
    "multiply("
    "rank(divide(ts_backfill(operating_income, 66), ts_backfill(assets, 66))), "
    "rank(multiply(-1, ts_backfill(sales_growth, 66))))",
)

# Gộp toàn bộ core mới — thứ tự XEN KẼ family để phiên ngắn (--max-ideas nhỏ) vẫn chạm nhiều
# họ orthogonal thay vì cạn quota trong 1 họ duy nhất (giống nguyên tắc ALT_DATA_CORES).
HYPOTHESIS_CORES: tuple[str, ...] = (
    _ANALYST_REVISION_CORES[0],
    _SHORT_INTEREST_CORES[0],
    _EARNINGS_DRIFT_CORES[0],
    _VALUE_QUALITY_CORES[0],
    _ANALYST_REVISION_CORES[1],
    _SHORT_INTEREST_CORES[1],
    _EARNINGS_DRIFT_CORES[1],
)

# Field xuất hiện trong HYPOTHESIS_CORES — dùng để test/tra cứu; KHÔNG có nghĩa "đã verify
# live" (khác FUNDAMENTAL_FIELDS) — chỉ operating_income/assets/sales_growth (value_quality)
# là verify live thật; phần còn lại chờ field-validity guard (known_fields) tự lọc nếu sai.
HYPOTHESIS_FIELDS: frozenset[str] = frozenset({
    "anl4_afv4_eps_mean", "anl4_afv4_cfps_mean",
    "days_to_cover", "shares_short",
    "anl4_af_eps_value",
    "operating_income", "assets", "sales_growth",
})

# Field ĐÃ verify live (khớp fundamental_seeds.FUNDAMENTAL_FIELDS) — subset của HYPOTHESIS_FIELDS.
_VERIFIED_LIVE_FIELDS: frozenset[str] = frozenset({"operating_income", "assets", "sales_growth"})


def hypothesis_fields_in(expr: str, registry=None) -> set[str]:
    """Tập field tham chiếu trong 1 core hypothesis — tiện cho test/log guard."""
    reg = registry or default_registry()
    return FieldCollector(reg).visit(parse(expr))
