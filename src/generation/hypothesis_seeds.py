"""Hypothesis-driven seed cores cho các HỌ NHÂN TỐ MỚI — mở khỏi 5 cụm đã có (pv_reversal,
momentum, fundamental value/quality đơn-ratio, option IV skew, social sentiment) để bớt trùng
self-corr khi pool đã đông (RC1/RC2 fix idea-generator).

QUAN TRỌNG (cardinal rule #1): TOÀN BỘ field trong module này đã verify LIVE 2026-07-14 qua
`tools/verify_datasets.py` (`logs/verified_fields_20260714.json`) — analyst4 (anl4_afv4_*,
anl4_af_*), securities lending (shortinterest3: loan_utilization_ratio/mean_loan_rate),
SI surprise (short_interest_pred: short_interest_surprise_ratio) và fundamental
(operating_income/assets/sales_growth). Tên short-interest SUY ĐOÁN cũ (days_to_cover/
shares_short) account KHÔNG có -> field-validity guard (`build_closed_loop(known_fields=...)`)
chặn cả họ suốt — đã thay bằng field có thật ở trên. Guard vẫn giữ nguyên vai trò: field nào
không nằm trong CATALOG CACHE của account (data_fields:USA:TOP3000:1) vẫn bị lọc + log; nếu
cache tải trước 14/07 thiếu field shortinterest3 thì chạy menu 2 (tải lại fields) trước menu 5.

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

# --- Family "short_interest": cho vay chứng khoán (securities lending) + SI surprise ---------
# Field verify LIVE 2026-07-14 qua tools/verify_datasets.py (logs/verified_fields_20260714.json):
# tên suy đoán cũ days_to_cover/shares_short account KHÔNG có (field guard chặn cả họ suốt) —
# thay bằng shortinterest3 (securities lending, coverage 1.0, daily) và short_interest_pred
# (coverage 0.9987, theo kỳ công bố SI ~2 tuần).
#
# Hypothesis 4 phần cho từng core:
# (1) Fade utilization — [quan sát] loan_utilization_ratio = cầu vay / cung cho vay chứng
#     khoán; [nền tảng] Cohen-Diether-Malloy (2007) "Supply and Demand Shifts in the Shorting
#     Market" + Asquith-Pathak-Ritter (2005); [cơ chế] short-seller là nhà đầu tư CÓ THÔNG
#     TIN — cầu vay chiếm gần hết cung = bear crowding có conviction -> lợi suất tương lai
#     THẤP; [đặc tả] fade mức utilization làm mượt ~1 tháng (22 phiên), dấu âm.
# (2) Borrow-cost momentum — [quan sát] mean_loan_rate = phí vay chứng khoán; [nền tảng]
#     Jones-Lamont (2002) "Short-sale constraints and stock returns" + Drechsler-Drechsler
#     (2014) shorting premium; [cơ chế] phí vay TĂNG nhanh = cầu short mới với conviction cao,
#     tin xấu chưa phản ánh hết vào giá -> tiếp tục underperform; [đặc tả] fade thay đổi phí
#     ~1 tháng (ts_delta 22).
# (3) SI surprise — [quan sát] short_interest_surprise_ratio = SI công bố vượt mức dự đoán;
#     [nền tảng] Boehmer-Jones-Zhang (2008): THAY ĐỔI/bất ngờ short interest có tín hiệu mạnh
#     hơn mức tĩnh; [cơ chế] SI bất ngờ cao = thông tin bear MỚI chưa được giá hấp thụ ->
#     drift âm sau công bố; [đặc tả] fade surprise, backfill 66 giữ tín hiệu sống giữa các kỳ
#     công bố (~2 tuần/kỳ, cùng quy ước event-seed 66 của earnings_drift).
_SHORT_INTEREST_CORES: tuple[str, ...] = (
    # (1) shortinterest3 daily coverage 1.0 — backfill NGẮN 5 chỉ vá lỗ dữ liệu cục bộ,
    # ts_mean 22 làm mượt mức utilization (signal chậm, smooth không giết edge).
    "multiply(-1, ts_mean(ts_backfill(loan_utilization_ratio, 5), 22))",
    # (2) Thay đổi phí vay ~1 tháng — cấu trúc DELTA (không phải LEVEL đơn thuần).
    "multiply(-1, ts_delta(ts_backfill(mean_loan_rate, 5), 22))",
    # (3) Surprise ratio đã là cấu trúc GAP (thực tế vs dự đoán) do dataset tính sẵn.
    "multiply(-1, ts_backfill(short_interest_surprise_ratio, 66))",
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
    _SHORT_INTEREST_CORES[2],
)

# Field xuất hiện trong HYPOTHESIS_CORES — TOÀN BỘ đã verify LIVE 2026-07-14 qua
# tools/verify_datasets.py (logs/verified_fields_20260714.json): analyst4 (anl4_*),
# securities lending (shortinterest3), SI surprise (short_interest_pred), fundamental
# (fundamental6). LƯU Ý: verify live ≠ có trong catalog cache DB — field guard (known_fields)
# đọc cache data_fields:USA:TOP3000:1; nếu cache cũ hơn 14/07 thiếu field shortinterest3 thì
# cần tải lại fields (menu 2 run.bat) để seed (1)(2) không bị lọc oan.
HYPOTHESIS_FIELDS: frozenset[str] = frozenset({
    "anl4_afv4_eps_mean", "anl4_afv4_cfps_mean",
    "loan_utilization_ratio", "mean_loan_rate", "short_interest_surprise_ratio",
    "anl4_af_eps_value",
    "operating_income", "assets", "sales_growth",
})

# Field ĐÃ verify live — sau lần verify 2026-07-14, TRÙNG với HYPOTHESIS_FIELDS (giữ tên
# riêng cho tương thích: trước 14/07 đây là subset fundamental duy nhất đã verify).
_VERIFIED_LIVE_FIELDS: frozenset[str] = HYPOTHESIS_FIELDS


def hypothesis_fields_in(expr: str, registry=None) -> set[str]:
    """Tập field tham chiếu trong 1 core hypothesis — tiện cho test/log guard."""
    reg = registry or default_registry()
    return FieldCollector(reg).visit(parse(expr))
