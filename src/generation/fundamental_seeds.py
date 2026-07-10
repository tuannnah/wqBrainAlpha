"""Fundamental seed cores — đường ĐI THẲNG Brain sim (như alt-data, field ngoài panel local).

IMPROVEMENT_SPEC §2.1: mở khỏi cụm PV/VWAP bão hòa bằng họ fundamental có nền tảng học thuật
(gross-profitability, cash-flow yield, asset growth, quality). Field ĐÃ VERIFY LIVE qua
get_datafields (fundamental6, USA/TOP3000/delay1, 2026-07-10) — KHÔNG bịa (cardinal rule #1).

Mọi field fundamental sparse (coverage ~0.5, cập nhật theo quý) BẮT BUỘC ts_backfill để lấp
NaN giữa các kỳ báo cáo, nếu không alpha chết (cardinal rule #3). Cấu trúc RATIO/GROWTH (yield,
tăng trưởng) chứ không LEVEL. Neutralization -> INDUSTRY (docs WQ, chọn qua neutralization_for_expr).

Nền tảng học thuật:
- Gross-profitability (Novy-Marx 2013): lợi nhuận gộp/tài sản dự báo lợi suất chéo.
- Cash-flow yield (Sloan 1996 accruals): dòng tiền thực > lợi nhuận kế toán về chất lượng.
- Asset growth (Cooper-Gulen-Schill 2008): công ty bành trướng tài sản nhanh -> lợi suất thấp (fade).
"""

from __future__ import annotations

# Field fundamental đã verify LIVE (get_datafields fundamental6). Giữ khớp với
# alt_data_seeds._FUNDAMENTAL_FIELDS (dùng cho neutralization INDUSTRY).
FUNDAMENTAL_FIELDS: frozenset[str] = frozenset({
    "assets", "cashflow_op", "revenue", "operating_income", "sales_growth",
})

# Mỗi core: signal THUẦN (Brain áp neutralization/decay qua settings). Ratio chuẩn hoá theo
# assets để so sánh chéo công ty; ts_backfill(field, 66) ≈ 1 quý giao dịch lấp NaN.
FUNDAMENTAL_CORES: tuple[str, ...] = (
    # Gross-profitability proxy (Novy-Marx): operating_income / assets — quality dài hạn, long.
    "divide(ts_backfill(operating_income, 66), ts_backfill(assets, 66))",
    # Cash-flow yield (Sloan): operating cash flow / assets — dòng tiền thực trên tài sản, long.
    "divide(ts_backfill(cashflow_op, 66), ts_backfill(assets, 66))",
    # Asset growth (Cooper-Gulen-Schill): tăng trưởng tài sản 1 năm -> FADE (nhân -1).
    "multiply(-1, divide(ts_delta(ts_backfill(assets, 66), 250), ts_backfill(assets, 66)))",
    # Accruals gap: dòng tiền vượt lợi nhuận kế toán = chất lượng cao (cashflow_op − operating_income)/assets.
    "divide(subtract(ts_backfill(cashflow_op, 66), ts_backfill(operating_income, 66)), "
    "ts_backfill(assets, 66))",
    # Sales growth momentum (fundamental): tăng trưởng doanh thu quý -> long (đã là ratio sẵn).
    "ts_backfill(sales_growth, 66)",
)
