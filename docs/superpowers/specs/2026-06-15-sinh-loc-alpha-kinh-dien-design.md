# Thiết kế: Sinh & lọc alpha kinh điển → log chi tiết

Ngày: 2026-06-15
Scope mô phỏng: USA / TOP3000 / delay=1 (đã cache 8599 fields, 67 operators).

## Mục tiêu
Claude tự research các họ alpha kinh điển, biến đổi thành biểu thức FASTEXPR
hợp lệ, chọn lọc bằng bộ lọc local (thuật toán + tài liệu), rồi log chi tiết
từng alpha kèm setting đạt chuẩn ra file `.txt` để user tự mô phỏng trên WQ Brain.

## Ràng buộc đặc thù
User tự mô phỏng → **không có metric backtest** (sharpe/fitness/turnover) ở bước
sinh. Do đó `filter.py`/`scorer.py` (cần số liệu backtest) KHÔNG dùng ở đây. Việc
"chọn lọc theo thuật toán" chạy bằng bộ lọc local thuần cấu trúc đã có test.

## Luồng
```
Research theo họ (Claude/subagent song song) → ứng viên (giả thuyết + FASTEXPR)
  → bộ lọc local → xếp hạng + đảm bảo đa dạng họ → log .txt chi tiết
```

## Sinh ứng viên
7 họ kinh điển, mỗi họ nhiều biến thể (đổi field/cửa sổ/neutralization):
Reversal, Momentum, Volatility, Volume/Liquidity, Fundamental value,
Analyst revisions, Seasonality. Tổng ~200-300 ứng viên thô.

Nguyên liệu field thật đã xác nhận tồn tại trong DB:
- PV: close, open, high, low, volume, vwap, returns, cap, adv20, sharesout, dividend
- Fundamental: assets, equity, ebit, ebitda, revenue, cashflow, debt, eps, cogs,
  capex, current_ratio, bookvalue_ps, enterprise_value, cash
- Analyst: anl4_afv4_eps_mean, anl4_af_eps_value, anl4_afv4_sales_mean, ...

## Bộ lọc local (tái dùng code đã test)
1. `PreFilter.check` — **cửa cứng**: cân ngoặc, parse được, depth ≤ trần,
   node ≤ trần, operator ∈ 67 thật, field ∈ 8599 thật.
2. `complexity_penalty` ∈ [0,1] — phạt mềm độ phức tạp.
3. `similarity_ratio` so zoo Alpha101 — loại trùng cấu trúc (ngưỡng ~0.8).
4. Khử trùng nội bộ giữa ứng viên đôi một (ngưỡng ~0.85).
5. Điểm local (không backtest): `originality*0.6 + (1-complexity)*0.4`,
   cộng quota mỗi họ để output cân bằng đa dạng.

## Output
`output/alphas_<ngày>.txt`. Mỗi alpha: mã, họ, giả thuyết + lý giải kinh tế,
biểu thức FASTEXPR, **setting đầy đủ** (region/universe/delay/decay/
neutralization/truncation/pasteurization/unitHandling/nanHandling/language),
điểm local + lý do đạt chuẩn. Giữ lại ~60-100 alpha đạt chuẩn.

## TDD
Code mới = `src/generation/local_select.py` (điểm local + xếp hạng + quota đa dạng)
và formatter log. Viết test trước. Tái dùng PreFilter/complexity/similarity.
Script CLI: `scripts/generate_alphas.py`.
