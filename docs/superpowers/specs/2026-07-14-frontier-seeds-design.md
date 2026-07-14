# Thiết kế: Kho seed "frontier" từ dataset ít người đào (giai đoạn A) + khung mở rộng (B)

**Ngày:** 2026-07-14
**Trạng thái:** Đã duyệt (user chốt hướng C: A trước, chừa điểm cắm cho B)

## Bối cảnh & vấn đề

Phiên chạy menu 5 ngày 2026-07-14 cho thấy nút thắt của engine không phải thuật toán
mà là **nguồn ý tưởng đầu vào quá hẹp**:

- Kho seed tuyển chọn chỉ có 19 core (6 alt-data + 5 fundamental + 8 hypothesis) và
  **toàn bộ đã sim** (log: "đã sim & bão hoà, bỏ phục vụ lại").
- Chỉ còn GP gánh → sinh toàn momentum/pv_reversal trên 6 field giá/khối lượng yfinance;
  11/19 ứng viên chết ở sàn local, 4 sim Brain chỉ đạt Sharpe 0.06–0.54. Họ price/volume
  đã bão hòa trên TOP3000 (skill WQ: factor kinh điển hiếm khi qua ngưỡng standalone).
- ρ(local↔Brain) ≈ 0.31 → bộ lọc local gần như mù; combiner đói nguyên liệu vì không
  có tín hiệu khá để ghép.

Trong khi đó account có quyền dùng **299 dataset** trên USA/TOP3000/D1 (quét API
2026-07-14), engine mới khai thác ~6. Nhiều dataset gần như chưa ai đào (users thấp)
→ self-corr thấp, đúng chỗ dễ qua gate nộp — nguồn seed alt-data từng là đường ra
alpha tốt nhất của tool (VWAP reversal Sharpe 1.49 Brain).

## Mục tiêu

- Nạp **~50 core mới** (mỗi dataset 4–6 core) từ **~10 dataset ít người dùng**, mỗi core
  có căn cứ kinh tế 4 phần (quan sát → cơ chế → hiệu ứng kỳ vọng → công thức).
- Mọi field trong core **được verify với catalog thật** trước khi vào kho (cardinal
  rule #1 — không bịa field, không đốt quota oan).
- Không đổi kiến trúc engine: seed mới chảy qua đúng đường alt-data direct-sim sẵn có
  (sim thẳng Brain, mini-sweep, saturation skip, avoid-list, field-guard).
- Chừa **điểm cắm cho giai đoạn B** (generator template) mà không cần refactor.

**Không nằm trong phạm vi (YAGNI):** generator tự động (B) — chỉ chừa chỗ; thay đổi
GP/calibration/combiner; đường nộp Power Pool tự động.

## Dataset nhắm đến (ứng viên, chốt sau khi verify field)

Tiêu chí chọn: users < ~350 (ít cạnh tranh → self-corr thấp), cơ sở kinh tế rõ, field
MATRIX (hoặc VECTOR xử lý được bằng vec_avg/vec_sum), coverage đủ trên TOP3000.

| Chủ đề kinh tế | Dataset ứng viên | Cơ chế kỳ vọng |
|---|---|---|
| Insider giao dịch | insiders3, insider_trx_matrix | Insider mua ròng → thông tin nội bộ tích cực |
| NLP earnings call/filing | earningscall_sentiment, filing_sentiment | Giọng điệu quản trị dự báo revision |
| Chú ý nhà đầu tư | stock_search_trends, web_traffic_engage | Search/traffic đột biến → áp lực mua lẻ ngắn hạn |
| Tuyển dụng | hiring_trends, other335 | Mở rộng tuyển dụng → tăng trưởng tương lai |
| Vi cấu trúc | order_book_imbalance, order_flow_imb | Mất cân bằng lệnh → drift ngắn hạn |
| Option kỳ vọng | expected_move | Expected vs realized move → risk premium |
| Sở hữu tổ chức | institutions18, fund_holdings_panel | Dòng tiền quỹ → momentum sở hữu |
| Short interest mở rộng | short_interest_pred, us_short_sale | Dự báo short tăng → tín hiệu âm |

Dataset nào field không dùng được (toàn VECTOR khó xử lý, coverage thấp, field không
tải được) → loại, lấy dataset dự bị cùng chủ đề. Quy trình verify: tải field từng
dataset qua API (`/data-fields?dataset.id=...`), lưu vào DB catalog, đối chiếu 100%
field trong core với catalog trước khi merge.

## Kiến trúc

- **Module mới `src/generation/frontier_seeds.py`:**
  - `FRONTIER_CORES: tuple[str, ...]` — core FASTEXPR thô (chưa wrapper), nhóm theo
    dataset, mỗi nhóm có comment hypothesis 4 phần.
  - `FRONTIER_NEUTRALIZATION: dict` — map dataset/prefix field → neutralization
    (insider/sentiment → SUBINDUSTRY; microstructure/option → MARKET; theo docs WQ).
  - Điểm cắm B: sau này thêm `generate_frontier_cores(catalog) -> tuple[str, ...]`
    cùng module, nối cùng chỗ wire — không đổi kiến trúc.
- **Wire:** `closed_loop_adapters.py` chỗ gom `direct_cores` (hiện ~dòng 1167):
  `direct_cores += FRONTIER_CORES`. Nhờ đó tự hưởng: đường `_sim_direct` (sim thẳng
  Brain khi `local_usable=False`), mini-sweep flip-dấu/decay (budget 2), saturation
  skip, avoid-list, field-guard chặn field ngoài catalog.
- **`neutralization_for_expr`:** mở rộng nhận diện field frontier → chọn neutralization
  theo `FRONTIER_NEUTRALIZATION`; không khớp → giữ hành vi cũ.

## Ràng buộc cấu trúc core (theo skill WQ Brain)

- Core trần (chưa wrapper) depth đủ thấp để cộng stack scale/decay/neut không vượt ~7.
- Field thưa (insider/fundamental/quarterly) bắt buộc `ts_backfill(field, d)` với d
  theo chu kỳ báo cáo (22/66).
- Field VECTOR bắt buộc bọc `vec_avg`/`vec_sum` trước khi vào operator MATRIX.
- Ưu tiên bounded (`ts_rank`) hơn unbounded (`ts_zscore`) cho tín hiệu regime-nhạy.
- Không smooth tín hiệu nhanh (microstructure) — turnover cao là bản chất tín hiệu.

## Kỷ luật quota

Không đổi cơ chế: mỗi core 1 sim + sweep ≤ 2 → 50 core ≈ 50–150 sim, tự dàn qua nhiều
ngày nhờ saturation skip (core đã sim không phục vụ lại phiên sau). GP giữ cap 3
sim/vòng — quota dồn cho seed mới.

## Kiểm thử (TDD)

1. **Cấu trúc:** mọi core trong `FRONTIER_CORES` parse được (AST), depth trần ≤ ngưỡng,
   field VECTOR có vec_avg/vec_sum, field thưa có ts_backfill.
2. **Mapping:** `neutralization_for_expr` trả đúng neutralization cho core mỗi nhóm;
   expr không khớp giữ hành vi cũ.
3. **Wire:** FRONTIER_CORES nằm trong direct_cores; field-guard lọc core có field
   ngoài catalog (test với catalog giả thiếu field).
4. **Verify field trước merge:** script đối chiếu 100% field của FRONTIER_CORES với
   catalog DB thật (chạy sau khi load-fields các dataset mới); CI/test bỏ qua nếu
   không có DB thật (guard như pattern test hiện có).

## Tiêu chí nghiệm thu

- `pytest` xanh toàn bộ; script verify field 0 field lạ.
- Chạy menu 5 thật: log cho thấy core frontier được phục vụ (không bị field-guard
  chặn hàng loạt), có sim Brain thật từ ≥ 5 dataset mới.
- Theo dõi qua vài phiên: tỷ lệ sim đạt Sharpe > 0.5 của nhóm frontier cao hơn nhóm GP
  (baseline phiên 2026-07-14: GP max 0.54, đa số < 0.2).
