# Nghiên cứu: SuperAlpha + Chất lượng nộp — hướng cải thiện Closed-loop AI+MiniBrain tool

Ngày: 2026-07-08
Nguồn: docs/worldquantbrain/docs/{superalpha, consultant-information, interpret-results}/*.md

## 1. Kỹ thuật/tiêu chí then chốt rút ra từ docs

1. **Ngưỡng nộp Delay-1 (không phải CHN)**: Sharpe > 1.58, Fitness > 1, turnover trong (1%, 70%), self-corr < 0.7 (hoặc Sharpe cao hơn ≥10% alpha trùng), sub-universe test, IS-Ladder test, weight test (max 1 mã <10%), bias test.

2. **IS-Ladder Sharpe test** — đây là rào cản thực sự khó, không chỉ "Sharpe > 1.58 toàn kỳ":
   - Test lặp từ N_YEARS=2 tăng dần tới 10. Ở N=2..5 PASS_THRESHOLD = 2.38 (D1), FAIL_THRESHOLD luôn = 1.58.
   - Nếu turnover < 30% thì PASS_THRESHOLD được nhân 0.85 (dễ pass hơn) — ví dụ 2.38×0.85=2.023 cho N=2-5. Đây là đòn bẩy quan trọng: turnover thấp không chỉ giúp fitness mà còn hạ ngưỡng ladder.
   - Alpha "tốt nhưng chỉ đủ Sharpe~1.4-1.6 toàn kỳ" gần như chắc chắn KHÔNG qua ladder ở các năm gần (2-5 năm) vì cần ≥2.38 (hoặc ≥2.02 nếu turnover<30%). Đây chính là lý do "trần Sharpe ~1.4-1.5" của tool hiện tại không đủ để nộp ổn định — số liệu PASSED 1.41 là biên rất mỏng, dễ rớt ladder ở simulation khác hoặc khi Brain re-run.

3. **Single Dataset Alpha**: nếu alpha chỉ dùng field của đúng 1 dataset (được phép thêm 6 grouping field: country/exchange/market/sector/industry/subindustry), thì KHÔNG cần qua toàn bộ IS-Ladder — chỉ cần Last-2Y Sharpe ≥ 2.38 (D1), cũng được nhân 0.85 nếu turnover<30%. Đây là con đường nộp "dễ hơn" rất đáng khai thác vì tool hiện dùng price/volume (thường đã là 1 dataset — pv1) — cần kiểm tra hiện engine có vô tình trộn thêm field khác (vd volume + fundamental) làm mất tư cách single-dataset hay không.

4. **Power Pool Alpha** — chuẩn dễ đạt hơn hẳn Regular:
   - Sharpe ≥ 1.0 (thấp hơn nhiều so với 1.58), số operator riêng biệt ≤ 8 (không tính ts_backfill/group_backfill), số datafield riêng biệt (trừ field group) ≤ 3, Power Pool correlation < 0.5, turnover PASS, sub-universe PASS.
   - Đây gần như "may đo" cho alpha đơn giản, ít field, ít operator — đúng kiểu alpha price/volume ngắn gọn mà tool đang sinh. Cần workflow riêng: khi alpha không đạt Regular (Sharpe 1.0-1.58) nhưng đạt các điều kiện Power Pool, vẫn có giá trị nộp (kèm mô tả Idea+Rationale ≥100 ký tự).
   - [Power Pool + ATOM]: nếu chỉ dùng 1 dataset và pass mọi test trừ IS-Ladder → cũng được tag kép, giá trị cao hơn Pure Power Pool.

5. **Fitness = Sharpe × sqrt(|Returns| / max(Turnover, 0.125))** — turnover thấp (nhưng không dưới 0.125 hiệu lực) và return cao đều kéo fitness lên; đây khớp với hướng LocalTuner đang tối ưu, nhưng công thức xác nhận: hạ turnover có lợi kép (tăng fitness VÀ hạ ngưỡng ladder nếu <30%).

6. **Nguyên tắc "Dos/Don'ts" cho consultant** (áp dụng để tránh overfit khi refine ở Brain thật):
   - Chỉ chọn tham số đơn giản (5/20/60/120/252 ngày...), không dò quá nhiều tham số/nhiều field.
   - Cải thiện bằng cách tinh chỉnh Ý TƯỞNG, không phải thêm điều kiện if-else/reversion vá lỗi.
   - Kiểm tra ổn định qua nhiều sub-universe (TOP500/1000/2000/3000) khi tune — đúng vấn đề panel local 478 mã hiện đang thiếu.
   - Không cố tình thêm noise để giảm correlation (rủi ro bị phát hiện/không đúng tinh thần).

7. **Sub-universe test công thức cụ thể** (TOPXXX): `subuniverse_sharpe >= 0.75 * sqrt(subuniverse_size/alpha_universe_size) * alpha_sharpe`. Có thể tính trước bằng proxy nếu biết Sharpe alpha và size subuniverse — gate hiện tại đã có proxy, nên đối chiếu công thức chính xác này để căn chỉnh hệ số 0.75.

## 2. Danh sách ưu tiên thay đổi cụ thể cho tool

### Ưu tiên cao nhất (đánh trúng "trần Sharpe 1.4-1.5" hiện tại)

**(A) Thêm IS-Ladder proxy vào Refiner/gate LOCAL trước khi sim Brain**
- Module: `LocalTuner` + `gate`.
- Hiện tại LocalTuner xếp hạng theo min(Sharpe/1.25, fitness/1.0) toàn kỳ — không mô phỏng ladder theo từng cửa sổ năm gần. Cần thêm bước: tính Sharpe rolling trên panel local cho 2-năm-gần-nhất (hoặc window ngắn nhất có thể xấp xỉ), so với ngưỡng 2.38 (hoặc 2.38×0.85 nếu turnover<30%). Loại bỏ sớm config có Sharpe toàn kỳ ổn nhưng Sharpe 2 năm gần yếu (dấu hiệu "decay theo thời gian" — đúng cảnh báo trong finding-consultant-alphas.md).
- Tác động kỳ vọng: giảm số alpha "PASSED biên" (như 1.41/1.04 hiện tại) dễ rớt khi Brain re-run test theo năm; tăng tỷ lệ pass thật khi nộp.

**(B) Ưu tiên nhánh turnover < 30% trong LocalTuner (không chỉ ≤0.70)**
- Module: `LocalTuner` (gate turnover hiện là ≤0.70).
- Thêm hệ số thưởng trong điểm xếp hạng khi turnover < 0.30 (vì được nhân 0.85 vào ngưỡng ladder + tăng fitness qua công thức Sharpe×sqrt(Returns/turnover)). Có thể thêm cấu hình decay lớn hơn / truncation chặt hơn để hạ turnover chủ động thay vì chỉ lọc sau.
- Tác động: cùng 1 Sharpe, alpha turnover<30% có ngưỡng ladder thấp hơn hẳn → dễ pass hơn nhiều so với chỉ tối ưu Sharpe/fitness đơn thuần.

**(C) Track "Single Dataset" tag + workflow bỏ qua IS-Ladder đầy đủ**
- Module: `seed`/`CuratedIdeaSource` + `refiner` + `gate`.
- Kiểm tra expression sinh ra chỉ dùng field pv1 (+ grouping fields được miễn) → gắn cờ single_dataset=True. Khi cờ này bật, gate chỉ cần check Last-2Y Sharpe ≥ 2.38 (×0.85 nếu turnover<30%) thay vì toàn bộ ladder — điều này nới lỏng đáng kể yêu cầu so với hiện tại (engine có thể đang áp gate ladder đầy đủ một cách không cần thiết cho alpha thuần price/volume).
- Tác động: alpha thuần PV hiện có mà tool coi là "chưa đạt ladder" thực ra có thể đã đủ điều kiện single-dataset — cần audit lại tiêu chí PASS hiện dùng trong refiner.

**(D) Thêm nhánh nộp Power Pool riêng khi Sharpe 1.0-1.58 (không đạt Regular)**
- Module: `refiner` + `set_alpha_properties` (đã có MCP tool) + `gate`.
- Với alpha Sharpe∈[1.0,1.58), fitness có thể thấp hơn 1 nhưng vẫn còn giá trị: kiểm tra điều kiện Power Pool (≤8 operator riêng biệt trừ ts_backfill/group_backfill, ≤3 datafield riêng biệt trừ grouping field, turnover PASS, sub-universe PASS). Nếu đạt, tự sinh mô tả Idea+Rationale ≥100 ký tự và đề xuất nộp dạng Power Pool thay vì loại bỏ.
- Tác động: mở thêm kênh giá trị cho phần lớn kết quả GP/LocalTuner hiện đang bị loại vì không đạt ngưỡng Regular — đặc biệt hợp với alpha price/volume đơn giản, ít field.

### Đòn bẩy lớn (cần đầu tư, giải quyết gốc rễ "panel 478 mã")

**(E) Mở rộng panel local lên gần TOP3000 (hoặc ít nhất TOP1000-1500)**
- Module: MiniBrain data layer / panel loader.
- Panel 478 mã hiện tại underestimate fitness và không cho phép test sub-universe đa cấp (TOP500/1000/2000/3000) như "Dos" khuyến nghị — đây là gốc rễ khiến LocalTuner không tự tin được về ladder/sub-universe trước khi tốn quota sim Brain. Đầu tư mở rộng panel là đòn bẩy lớn nhất để giảm sai số hiệu chỉnh Brain≈local×1.28 hiện đang phải bù bằng hệ số kinh nghiệm.

**(F) Multi-window rolling-Sharpe backtest trong LocalTuner thay vì 1 cửa sổ toàn kỳ**
- Module: `LocalTuner`.
- Chạy backtest trên nhiều sub-window lịch sử (không chỉ toàn kỳ) để ước lượng cả IS-Ladder proxy (A) lẫn độ ổn định qua thời gian (đúng tinh thần "OS Component Activation" và "reality check" của SuperAlpha docs: đổi từ đánh giá IS sang OS/kỳ gần để tránh overfit). Đầu tư hơn (A) vì cần tái cấu trúc backtest engine để trả về Sharpe theo từng năm/cửa sổ trượt, không chỉ 1 số tổng.

**(G) Bổ sung bộ đếm operator/datafield distinct cho mọi expression (hỗ trợ cả C và D)**
- Module: `seed`/expression builder.
- Cần một hàm tiện ích đếm operator riêng biệt (loại trừ ts_backfill/group_backfill) và datafield riêng biệt (loại trừ 6 grouping field) gắn vào mọi candidate alpha — nền tảng cho cả gate Power Pool (D) và gate Single Dataset (C). Đầu tư vừa phải nhưng là hạ tầng dùng chung.

## 3. Quick win vs Đòn bẩy lớn

**Quick win (rẻ, làm ngay):**
- (B) Thưởng điểm turnover<30% trong ranking LocalTuner — chỉ sửa công thức điểm, không cần dữ liệu mới.
- (G) Bộ đếm operator/datafield distinct — hàm thuần logic trên AST/expression string có sẵn.
- (C) Gắn cờ single_dataset dựa trên field prefix đã biết (pv1 vs fundamental) — logic đơn giản, tái dùng field metadata đã có.
- (D) Nhánh Power Pool khi Sharpe 1.0-1.58 — chủ yếu là thêm điều kiện rẽ nhánh + template mô tả, tái dùng gate/refiner sẵn có.

**Đòn bẩy lớn (cần đầu tư):**
- (A) IS-Ladder proxy theo rolling-window — cần backtest engine trả Sharpe theo từng năm/cửa sổ, không chỉ số tổng.
- (E) Mở rộng panel local lên gần TOP3000 — cần thu thập/lưu trữ dữ liệu giá nhiều mã hơn, tốn thời gian download/calibrate lại.
- (F) Multi-window rolling backtest toàn diện trong LocalTuner — phụ thuộc (A) và ảnh hưởng kiến trúc backtest.
