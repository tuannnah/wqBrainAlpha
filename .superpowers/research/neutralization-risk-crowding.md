# Nghiên cứu: Neutralization / Risk / Crowding — hướng nâng fitness, giảm self-corr, tăng robustness

Ngày: 2026-07-08. Phạm vi: 8 docs advanced-topics + discover-brain (vector_neut), đối chiếu code
hiện tại (`src/backtest/config.py`, `src/backtest/local_tuner.py`, `src/operators_local/*`,
`src/llm/refiner.py`, `src/scoring/power_pool_theme.py`).

## 1. Tóm tắt 6 kỹ thuật từ docs

| Kỹ thuật | Cơ chế | Tác động chính |
|---|---|---|
| **vector_neut** (discover-brain) | Trừ phần chiếu của x lên y: x* = x − (x·y/y·y)·y, trực giao hoá alpha với 1 vector risk factor (vd beta thị trường, momentum) | Giảm phơi nhiễm 1 factor cụ thể → có thể nâng Sharpe/giảm drawdown |
| **Double neutralization** | `group_neutralize(alpha, densify(group_cartesian_product(g1, g2)))` — neutralize theo TÍCH của 2 group (vd sector × sta1_top1000c50), KHÁC với neutralize 2 lần tuần tự (lần 2 phá một phần lần 1) | Docs: "một số alpha cải thiện hiệu năng VÀ giảm correlation với alpha đã nộp" |
| **Risk-neutralized (Slow/Fast/Slow+Fast)** | Setting `neutralization` mới ở tầng Brain, hồi quy alpha lên basket factor phong cách (Fama-French style) ẩn công thức (confidential) | Alpha "risk-neutralized" → phần dư (residual) không giải thích được bởi factor chung → độc đáo hơn, ít bị coi là "tái tạo factor cũ" |
| **RAM** (Reversion And Momentum) | `neutralization="REVERSION_AND_MOMENTUM"` — factor phản ứng ngắn hạn (5 ngày) + momentum dài hạn (12 tháng, cumulative excess return) | Giảm phơi nhiễm crowded momentum/reversion trades, cải thiện Sharpe/drawdown, đặc biệt hữu ích để bóc ảnh hưởng của "price" khỏi alpha fundamental |
| **Statistical** | `neutralization="STATISTICAL"` — risk factor rút ra bằng PCA/cluster trên historical returns (không phải factor kinh tế học đặt tên sẵn) | Bổ sung góc nhìn khác Slow/Fast (vốn là fundamental-style); tăng đa dạng, giữ hiệu năng mà giảm rủi ro |
| **Crowding** | `neutralization="CROWDING"` — risk factor đo mức độ tập trung vị thế giống nhiều nhà đầu tư khác (bao gồm cả momentum-driven crowding) | Trực tiếp nhắm vào **self-correlation / Power Pool crowding** — đúng vấn đề tool đang gặp (self_corr 0.51, cần ≤0.5) |
| Investability-constrained metrics / Max Trade | Bật `Max Trade=ON` trong settings → PnL dưới ràng buộc thanh khoản (ADV) | Chỉ số phụ để đánh giá capacity/robustness sau chi phí; không phải neutralization nhưng liên quan robustness |
| Fast D1 | Field suffix `_fast_d1` (overnight data), submit chỉ khi > D1 thường | Không liên quan neutralization; cơ hội dataset ít người khai thác → self-corr thấp tự nhiên nếu dùng, nhưng KHÔNG phải trọng tâm price/volume hiện tại |

**Phát hiện quan trọng liên quan trực tiếp tool**: file `src/scoring/power_pool_theme.py` (dòng
8-13) đã ghi chú: filter Power Pool Theme tuần 29/6-5/7/2026 chứa cụm nguyên văn
`"neutralization in (slow, fast, slow and fast, ram, statistical, crowding)"` — tức **theme Power
Pool GẦN ĐÂY đòi hỏi chính xác các risk-neutralization này**, không phải optional. Cụm này hiện bị
để `unparsed_constraints` (không dùng để pass/fail). Đây là bằng chứng cụ thể tool cần hỗ trợ các
neutralization này, không chỉ là "nice to have" lý thuyết.

## 2. Đối chiếu với code hiện tại — cái gì local eval được, cái gì chỉ chạy trên Brain

- `src/backtest/config.py::Neutralization` chỉ có `NONE/MARKET/SECTOR/INDUSTRY/SUBINDUSTRY`.
  Brain hỗ trợ thêm `SLOW/FAST/SLOW_AND_FAST/RAM(REVERSION_AND_MOMENTUM)/STATISTICAL/CROWDING`
  — **CHƯA có trong enum, chưa gửi được trong settings khi refine → cần thêm**.
- `local_tuner.py` dòng 61-62 CHỦ Ý chỉ sweep `(MARKET, SECTOR)` vì "panel có group sector" —
  đã kiểm tra: `INDUSTRY/SUBINDUSTRY` có trong enum nhưng KHÔNG có nguồn dữ liệu group nào khác
  ngoài `sector` trong loader/panel (`src/backtest/portfolio.py` map string nhưng không dataset
  nào cấp industry/subindustry cho panel 478 mã) → xác nhận industry/subindustry vẫn Brain-only
  cho tới khi panel có thêm group field.
- `src/operators_local/neutralization.py` đã có **regression_neut** và **vector_neut** implement
  đúng công thức trong doc discover-brain (residual OLS 1 biến / trừ hình chiếu), nhưng
  `gp_usable=False` — GP không tự chèn, chỉ được LLM refiner gợi ý dùng cho mục tiêu `pool_fit`
  (giảm corr với pool đã nộp) theo `REFINEMENT_GOALS` trong `src/llm/refiner.py` dòng 24-26.
  Hiện KHÔNG dùng để xấp xỉ RAM/style-factor neutralization một cách chủ động.
- Các toán tử sẵn có để build "proxy risk factor" hoàn toàn từ price/volume nội bộ:
  `ts_sum`, `ts_mean`, `ts_delta`, `ts_std`, `ts_corr`, `regression_neut`, `vector_neut` — ĐỦ để
  tính momentum proxy (`ts_sum(returns, 252)`) và reversion proxy (`ts_sum(returns, 5)`) rồi
  `regression_neut(alpha, proxy)` — KHÔNG cần toán tử mới. `group_mean`/`ts_regression`
  (dùng trong ví dụ market-beta của doc) CHƯA registered — cần thêm nếu muốn xấp xỉ market-beta
  cụ thể (không bắt buộc, vì MARKET neutralization ở tầng config đã trừ mean thị trường rồi).
- `RAM/Statistical/Slow/Fast/Crowding`: công thức factor **confidential** (docs nói rõ "cannot
  share granular details") → KHÔNG thể tái tạo local dù có đủ operator. Chỉ kiểm chứng được bằng
  cách thực sự gửi simulation lên Brain với `neutralization` = giá trị đó.
- `Double neutralization` cần `group_cartesian_product` + 1 group thứ hai — panel local chỉ có
  `sector`; muốn double-neut local cần tự tạo group thứ hai từ dữ liệu có sẵn (vd decile theo
  dollar-volume/độ biến động trailing) — khả thi về mặt kỹ thuật nhưng là group TỰ CHẾ, không phải
  `sta1_top1000c50` chính thức của Brain (không đảm bảo tương đương).
- Investability-constrained/Max Trade: cần dữ liệu ADV$ và participation model của Brain — panel
  478 mã không có mô hình thanh khoản này → Brain-only, chỉ đọc được sau khi sim thật.
- Fast D1: là vấn đề DATASET (field catalog), không phải neutralization — không nằm trong scope
  neutralization nhưng đáng ghi nhận cho backlog đa dạng hoá ý tưởng.

## 3. Danh sách ưu tiên đề xuất (cụ thể theo module)

### Quick win (ít effort, có thể làm ngay)

1. **[Refiner] Thêm bước "risk-neutralization ladder" sau khi LocalTuner chọn config tốt nhất**
   — Module: `src/llm/refiner.py` (nơi hiện chỉ sim Brain 1 lần cho ứng viên thắng cuộc local).
   Thay vì 1 sim, thử thêm 2-3 sim Brain với `neutralization` đổi thành `CROWDING` rồi
   `REVERSION_AND_MOMENTUM`, giữ nguyên decay/truncation đã tune local; chọn kết quả pass gate
   tốt nhất (Sharpe/fitness/self_corr). Đây là nơi ĐÚNG để test các setting Brain-only —
   không sweep được local nên phải trả bằng vài sim thật, nhưng rẻ hơn nhiều so với để tuột 1
   alpha vì self_corr 0.51 > 0.5. **Trực tiếp nhắm self_corr** vì CROWDING neutralization được
   thiết kế cho đúng vấn đề này.
2. **[power_pool_theme.py] Parse cụm `neutralization in (...)` thành constraint dùng được** thay
   vì để `unparsed_constraints` bỏ xó — nếu theme tuần hiện tại yêu cầu neutralization thuộc tập
   risk-neutralized, gate cần biết để không tự tin nộp alpha chỉ neutralize MARKET/SECTOR khi
   không match theme.
3. **[Neutralization enum] Thêm giá trị Brain-only** `SLOW/FAST/SLOW_AND_FAST/RAM/STATISTICAL/
   CROWDING` vào `src/backtest/config.py::Neutralization` (đánh dấu rõ "Brain-only, không sweep
   local") để Refiner có thể set chúng khi gọi API simulate — hiện thiếu nên không gửi được.
4. **[LocalTuner] Thêm biến thể "RAM-proxy" bằng operator có sẵn** — không cần code operator
   mới: thử ứng viên `regression_neut(alpha_core, ts_sum(returns, 5))` (bóc reversion 5 ngày) và
   `regression_neut(alpha_core, ts_sum(returns, 252))` (bóc momentum 12 tháng) như 2 candidate
   bổ sung trong vòng tune hiện có, chấm điểm bằng đúng `_submission_score` đang dùng. Đây là
   cách xấp xỉ lợi ích của RAM neutralization **hoàn toàn eval local được** (không cần Brain),
   vì chỉ dùng price/volume đã có trong panel.

### Đòn bẩy lớn hơn (effort vừa/cao, tác động sâu)

5. **[operators_local] Thêm `group_cartesian_product` + 1 group phụ tự tạo (vd cap-decile từ
   dollar-volume trailing)** để LocalTuner có thể sweep double-neutralization `sector × cap_decile`
   hoàn toàn local — đúng tinh thần doc double-neutralization ("thử group khác ngoài sector để
   cải thiện hiệu năng + giảm correlation"), nhưng cần validate: group tự chế khác
   `sta1_top1000c50` chính thức nên hiệu quả không đảm bảo giống hệt — nên coi là thử nghiệm, so
   sánh A/B với `_submission_score`, không thay thế sweep MARKET/SECTOR hiện tại.
6. **[GP/refiner] Bật `vector_neut`/`regression_neut` chủ động hơn** — hiện chỉ được gợi ý khi
   mục tiêu là `pool_fit`; nên thêm nhánh gợi ý dùng chúng ngay cả khi mục tiêu là `regime_fit`
   hoặc mặc định (không chỉ khi đã phát hiện corr cao), vì bóc momentum/reversion còn giúp ổn
   định theo năm (docs RAM: "giảm rủi ro trong giai đoạn thị trường điều chỉnh").

### Không ưu tiên / theo dõi thêm

- Investability-constrained metrics / Max Trade: chỉ hữu ích khi tool nhắm capacity cao hoặc
  vùng GLB/ASI; với alpha price/volume TOP3000 hiện tại, chỉ cần xem chỉ số này SAU sim Brain
  như một cảnh báo phụ (không đầu tư module riêng).
- Fast D1: mở dataset mới cho seed NOVEL idea trong tương lai — không thuộc cụm neutralization,
  để backlog riêng, không trộn vào việc này.

## 4. Local-eval được vs Brain-only — bảng tổng hợp

| Kỹ thuật | Local eval được? | Ghi chú |
|---|---|---|
| MARKET/SECTOR neutralize | Có (đang dùng) | Panel có group sector |
| INDUSTRY/SUBINDUSTRY | Không (chưa có data) | Cần thêm group field vào panel loader |
| RAM-proxy qua `regression_neut(alpha, ts_sum(returns,N))` | **Có** | Chỉ cần price/volume, dùng op sẵn có |
| Double-neut sector × group tự tạo | Có (cần thêm `group_cartesian_product` + group phụ) | Group tự chế, không phải `sta1_top1000c50` thật |
| SLOW/FAST/SLOW_AND_FAST/RAM/STATISTICAL/CROWDING (setting Brain) | **Không** | Công thức confidential, chỉ test qua Brain sim thật |
| Investability-constrained / Max Trade | Không | Cần ADV/participation model của Brain |
| Fast D1 | N/A (không phải neutralization) | Vấn đề dataset, Brain-only field |
