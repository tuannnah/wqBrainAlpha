# Nghiên cứu: Tăng throughput vận hành cho closed-loop AI+MiniBrain

Nguồn: docs/worldquantbrain/docs/{brain-api,discover-brain,interpret-results,advanced-topics,consultant-information}/*.md
Ngày: 2026-07-08

## 1. Điểm then chốt rút từ docs

### 1.1 Giới hạn simulation (understanding-simulation-limits.md)
- Có **giới hạn hằng ngày** (per-account, reset theo giờ EST), lộ ra qua response headers của POST /simulations:
  `X-Ratelimit-Limit`, `X-Ratelimit-Remaining`, `X-Ratelimit-Reset` (giây tới lúc reset).
- **Mọi simulation thành công đều bị tính**, kể cả: (a) child simulation trong multi-simulation, (b) sim lại một Alpha đã tồn tại/trùng lặp. Brain **không tự loại trừ** sim trùng khỏi quota — trách nhiệm dedup thuộc về client.
- Platform cảnh báo khi còn <1000 sim/ngày. Nếu vượt hạn ngạch → không sim được tới hôm sau (giờ EST).
- Gợi ý dùng: giới hạn search space trước (1 field/1 operator/1 timeframe mỗi lần), mở rộng dần sau khi có signal — tránh "quét vét cạn" lãng phí quota.

### 1.2 Tránh sim trùng (how-can-you-avoid-duplicate-simulations.md)
- Pattern chuẩn của WQ: **hash toàn bộ alpha config dict** (expression + toàn bộ settings: region/universe/delay/decay/neutralization/truncation/pasteurization/testPeriod/unitHandling/nanHandling/language) bằng SHA-256 trên JSON `sort_keys=True`, lưu vào bảng cache local (cột `alpha_hashed, alpha_id, date_created`, ví dụ file parquet).
- Trước mỗi lần sim: tính hash → tra cache → nếu có thì tái dùng `alpha_id` có sẵn (không sim lại); nếu không có thì sim rồi ghi cache.
- Áp dụng batch: lọc list alpha cần sim thành `new_alpha_list` (chưa từng sim) trước khi gọi multi-simulation, chỉ multi-sim phần mới.

### 1.3 Multi-Simulation (consultant-simulation-features.md)
- **Consultant được chạy tối đa 8 Multi-Simulation đồng thời.** Mỗi Multi-Simulation chứa tối đa **10 Alpha chạy TUẦN TỰ bên trong nó** (mỗi alpha có thể khác operator/data field/setting, nhưng toàn bộ 10 alpha trong 1 multi-sim phải cùng REGION và DELAY).
- ⇒ Cấu trúc song song thực sự = **8 batch chạy song song**, mỗi batch xử lý tuần tự 10 alpha bên trong. Không phải "80 sim chạy song song" nhưng vẫn là **8x thông lượng** so với chạy 1 sim/lần.
- Lỗi "requires more resources than available" → giảm số alpha trong 1 multi-sim rồi thử lại.
- Có thể lấy thêm **risk-handled performance** (SLOW_AND_FAST neutralization) và **investability-constrained performance** (MaxTrade ON) ngay trong cùng 1 lần sim, thay vì phải sim thêm lần riêng để kiểm tra các biến thể này.
- Test Period: có thể set riêng biệt (0-6 năm cuối của IS) để đánh giá mà không cần sim thêm lần khác.

### 1.4 High-Turnover Alpha (getting-started-with-high-turnover-alphas.md)
- Ngưỡng "High TVR": turnover > 20% (KHÔNG phải >70%). Điều kiện đủ để coi là High TVR: turnover>20% VÀ `hightvrReturns/original_return > 0.75`.
- 4 phân loại (không bắt buộc alpha nào cũng cần đạt, đây là các hướng công nhận khác nhau):
  1. After-Cost: Sharpe sau cost > 1.0
  2. Investable: Sharpe sau maxtrade/maxpos > 2.0 VÀ turnover sau đó > 20%
  3. Liquid: Sharpe TOP200 > 1.0 và sharpe_top500/sharpe_top200 > 0.7
  4. Orthogonal: submit-able sau khi áp RAM neutralization
- Turnover cao phải là **hệ quả** của một effect ngắn hạn thật (event reaction, flow/activity, microstructure, fundamental-refresh nhanh) — không phải mục tiêu tối ưu trực tiếp. Turnover cao do nhiễu/bất ổn là "artificial turnover" — lỗi phổ biến cần tránh.

### 1.5 PnL Realization Horizon (understanding-pnl-realization-horizon.md)
- Metric đo tốc độ position → PnL thực hiện. 2 thành phần: short-term (1-5 ngày) và long-term (10-20+ ngày).
- Tiêu chí HTVR Campaign: turnover>20% VÀ (horizon <20 ngày HOẶC hightvrReturns >75% return).
- Momentum/news alpha nên có horizon <10 ngày; fundamental alpha có thể hợp lý ở 20-40 ngày. Horizon không khớp thesis ⇒ dấu hiệu overfit/lỗi data.
- Muốn giảm long-term component (rút ngắn horizon): trừ moving-average của position, hoặc dùng data đổi nhanh hơn (tick data/news/options).

### 1.6 Must-read posts & ACE library
- Chỉ là danh sách link ngoài (Sharpe cao hơn, giảm turnover, giảm correlation, làm mượt PnL, neutralization, tránh overfit) — không có nội dung chi tiết trong docs local, chỉ tiêu đề tham khảo.
- ACE library (ace_lib.py) là thư viện Python chính thức của WQ để gọi API (simulate, multi-sim, lấy alpha stats) — nếu tool hiện tại tự viết wrapper riêng, có thể đối chiếu ACE để bắt kịp pattern chính thức (rate-limit headers, hash cache, `simulate_alpha_list_multi(limit_of_concurrent_simulations, limit_of_multi_simulations)`).

## 2. Đề xuất ưu tiên cho TOOL (closed-loop AI+MiniBrain)

### Quick win (effort thấp, làm ngay)
1. **Đọc & tôn trọng rate-limit headers**: sau mỗi POST simulation, lưu `X-Ratelimit-Remaining`/`Reset`; khi gần cạn (<1000 hoặc theo ngưỡng cấu hình) tự động tạm dừng closed-loop tới giờ reset (EST) thay vì để lỗi 429/block cứng.
2. **Hash-dedup cache đúng chuẩn WQ** (nâng cấp avoid-list hiện có): hash SHA-256 trên toàn bộ alpha_dict (expression + region/universe/delay/decay/neutralization/truncation/pasteurization/testPeriod...) chứ không chỉ theo "ý tưởng/tên", lưu SQLite/parquet có `alpha_id`. Refiner/LocalTuner tra cache này TRƯỚC khi gọi Brain sim — loại các biến thể LocalTuner sinh ra trùng hệt config đã sim (rất dễ xảy ra khi tune quanh 1 optimum).
3. **Lấy stat phụ trong cùng 1 lần sim** (risk-handled SLOW_AND_FAST, investability MaxTrade ON) thay vì sim thêm lần riêng để kiểm tra biến thể neutralization/investability — giảm số round-trip Brain cần cho mỗi idea.

### Đòn bẩy lớn (effort trung-cao, tăng throughput cấu trúc)
4. **Chuyển Simulator sang Multi-Simulation batch + async poll**: gom N (≤10) biến thể LocalTuner sinh ra cho cùng 1 ý tưởng (cùng region/delay, khác decay/neutralization/tham số) thành 1 Multi-Simulation; chạy song song tối đa 8 Multi-Simulation cùng lúc (8 ý tưởng khác nhau x 10 biến thể/ý tưởng cùng lúc). Đây là thay đổi kiến trúc lớn nhất: từ "1 sim tuần tự/ý tưởng" (5-10 sim/giờ) sang "8 batch song song x 10 biến thể tuần tự/batch" — tăng thông lượng theo cấp số nhân mà không đổi hạ tầng Brain.
   - Cần: closed-loop tách 2 pha — (a) LocalTuner sinh candidate set (không gọi Brain) → (b) Simulator dispatch theo batch multi-sim + polling bất đồng bộ (asyncio/thread pool) thay vì `wait-block` từng sim.
   - Rủi ro: lỗi "resource requires more than available" khi batch quá lớn → cần cơ chế giảm batch size tự động và retry.
5. **Tách turnover gate ≤0.70 thành 2 nhánh xử lý**: alpha turnover thấp (chuẩn hiện có) và alpha turnover >20% được định tuyến qua pipeline "HTVR-aware" (kiểm tra thêm sharpe-after-cost, maxtrade/maxpos, sharpe TOP200) thay vì bị loại/gộp chung — tận dụng được các ý tưởng turnover cao (event/flow/microstructure) mà hiện tool có thể đang bỏ qua hoặc xử lý sai ngưỡng.
6. **PnL horizon như tín hiệu chẩn đoán**, không phải gate cứng: khi refiner giữ lại alpha để nộp, đối chiếu horizon với thesis loại alpha (momentum/event ⇒ kỳ vọng <10-20 ngày; fundamental ⇒ có thể 20-40 ngày) để phát hiện overfit sớm, tránh mất 1 slot submit quý giá cho alpha có horizon lệch thesis.

## 3. Ghi chú triển khai
- Cả (2) hash-cache và (4) multi-sim batch nên implement cùng lúc vì multi-sim cần "new_alpha_list" đã lọc dedup làm đầu vào (xem code mẫu trong how-can-you-avoid-duplicate-simulations.md: `check_if_alpha_already_simulated` → lọc → `simulate_alpha_list_multi(limit_of_concurrent_simulations=1, limit_of_multi_simulations=8)`).
- Đây là thay đổi có tác động lớn nhất tới nút thắt hiện tại (mỗi sim 5-12 phút, sim timeout 1200s, chỉ 5-10 sim/giờ): multi-sim + async poll trực tiếp giải quyết bottleneck "tuần tự" nêu trong bối cảnh tool.
