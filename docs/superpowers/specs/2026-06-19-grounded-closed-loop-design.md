# Thiết kế: Vòng đóng neo thực tế cho pipeline sinh alpha (Con đường 1)

Ngày: 2026-06-19
Mục tiêu: alpha **vừa vượt ngưỡng submit vừa độc đáo**, dưới ngân sách sim hẹp (vài chục–~100 sim/đêm, chạy tự động).

## 1. Bối cảnh & vấn đề (từ log `logs/wq_alpha_2026-06-19.log`)

Lần chạy thật 18:18→20:30 (engine hybrid + DeepSeek) cho **0 alpha đạt ngưỡng**. Ba nguyên nhân gốc:

1. **LLM bịa tên datafield** → ~15+ lượt sim chết với `Invalid data field` (`opt6_ivspyratioavg1m`, `asset_growth_rate_sensitivityfactor`, `opt6_1dorhv`…). DB *có* 8.599 field thật (USA/TOP3000/delay=1) trải khắp `model77`, `analyst4`, `news12`, `sentiment1`, `socialmedia12`, `option6`… nhưng LLM không dùng. `PreFilter._check_symbols` *có* khả năng loại field lạ, nhưng field bịa vẫn lọt tới API → có nhánh sinh bỏ qua prefilter.
2. **Sharpe/fitness là số LLM tự bịa** trong `auto:directions` (vd `sharpe=2.1, fitness=0.92`) — gán *trước khi* sim. Sim thật cho `best total ≈ 0.05`. Việc chọn/xếp hạng hướng dựa số bịa → sai từ gốc.
3. **Công thức rập khuôn** `scale(ts_decay_linear(group_neutralize(multiply(-1, ts_zscore(X)), volume_ratio), sector), n))` → vừa điểm thấp vừa dễ trùng.

Ngưỡng submit (`src/scoring/filter.py`): `sharpe ≥ 1.25`, `fitness > 1.0`, `turnover ∈ [0.01, 0.70]`, drawdown cap.

**Vấn đề cốt lõi:** pipeline là vòng *hở* — LLM đề xuất + tự chấm điểm tưởng tượng; số sim thật không quay lại dẫn dắt. Cộng field bịa nên phần lớn sim không chạy được.

## 2. Kiến trúc: vòng đóng neo thực tế

```
        ┌─ Catalog 8599 field thật (id, dataset_id, type) + blocklist invalid_fields ─┐
        │                                                                             │
   [A] Sinh ý tưởng neo-field → Translator neo-dataset                                │
        │                              ↓                                              │
        │                     [Cổng tiền-sim]  ── A1 validate field cứng              │
        │                                       ── C  chống trùng/zoo + steer dataset │
        │                                          ↓ (chỉ ứng viên sạch mới tốn sim)  │
        │                                        SIM thật                             │
        │                                          ↓                                  │
        └──── [B] phản hồi số THẬT (sharpe/fitness/turnover/corr) ─────────────────────┘
                   → xếp hạng & cấp ngân sách theo số thật → refine ứng viên gần ngưỡng
```

**Nguyên tắc neo:** mọi quyết định (sinh, chọn, refine, dừng) dựa trên *catalog field thật* hoặc *số sim thật* — không bao giờ dựa trên thứ LLM tự bịa.

## 3. Thành phần A — Neo field thật + validation cứng

- **A1. Cổng validation cứng tại biên simulation.** Một cổng bắt buộc ngay trước khi `Simulator.simulate()` gọi API: mọi biểu thức (LLM seed / GA / refiner / translator) phải qua validate field+operator với catalog thật. Field ngoài catalog → **không gửi API**, chuyển repair hoặc loại. Bịt mọi lỗ rò ở một chỗ. Gồm root-cause nhánh đang bỏ qua prefilter hiện tại.
- **A2. Blocklist bền vững.** Nạp `invalid_fields` lúc khởi động → trừ khỏi `known_fields` + nhồi danh sách "TUYỆT ĐỐI KHÔNG dùng" vào prompt + hard-reject ở A1. Field WQ từng từ chối không bị thử lại qua các đêm.
- **A3. Translator neo theo dataset.** Khi hướng nêu đích danh dataset, bơm *đúng field ID thật của dataset đó* (cột `dataset_id`) thay vì top-40 fuzzy.
- **A4. Repair gợi ý field thật.** Khi field lỗi, prompt sửa-lại kèm field thật gần nhất (fuzzy match trong cùng dataset) để LLM sửa về field có thật.

Kỳ vọng: ~100% lượt sim chạy được, hết `Invalid data field`.

## 4. Thành phần B — Vòng phản hồi số thật

- **B1. Bỏ metric bịa khỏi quyết định.** LLM không xuất `(sharpe,fitness)` dự đoán cho ý tưởng mới; nếu có thì *không bao giờ* dùng để xếp hạng. Mọi ranking/selection/stop dùng số thật từ `SimulationModel`.
- **B2. Cấp ngân sách kiểu bandit.** Mỗi hướng sim 1 phát "trinh sát"; hướng có điểm thật khá (gần/đạt ngưỡng) được dồn thêm sim/biến thể; hướng điểm thật kém bị bỏ sớm. Phù hợp ngân sách hẹp.
- **B3. Refine nhắm chiều chặn.** Dùng `blocking_dimensions` (sharpe/fitness/turnover/drawdown) để refiner sửa đúng chiều đang chặn pass, thay vì sửa mù. Chỉ refine ứng viên *gần ngưỡng*.
- **B4. Feedback exploit.** Giữ cơ chế đưa top alpha thật `(expr, sharpe, fitness)` vào prompt sinh biến thể (đã có trong `generator.py`); đảm bảo số đưa vào là số sim thật.

## 5. Thành phần C — Cổng độc đáo tiền-sim

- **C1. Gate AST trước sim.** Dùng `ReferenceZoo` loại ứng viên có AST-similarity vượt ngưỡng so với zoo hoặc ứng viên đã chọn trong lượt chạy. (AST ≠ correlation thật — chỉ là bộ lọc rẻ.)
- **C2. Steer dataset.** Theo dõi dataset đã thử trong phiên; bias sinh sang dataset ít dùng để tăng độ độc đáo theo *lựa chọn dữ liệu*.
- **C3. Xác nhận submit.** Alpha pass ngưỡng mới chạy `CorrelationChecker` (correlation thật của WQ) để xác nhận đủ điều kiện nộp.

## 6. Kiểm thử & triển khai (TDD, mỗi thành phần ≥1 commit)

- **A1:** test cổng chặn biểu thức chứa field ngoài catalog không bao giờ tới `simulate()`; test root-cause nhánh rò.
- **A2:** test nạp `invalid_fields` → loại khỏi known_fields + xuất hiện trong prompt blocklist.
- **A3:** test khi hướng nêu dataset X, field bơm vào prompt thuộc dataset X.
- **A4:** test repair trả về gợi ý field thật gần nhất.
- **B1:** test ranking/selection không đọc metric do LLM xuất; chỉ đọc `SimulationModel`.
- **B2:** test phân bổ ngân sách ưu tiên hướng điểm thật cao.
- **B3:** test refiner nhận đúng `blocking_dimensions`.
- **C1:** test ứng viên trùng AST bị loại trước sim.
- **C2:** test steer ưu tiên dataset ít dùng.

Thứ tự triển khai: **A → B → C** (A là tiên quyết: không có sim hợp lệ thì B/C vô nghĩa).

## 7. Ngoài phạm vi (YAGNI)

- Không xây mô hình dự đoán sharpe local (cần dữ liệu giá local — chưa có).
- Không đổi engine GA/MCTS lõi; chỉ thay tín hiệu dẫn dắt (số thật) và cổng lọc.
- Không mở rộng đa region trong giai đoạn này (giữ USA/TOP3000/delay=1).
