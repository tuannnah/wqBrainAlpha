# Thiết kế: Căn chỉnh config để tăng chất lượng alpha (Sharpe/fitness)

Ngày: 2026-07-06

## Bối cảnh & vấn đề

Engine closed-loop (GP + LLM refine) sinh alpha có chất lượng thấp trên WQ Brain thật:
alpha tốt nhất chỉ đạt Sharpe ~0.79, fitness 0.21 — rất xa ngưỡng nộp consultant
(Delay-1: Sharpe > 1.58, fitness > 1, IS Ladder 2.38 cho 2 năm gần nhất).

Điều tra code phát hiện closed-loop dùng **hai bộ config lệch nhau**:

- **Local gate** (backtest lọc rẻ trước khi tốn quota Brain sim) — `PortfolioConfig`:
  `neutralization=NONE, decay=0, truncation=0.10`
  (từ `_run_closed_loop_session` defaults `NONE/0/0.10`).
- **Brain sim** (chấm điểm thật) — `SimConfig.default`:
  `neutralization=SUBINDUSTRY, decay=0, truncation=0.08`
  (closed-loop truyền `sim_config=None` → dùng default).

Hai hệ quả:
1. **Mismatch**: local gate đánh giá alpha KHÁC cách WQ Brain chấm (NONE vs SUBINDUSTRY)
   → bộ lọc rẻ loại/giữ sai, calibration ρ (local↔Brain sharpe) kém tin.
2. **decay=0 ở cả hai**: tài liệu consultant (`consultant-dos-and-don-ts.md`,
   `finding-consultant-alphas.md`) khuyên dùng decay 3–4 để làm mượt tín hiệu, giảm
   turnover, thường nâng Sharpe/fitness.

## Mục tiêu

Tăng Sharpe/fitness thực tế của alpha bằng cách đưa closed-loop về **một bộ config
thống nhất** cho cả local gate lẫn Brain sim, và bật decay theo khuyến nghị docs.

Ngoài phạm vi (YAGNI, để sau): backfill coverage (hướng B), sub-universe robustness
gate (hướng C), sweep decay/neutralization tự động.

## Thay đổi

Bộ config thống nhất mới cho closed-loop: **`neutralization=SUBINDUSTRY, decay=4,
truncation=0.08`**.

| Nơi | Hiện tại | Đổi thành |
|---|---|---|
| `closed_loop_cmd` options (`main.py` ~669-671) | `NONE, 0, 0.10` | `SUBINDUSTRY, 4, 0.08` |
| `_run_closed_loop_session` params default | `NONE, 0, 0.10` | `SUBINDUSTRY, 4, 0.08` |
| `_run_closed_loop_session` thân | tạo `cfg` (local), KHÔNG truyền `sim_config` (Brain dùng default) | tạo `sim_config` khớp `cfg` và truyền `_make_research_loop(sim_config=...)` |

Sau sửa: local gate `PortfolioConfig` và Brain `SimConfig` luôn dùng cùng
neutralization/decay/truncation → hết mismatch. `_menu_auto_sim` gọi
`_run_closed_loop_session` không tham số nên tự hưởng default mới.

Chi tiết `sim_config`: dựng `SimConfig(region, universe, delay, neutralization, decay,
truncation)` từ cùng các giá trị dùng cho `cfg`. Xác nhận constructor `SimConfig` nhận
các field này khi triển khai (nếu tên field khác, ánh xạ tương ứng).

## Testing (TDD)

1. `_run_closed_loop_session` dựng `sim_config` có neutralization/decay/truncation
   BẰNG với `cfg` (local) — bắt regression mismatch tương lai.
2. Default mới của `closed_loop_cmd` / `_run_closed_loop_session` = `SUBINDUSTRY, 4, 0.08`.
3. KHÔNG test giá trị Sharpe/fitness (kết quả thực nghiệm, đo bằng chạy thật sau khi merge).

## Rủi ro

- `decay=4` giảm turnover: một alpha turnover vốn rất thấp có thể tụt < 1% (fail
  turnover-min). Hiếm; đánh đổi lại phần lớn alpha mượt hơn, Sharpe/fitness cao hơn.
- Đổi default ảnh hưởng menu 5 (đúng yêu cầu "đổi mặc định engine luôn").

## Cách đo hiệu quả

Sau khi triển khai: chạy closed-loop 1 tiến trình sạch (session còn hạn), so Sharpe/
fitness các alpha mới (`brain_sim_links`, lưu ý `created_at` là UTC = giờ máy − 7h) với
mốc nền hiện tại (Sharpe max 0.79). Kỳ vọng phân bố Sharpe dịch lên.
