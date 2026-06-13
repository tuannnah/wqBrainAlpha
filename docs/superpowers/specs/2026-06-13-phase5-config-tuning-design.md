# GĐ5 — Tinh chỉnh cấu hình (search giai đoạn hai)

> Spec triển khai GIAI ĐOẠN 5 của `tailieu/BUILD_GUIDE_AI_alpha_tool.md`.
> Tiền đề: GĐ1-GĐ4 xong. Nhánh: `phase2-ai-loop`.

## Mục tiêu

Một alpha = (biểu thức + cấu hình). Cùng biểu thức, đổi neutralization/decay/
truncation cho metrics khác hẳn. Đây là **không gian tìm kiếm thứ hai**, nhỏ, xử
lý SAU khi đã có biểu thức tốt. Trọng tài cuối là **Out-of-Sample (OOS)** — tinh
chỉnh cấu hình chỉ theo In-Sample là một dạng overfitting.

## Hai không gian tìm kiếm tách biệt (T5.1, T5.2)

- **Không gian biểu thức** — nơi LLM/GA hoạt động (GĐ2-4), dùng **cấu hình mặc
  định cố định** (`SimConfig.default()`: neutralization=SUBINDUSTRY, decay=0,
  truncation=0.08, delay=1). KHÔNG quét cấu hình ở giai đoạn này để tránh bùng
  nổ số lần sim.
- **Không gian cấu hình** — neutralization, decay, truncation, delay. Chỉ quét
  trên các alpha hứa hẹn (đã vượt ngưỡng) ở giai đoạn hai.

`SimConfig` (`src/simulation/config.py`) đóng gói không gian cấu hình:
`default()`, `with_overrides(**)` (immutable, trả bản sao), `to_settings()` (dict
cho `Simulator.simulate`), `key()` (khoá ổn định, người đọc được, cho cache phân
biệt theo config).

## Vai trò các núm cấu hình (T5.4, T5.5)

| Núm | Vai trò chính | Vai trò kép / lưu ý |
|---|---|---|
| **neutralization** | Tinh chỉnh: trung hoà theo nhóm (market/sector/industry/subindustry) | **Công cụ decorrelation**: mức chi tiết hơn (subindustry > industry > sector > market) → tín hiệu tương đối hơn → correlation thấp hơn. Không chỉ là tinh chỉnh. |
| **decay** | **Núm điều khiển turnover chính**: làm mượt tín hiệu theo thời gian | Ảnh hưởng trực tiếp việc qua ngưỡng turnover và Fitness sau phí. Tăng decay → giảm turnover. |
| **truncation** | Giới hạn trọng số tối đa mỗi vị thế | Ảnh hưởng **drawdown/margin**: truncation chặt hơn → bớt tập trung → kiểm soát rủi ro đuôi. |

## Quét cấu hình (T5.3)

`ConfigSweeper` (`src/simulation/sweep.py`):
- `sweep(expression, base_config, grid, oos_min_ratio)`: quét tích Descartes các
  chiều trong `grid` (vd `{"decay": [0,4,8], "truncation": [0.05,0.1]}`), mỗi tổ
  hợp ghi đè lên `base_config`.
- Chỉ giữ cấu hình tốt CẢ IS lẫn OS; chọn cấu hình có **IS sharpe cao nhất trong
  số đã qua kiểm chứng OOS**. Bỏ qua kết quả `error`.
- Trả `SweepResult(best_config, best_result, trials)` — `trials` lưu lịch sử mọi
  tổ hợp (config, sharpe, os_sharpe, oos_ok) để quan sát.

Lưu ý: spec gốc nhắc tìm kiếm Bayesian HOẶC grid. Triển khai hiện dùng **grid**
(đủ rẻ và minh bạch cho không gian cấu hình nhỏ). Có thể nâng lên Bayesian (tái
dùng `src/optimization/bayesian.py` / optuna) nếu không gian cấu hình lớn lên.

## Kiểm chứng OOS (T5.6)

- `Simulator` parse thêm block `os` (Out-of-Sample) → `SimulationResult.os_sharpe`,
  `os_fitness`. Thiếu block `os` → `None` (không vỡ).
- `oos_passes(result, min_ratio)` (`src/simulation/oos.py`): OOS sharpe ≥
  `min_ratio · IS sharpe`? Thiếu OS hoặc IS ≤ 0 → False (an toàn — coi như chưa
  kiểm chứng được). Mặc định `min_ratio=0.5`.
- **Bắt buộc** áp vào mọi lần quét cấu hình: `ConfigSweeper` loại thẳng cấu hình
  chỉ đẹp ở IS.

## CLI / tích hợp

- `python main.py sweep-config --expr "..." [--decays 0,2,4,8]
  [--truncations 0.05,0.08,0.1] [--neutralizations SUBINDUSTRY,INDUSTRY]
  [--oos-ratio 0.5]`: quét cấu hình cho một alpha tốt, in bảng mọi tổ hợp kèm
  cờ OOS và cấu hình tốt nhất.

## Test (TDD, local thuần — không mạng)

- `sim_config`: default cố định; `to_settings`/`key`/`with_overrides` đúng, gốc
  bất biến.
- `oos`: parse block `os`; `oos_passes` qua khi OOS đủ cao, loại khi sụt mạnh,
  thiếu OS / IS≤0 → không qua.
- `config_sweep`: quét đủ tổ hợp grid; chọn cấu hình điểm cao qua OOS; loại cấu
  hình chỉ đẹp ở IS; không tổ hợp nào qua → best=None; bỏ qua kết quả error;
  lưu lịch sử trials.

## Acceptance (guide)

- Giai đoạn sinh biểu thức dùng cấu hình cố định (`SimConfig.default()`).
- Quét cấu hình chỉ chạy trên alpha tốt (qua lệnh `sweep-config`).
- Kết quả quét được lọc qua OOS — cấu hình chỉ đẹp ở IS bị loại.
- Toàn bộ test cũ + mới pass; không phá GĐ1-GĐ4.
