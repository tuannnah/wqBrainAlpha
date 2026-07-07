# Spec 2 — Tuning hướng nộp (Submission-oriented tuning)

Ngày: 2026-07-07 · Trạng thái: đã duyệt scope (4 đòn bẩy), chuẩn bị plan

## Bối cảnh & mục tiêu

Verify live (2026-07-07, `--refiner local`): best Brain Sharpe **0.74** (baseline 0.59), vẫn 0 alpha đạt. Mục tiêu người dùng: **1 tiếng auto-search ra ≥1 alpha đạt chuẩn nộp**. Đọc lại `docs/worldquantbrain/docs` rút ra 4 đòn bẩy cụ thể; đáng chú ý nhất là **neutralization đang SAI cho price/volume**.

Ngưỡng liên quan (Delay-1, từ docs): Regular Sharpe>1.25 & Fitness>1.0 & TO 1-70% & self-corr<0.7 & sub-universe. **Power Pool** (bar dễ nhất, mục tiêu thực tế): Sharpe≥1.0, ≤8 operator, ≤3 field, self-corr≤0.5, TO 1-70%, USA D1.

## 4 đòn bẩy (thành phần)

### A. Neutralization vào sweep + default đúng cho PV
Docs: **Price/Volume → MARKET hoặc SECTOR** (Industry/Subindustry làm GIẢM hiệu năng). Hiện `_closed_loop_configs` default **SUBINDUSTRY** cho Brain sim + LocalTuner KHÔNG sweep neutralization → nghi là lý do lớn Sharpe kẹt.
- LocalTuner thêm chiều neutralization ∈ **{MARKET, SECTOR}** (cả hai eval local được: MARKET trừ row-mean, SECTOR dùng group `sector` có trong panel; đúng khuyến nghị PV — không phí).
- `Neutralization` enum (NONE/MARKET/SECTOR/INDUSTRY/SUBINDUSTRY) → chuỗi Brain = tên enum uppercase. Refiner map `best_config.neutralization` sang `sim_config.with_overrides(neutralization=...)`.
- Đổi default closed-loop: neutralization do sweep chọn (khởi MARKET thay SUBINDUSTRY).

### B. Gate turnover + decay
Idea thực tế TO=101% → tự fail (Brain đòi 1-70%).
- LocalTuner ranking: config có **local turnover > 0.70 bị loại (điểm −inf)** → winner luôn trong dải. Sweep decay (đã có) tự kéo TO xuống.
- Pre-sim floor: local TO > 0.70 → bỏ, không tốn sim Brain.

### C. Nhắm Power Pool (gate cấu trúc + cờ)
- Kiểm cấu trúc: **unique operator ≤ 8** (OperatorCollector), **unique field ≤ 3** (FieldCollector, trừ grouping). Alpha đạt cấu trúc + Sharpe≥1.0 + self-corr≤0.5 → gắn cờ `power_pool_eligible`, log/đếm khi trúng. Thành "chuẩn đạt" thực tế của loop (báo cáo riêng).
- Không loại alpha không-power-pool (vẫn có thể là regular submit); chỉ ĐÁNH DẤU + báo cáo.

### D. Proxy robustness sub-universe (local)
Docs: `sub_sharpe ≥ 0.75·√(sub_size/univ_size)·alpha_sharpe`; tránh nhân size.
- Local: sub-universe = top `frac` thanh khoản (proxy `mean(volume*close)`); mask (pasteurize) tín hiệu về sub-universe, re-neutralize + re-backtest → sub_sharpe. Kiểm bất đẳng thức trên.
- Không đạt → phạt/loại ở tuner ranking (hoặc pre-sim floor). `frac` mặc định 0.5 (panel local nhỏ hơn TOP3000).

## Kiến trúc & luồng
Phần lớn dồn vào `LocalTuner` (A: chiều neut trong sweep; B,D: ràng buộc trong ranking) + gate/scoring (C: cờ power-pool; D: proxy) + đổi config default (A). Refiner map neutralization local→Brain. Ranking tuner đổi từ "tối đa Sharpe trần" → **tối đa Sharpe TRONG ràng buộc (TO≤0.70, sub-universe đạt)**; vi phạm ràng buộc = −inf (giữ gốc làm cận dưới như cũ).

## Xử lý lỗi
- Neutralization eval local lỗi (group thiếu) → biến thể −inf, bỏ qua (như hiện tại).
- Sub-universe/turnover tính lỗi (NaN) → coi ràng buộc KHÔNG đạt (bảo thủ), không sập.
- Mọi thứ khác giữ nguyên hành vi spec 1 (sim_error, Quota, monotone invariant).

## Kế hoạch test (TDD, fake/panel nhỏ, không mạng/LLM)
1. Neut map: enum MARKET/SECTOR → "MARKET"/"SECTOR"; refiner map đúng vào SimConfig.
2. LocalTuner sweep neut: kịch bản eval_fn cho SECTOR điểm cao → best_config.neutralization=SECTOR.
3. Turnover gate: config TO>0.70 bị loại; chọn config TO hợp lệ dù Sharpe thấp hơn config TO>0.70.
4. Power Pool cờ: expr ≤8 op & ≤3 field & Sharpe≥1.0 & self-corr≤0.5 → eligible=True; vi phạm 1 điều → False.
5. Sub-universe proxy: panel dựng sẵn nơi alpha đạt/không đạt bất đẳng thức → gate đúng.
6. Tích hợp: closed-loop chạy 1 batch, outcome có cờ power_pool, neutralization Brain = giá trị sweep.

## Ngoài phạm vi
- Không thêm dataset alt-data (spec riêng sau).
- Sweep neut chỉ {MARKET, SECTOR} (panel chỉ có group sector); INDUSTRY/SUBINDUSTRY để Brain, không eval local.
