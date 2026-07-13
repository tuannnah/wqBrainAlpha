# Quy trình sinh Alpha (MiniBrain closed-loop)

Tài liệu tóm tắt toàn bộ luồng tự động sinh — mô phỏng — chọn lọc alpha để nộp lên
WorldQuant Brain. Cập nhật theo code tại `src/app/closed_loop_adapters.py::build_closed_loop`
và trạng thái trong `PROGRESS.md` (2026-07-09).

## Tổng quan 1 dòng

Vòng kín (closed-loop): **sinh ý tưởng → chấm điểm cục bộ (local gate) → mô phỏng thật trên
Brain → verify self-corr → chọn alpha đạt chuẩn nộp**, học dần qua avoid-list + calibration.

Lệnh chạy: menu `run.bat` mục **5 (Auto SIM)** == CLI `closed-loop` (cùng lõi
`_run_closed_loop_session` trong `main.py`).

## Điều kiện trước khi chạy

1. **Đăng nhập Brain** (menu mục 1 hoặc `main.py login`) — cần **terminal thật** vì WQ đòi quét
   QR persona. Session lưu ở `.wq_session`, còn hạn ~2-3h thì tiến trình nền reuse, không hỏi QR.
2. **Cache sẵn** fields + operators (menu mục 1 tự `ensure`, hoặc mục 2/3 tải lại).
3. **DB tách theo email**: `.wq_account`/`WQ_EMAIL` → `wq_alpha_<slug>.db`.
4. **KHÔNG chạy 2 tiến trình closed-loop song song** trên cùng DB (tranh avoid-list/pool, khóa SQLite).
5. `LLM_BACKEND` trong `.env` (deepseek / claude-cli / codex-cli).

## Các bước trong vòng kín

### Bước 1 — Sinh ý tưởng (Idea Sources)

Các nguồn ý tưởng được bọc lồng nhau (ngoài cùng chạy trước). Thứ tự trong `build_closed_loop`:

| Nguồn | Vai trò | Bật/tắt |
|---|---|---|
| `CombinerIdeaSource` | Ghép N tín hiệu ít tương quan thành 1 alpha (Grinold–Kahn √N) | `--combine`, mặc định **BẬT** |
| `AltDataIdeaSource` | Seed dataset thay thế (option8 IV/HV, socialmedia8 sentiment) đi **thẳng** Brain | `--alt-data`, mặc định TẮT |
| `CuratedIdeaSource` | Seed core PV **đã kiểm chứng** (VWAP intraday-reversal ~Sharpe 1.5) trước GP | `curated_seeds`, mặc định BẬT |
| `GPIdeaSource` | Genetic Programming sinh biểu thức mới quanh seed | luôn có (nền) |

- Combiner: ghép `add(rank(s1), rank(s2), ...)` trọng số đều, greedy khử tương quan τ=0.3,
  N∈[2,4], tối đa 5 combo/run; **tước hết `group_neutralize`** để tuner tự chọn.
- Alt-data: khi `local_usable(expr,data)==False` (field ngoài panel local) → bỏ tune/floor
  local, sim Brain 1 lần với neutralization theo category (option→SECTOR, social→SUBINDUSTRY).

### Bước 2 — Chấm điểm cục bộ + gate (pre-filter tiết kiệm quota)

- Backtest trên panel local (`market_yf`) → tính metrics (Sharpe, fitness, turnover, IS-Ladder).
- **Pre-sim local floor**: bỏ qua alpha local Sharpe < 0.5 (rác) trước khi tốn sim Brain.
- Gate độc đáo (self-corr proxy) loại ứng viên trùng vùng đã mine trong pool.
- Calibration: **local ≈ Brain / 1.28** — local chỉ là proxy xếp hạng, KHÔNG dự báo tuyệt đối.
- Panel local chỉ ~478 mã (Brain TOP3000) → local **underestimate fitness** nặng.

### Bước 3 — Tune tham số (LocalTuner)

- Quét neutralization + decay + truncation.
- Xếp hạng theo **điểm nộp** `min(Sharpe/1.25, Fitness/1.0)` (không đuổi Sharpe trần → tránh
  chọn config turnover cao làm rớt fitness).

### Bước 4 — Mô phỏng thật trên Brain (SIM)

- Config chuẩn: `SUBINDUSTRY / decay=4 / truncation=0.08` (hoặc theo Power Pool Theme nếu có).
- `Simulator.TIMEOUT_SECONDS = 1200` (core mạnh có thể mất 8-20′/sim ở free-tier).
- Nhận diện `AuthExpiredError` (session hết hạn) + `QuotaExceededError` (429 / hết quota ngày)
  → dừng gọn.
- Nếu bật Power Pool Theme (`resolve_theme_sim_config`): override region/universe (USA/TOP1000)
  + risk-neutralization (STATISTICAL/CROWDING/SLOW...).

### Bước 5 — Verify self-correlation

- `GET /alphas/{id}/correlations/self` trả HTTP 200 **body rỗng** trong lúc WQ tính →
  `CorrelationChecker` **poll** tới khi có JSON (bug JSONDecodeError đã fix, commit da4887e).
- Ngưỡng self-corr < 0.70 (để lọt Power Pool nên nhắm ≤ 0.5).

### Bước 6 — Ghi nhận + học

- Kết quả SIM ghi vào cả `brain_sim_links`, `AlphaModel`, `SimulationModel` (nên `top`/`submit`
  thấy được alpha).
- **Log CSV mỗi phiên**: `logs/alphas_<YYYY-MM-DD_HHMMSS>.csv` — 1 dòng/ý tưởng (status, source,
  expression, region, neutralization, sharpe, fitness, turnover, self_corr, wq_alpha_id...).
- Alpha thất bại/gate → để trống metrics nhưng vẫn ghi expression + `stop_reason` để soi độ đa dạng.
- Feedback học: avoid-list (biểu thức đã fail), calibration ρ, pool self-corr Brain tầng-2.

### Bước 7 — Nộp (Submit)

- **Hành động KHÔNG đảo ngược** → cần người dùng đồng ý, KHÔNG auto-submit.
- `submit --no-dry-run` (dry-run mặc định để verify e2e trước).
- **Cuối mỗi phiên `closed-loop`, tool tự in khối "🚀 SẴN SÀNG NỘP"** (Task 8): liệt alpha đạt
  CẢ BA điều kiện Regular thật — `status=passed` + `failed_checks=[]` + `self_corr` **đã
  verify** < `SELF_CORR_MAX` (`config/thresholds.py`, mặc định 0.70) — kèm sẵn `wq_alpha_id`
  và lệnh nộp. Khối này **KHÁC HẲN** dòng "⭐ Power Pool eligible" cũng có thể xuất hiện cùng
  phiên: Power Pool eligible chỉ là cờ **cấu trúc** (Sharpe≥1.0, ≤8 operator, ≤3 field,
  self_corr≤0.5), **KHÔNG** xác nhận nộp được (xem commit `e27821d`) — đừng lẫn hai khối.
- **Alpha có thể đã nằm sẵn trong DB từ phiên trước mà không biết** — vì self-corr chỉ được
  ghi lại (persist) khi alpha đi qua cầu `BrainSimLinkModel` (closed-loop) hoặc khi bạn tự
  chạy `submit --dry-run` một lần để verify; nếu chưa từng verify, alpha sẽ KHÔNG xuất hiện
  trong khối SẴN SÀNG NỘP dù đã pass. Ví dụ thật: alpha `rKlkG9O8` (Sharpe 1.57, self-corr
  0.49, `failed_checks=[]`) đã nằm trong DB từ phiên 2026-07-06 — verify tay bằng:
  ```
  ./venv/Scripts/python.exe main.py top --sort sharpe   # xem danh sách alpha đã pass, có sharpe/fitness/score
  ./venv/Scripts/python.exe main.py submit               # dry-run: xem alpha nào sẽ được chọn nộp + tự verify self-corr LIVE
  ./venv/Scripts/python.exe main.py submit --no-dry-run   # nộp thật
  ```

## Ngưỡng nộp (Brain thật, Delay-1)

**QUAN TRỌNG — `failed_checks == []` lúc SIM chỉ là điều kiện CẦN, không phải ĐỦ.** Bằng
chứng thật 2026-07-14 (lần nộp THẬT đầu tiên qua tool, alpha `KP9nwpEg`): sim cho Sharpe 1.41,
fitness 0.99, self-corr 0.4265, `failed_checks=[]` — tưởng đã sẵn sàng — nhưng `POST
/alphas/{id}/submit` khi nộp THẬT vẫn trả **403** với body:
```json
{"is":{"checks":[
  {"name":"LOW_SHARPE","result":"FAIL","limit":1.58,"value":1.41},
  {"name":"LOW_FITNESS","result":"FAIL","limit":1.0,"value":0.99},
  {"name":"LOW_TURNOVER","result":"PASS","limit":0.01,"value":0.2908},
  {"name":"HIGH_TURNOVER","result":"PASS","limit":0.7,"value":0.2908}
]}}
```
Alpha vẫn `stage: IS, status: UNSUBMITTED` sau đó — không hề được nộp dù `POST /submit` ban
đầu trả 200 (WQ tính bất đồng bộ, phải poll `GET /alphas/{id}/submit` mới ra kết quả thật —
xem `SubmissionManager._poll_submit_result`, fix `fix-submit-async`).

- Ngưỡng NỘP THẬT (hard, đo trực tiếp từ `limit` trong response 403 trên):
  **Sharpe ≥ 1.58** và **Fitness ≥ 1.0** (`config/thresholds.py`:
  `SUBMIT_MIN_SHARPE`/`SUBMIT_MIN_FITNESS`) — khác ngưỡng lúc SIM (`status=passed` +
  `failed_checks=[]`), Brain enforce lại lúc nộp. `SubmissionManager.select_candidates`,
  `MiniBrainRepository.submit_ready_alphas` và khối "SẴN SÀNG NỘP" đều đã áp 2 ngưỡng này.
- `failed_checks == []` (qua hết IS submission checks LÚC SIM — điều kiện CẦN).
- Brain enforce thêm: `IS_LADDER_SHARPE`, `LOW_2Y_SHARPE`, `LOW_SUB_UNIVERSE_SHARPE`.
- self-corr < 0.70.
- Bằng chứng REJECTED (không đạt chuẩn, đã sửa hiểu lầm cũ): alpha `KP9nwpEg` Sharpe 1.41,
  fitness 0.99, self-corr 0.4265, `failed_checks=[]` lúc sim → **403 REJECTED** lúc nộp thật.

## Bẫy cần nhớ

- **Timezone**: `created_at`/`sim_at` ghi bằng **UTC**, giờ máy UTC+7 → query lọc thời gian trừ 7h.
- **Pool bão hòa**: vùng VWAP-reversal đã bị mine → alpha mới trùng self-corr; muốn nhiều alpha
  mới → mở rộng dataset (alt-data/novel-v2), KHÔNG phải tăng tốc.
- **Chỉ dùng field đã verify LIVE** qua `get_datafields` — đừng tin cứng `VERIFIED_FIELDS`
  (chứa field chưa xác nhận cho account hiện tại).
- **Test file tiếng Việt**: chạy `./venv/Scripts/python.exe -m pytest` (python hệ thống thiếu `lark`).

## Sơ đồ luồng

```
[Idea Sources]  Combiner → AltData → Curated → GP
      ↓ (biểu thức)
[Local score + gate]  backtest panel_yf → pre-sim floor (Sharpe<0.5 bỏ) → gate độc đáo
      ↓
[LocalTuner]  quét neut/decay/truncation, xếp theo điểm nộp
      ↓
[Brain SIM]  sim thật (SUBINDUSTRY/decay4/trunc0.08 | Power Pool Theme)
      ↓
[Verify self-corr]  poll /correlations/self < 0.70
      ↓
[Ghi DB + CSV log + avoid-list/calibration]
      ↓
[Submit]  chỉ khi người dùng đồng ý (không đảo ngược)
```
