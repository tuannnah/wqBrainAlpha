# MiniBrain / wqBrainAlpha — Improvement Spec **v2** (log + code analysis, 2026-07-10)

> Đọc bằng Claude Code trong repo `tuannnah/wqBrainAlpha`. Spec này dựa trên phân tích
> **code thật** (`src/simulation/simulator.py`, `src/pipeline/closed_loop.py`,
> `src/app/closed_loop_adapters.py`, `src/backtest/gate.py`, `config/thresholds.py`) +
> log phiên `wq_alpha_2026-07-10.log` và `alphas_2026-07-10_173500.csv`.
> Số dòng log trích dưới đây là bằng chứng trực tiếp.

## 0. Những gì v1 đã có (xác nhận từ code — không đụng lại)

Instrumentation Pha 0 (schema đầy đủ: `stage_reached/fail_check/family/expr_depth/dedup_key/
local_sharpe/brain_sharpe/*_ms`), depth guard (`refine_and_sim` chặn `MAX_DEPTH=7`), dedup
hook (`CanonicalHasher` tiêm qua `dedup_key_fn`), family-budget + đóng họ (`max_per_family=8`,
"🚪 Đóng họ"), calibration tracker (`CalibrationTracker`), alt-data + fundamental bật mặc định,
và **đã thử** dùng `regression_neut` để hạ self-corr (`neut_risk_factors`). Tốt. Nhưng vòng
kín vẫn **0 pass** vì các lỗi bên dưới — phần lớn là bug **correctness**, không phải thiếu tính năng.

---

## 1. Kết quả phiên 2026-07-10 (bằng chứng)

- **0/46 ý tưởng đạt** qua 2 lần chạy (10:44–13:04 và 17:30–18:16).
- Brain sim tốn **8–19 phút/lần** (`sim_ms` CSV: 722k, 1.158M, 486k, 778k ms) — tất cả `LOW_SHARPE`.
- Brain sharpe cao nhất chạm được = **0.75** (pv_reversal, CSV #7); fundamental 0.49–0.66 **hoặc âm**.
- **Calibration ρ = 0.362 < 0.50** (log dòng 51) — ranking local **hết đáng tin**.
- Rất nhiều batch sinh ~2 phút rồi **bỏ sạch** vì "họ đã đóng [pv_reversal]" (log 12:40→13:04).

---

## 2. Phát hiện gốc rễ — xếp theo tác động (mỗi mục: bằng chứng → gốc trong code → sửa → acceptance)

### C1 — 🔴 CRITICAL: pre-check loại nhầm `regression_neut` → **đòn bẩy hạ self-corr chết hoàn toàn**

**Bằng chứng.** Log dòng 35, 39, 44, 48, 56, 64, 146, 150, 155, 159:
`Bỏ sim (tiền-kiểm): Operator không tồn tại: regression_neut | expr=regression_neut(...)`.
Mọi core pv_reversal sau khi `LocalTuner` bọc `regression_neut(core, rank(volume))` (đường
`neut_risk_factors`, Pha 3.1) đều bị **`pre_sim_validator` loại trước khi sim**.

**Gốc trong code.** `regression_neut` **là operator WQ hợp lệ** (xác nhận trong skill
`fastexpr-operators.md` §Neutralization: `regression_neut(y, x)`, `vector_neut(x, y)`). Nhưng
`pre_sim_validator` (tiêm vào `Simulator` qua `pre_sim_validator=`) đang kiểm operator theo
**registry local** — `src/operators_local.py` chỉ đăng ký **"28 operator thật"** để *backtest
local* (`gate.py` dòng import: "side-effect: đăng ký 28 operator thật vào registry"). Các
operator **chỉ-Brain** (không tính được trên panel local: `regression_neut`, `vector_neut`,
`ts_regression`, …) **không** nằm trong 28 op đó → bị coi là "không tồn tại" và loại oan.
⇒ Toàn bộ chiến lược self-corr của v1 **không bao giờ chạm Brain**. Đây là lý do #1 pv_reversal
"0 pass rồi bị đóng" — **không phải vì tín hiệu yếu, mà vì bug pre-check.**

**Sửa.**
1. Tách **hai tập operator**: `LOCAL_COMPUTABLE_OPS` (28 op, chỉ dùng gate/backtest local) và
   `WQ_VALID_OPS` (nạp từ **live `/operators`** — QUY_TRINH bước 2 đã cache operators trong DB).
2. `pre_sim_validator` phải kiểm operator theo **`WQ_VALID_OPS`**, KHÔNG theo registry local.
3. Biểu thức dùng operator chỉ-Brain (không local-computable) → đi **thẳng Brain** như nhánh
   alt-data (`_sim_direct`), bỏ qua backtest/floor local (đằng nào local cũng không tính được).
4. Verify **LIVE** `regression_neut`/`vector_neut` có trong `/operators` của account hiện tại
   (cardinal rule của skill). Nếu account thật sự thiếu `regression_neut` → dùng `vector_neut`,
   hoặc hạ self-corr bằng **neutralization SETTING** (STATISTICAL/CROWDING) thay vì operator.

**Acceptance.** Một core pv_reversal + `regression_neut(core, rank(volume))` **chạm Brain**
(sim_ms > vài giây, có `brain_sharpe`/`self_corr` thật), không còn dòng "Operator không tồn tại:
regression_neut" trong log.

### C2 — 🔴 CRITICAL: instrumentation **dán nhãn sai** pre-sim reject thành `simmed / LOW_SHARPE`

**Bằng chứng.** CSV #1 (`regression_neut(...)`): `stage_reached=simmed`, `fail_check=LOW_SHARPE`,
`sim_ms=0.70` (ms!) — nhưng log cho thấy nó bị **pre-sim reject vì operator**. `sim_ms≈0.7ms`
= chỉ là lần gọi `pre_sim_validator`, chưa từng sim Brain.

**Gốc trong code.** Khi `pre_sim_validator` loại, `Simulator.simulate` trả
`SimulationResult(status="error", raw={"error": f"pre-sim reject: {reason}"})`. Rồi
`LocalTunerRefiner._finalize` vẫn được gọi trên result lỗi này: `passed=False`,
`fail_check=fail_check_from_reasons(_reasons)` (ra `LOW_SHARPE`), `stage_reached="simmed"`.
⇒ CSV **giấu** bug operator; chỉ có `.log` mới lộ. Không đọc `.log` thì không bao giờ thấy C1.

**Sửa.** Trong `simulate`, khi pre-check loại, trả result mang **category** rõ ràng
(`OPERATOR_INVALID` / `FIELD_INVALID` / `DEPTH`). Trong refiner, map sang
`stage_reached="op_invalid"`/`"field_invalid"` + `fail_check` tương ứng, `sims_used=0`
(nó **chưa** tốn quota Brain — đừng đếm là 1 sim). Thêm 2 cột CSV: `presim_reason`, `is_brain_sim`.

**Acceptance.** CSV phân biệt được op-invalid vs low-sharpe; `SessionSummary` funnel có bucket
`op_invalid` riêng; tổng `is_brain_sim=True` khớp số sim Brain thật (đối chiếu quota).

### C3 — 🔴 CRITICAL: GP sinh operator **không tồn tại** (`ts_std` — đúng phải là `ts_std_dev`)

**Bằng chứng.** Log dòng 166: `Operator không tồn tại: ts_std | expr=...ts_std(power(low,-2),20)...`.
`ts_std` xuất hiện rải rác trong output GP mọi phiên.

**Gốc trong code.** Từ vựng operator của GP (`src/gp/…`) chứa `ts_std` — **không phải operator
WQ** (WQ dùng `ts_std_dev`). Mỗi lần GP dùng `ts_std` là một ứng viên chết + ~20–30s phí.

**Sửa.** (a) Xoá `ts_std` khỏi từ vựng GP (thay `ts_std_dev`). (b) **Fail-fast ở startup**:
assert `GP_VOCAB ⊆ WQ_VALID_OPS` (và ⊆ `LOCAL_COMPUTABLE_OPS` cho nhánh cần backtest local);
in ra operator lệch. (c) Unit test giữ bất biến này.

**Acceptance.** Không còn dòng "Operator không tồn tại" cho operator do GP sinh; test CI chặn
mọi từ vựng lệch catalog.

### C4 — 🟠 HIGH: local↔Brain **mất hiệu chỉnh (ρ=0.362)** nhưng local_floor vẫn chặn theo nó

**Bằng chứng.** Log dòng 51 `ρ=0.362 < bar 0.50`. CSV: #23 local **1.135** → brain **−0.25**;
#24 local **0.968** → brain **−0.35** (đảo dấu!); #22 local 0.62 → brain 0.49; #7 local 1.00 →
brain 0.75. Local là **nhiễu**, đặc biệt cho họ fundamental/ts_corr.

**Gốc trong code.** Panel local ~478 mã yfinance không tính trung thực fundamental
(`operating_income`, `assets`, sparse + survivorship) và các cấu trúc `ts_corr` sâu. Nhưng
`LocalTunerRefiner` chỉ **bỏ qua** floor cho *alt-data thuần* (`local_usable(expr)==False`);
fundamental có field lọt panel vẫn bị `min_local_sharpe = calibrated_floor()` chặn — vừa **giết
oan** ý tưởng tốt, vừa **cho lọt** ý tưởng local-1.1/brain-âm tốn 8–19 phút sim. `CalibrationTracker`
**chỉ cảnh báo**, không có ai hành động theo ρ.

**Sửa.**
1. **Đóng vòng ρ:** khi `ρ < CALIBRATION_RHO_BAR`, tự động **nới/tắt** local_floor (hook đã có:
   `min_sharpe` opt-in trong `score_local_gate` — đảo logic: ρ không tin ⇒ **đừng** gate theo local
   sharpe, để Brain phân xử với ngân sách nhỏ). Đừng để floor chạy trên ranking ρ=0.36.
2. **Calibration + floor theo TỪNG HỌ:** theo dõi ρ riêng mỗi `family`; họ nào ρ thấp (hoặc field
   ngoài panel) → **abstain**: route thẳng Brain (như alt-data), không chấm local. Memory ghi rõ
   "local panel abstains rather than scores" cho fundamental/analyst/options — hiện code chưa làm.
3. Sửa `local_usable`/classifier: fundamental "tính-được-một-phần" phải xếp **abstain**, không score.

**Acceptance.** Trên tập held-out, ρ theo họ được log; không còn ứng viên local>0.9 mà brain<0
lọt vào sim (hoặc nếu lọt thì do abstain có chủ đích, budget-capped); tỉ lệ sim-đạt tăng.

### C5 — 🟠 HIGH: generator **tái sinh họ đã đóng** → đốt ~2 phút/batch sinh ra rác bị loại ngay

**Bằng chứng.** Log 12:40:59 "🚪 Đóng họ [pv_reversal]"; sau đó 12:42→13:04 và 17:54→18:16 hàng
loạt batch bị bỏ gần hết: "↩︎ Bỏ ý tưởng thuộc họ đã đóng [pv_reversal]". Mỗi `next_batch()` tốn
~2 phút LLM/GP nhưng sản lượng dùng được ≈ 0.

**Gốc trong code (đã định vị chính xác).** `ClosedLoop` **có** tham số `on_family_closed` và gọi
nó khi đóng họ — **nhưng `build_closed_loop` KHÔNG truyền `on_family_closed=`** khi khởi tạo
`ClosedLoop(...)` (xem cuối `closed_loop_adapters.py`: danh sách tham số thiếu hẳn nó). ⇒ Tập họ
bão hoà **không bao giờ** phản hồi về generator; `GPIdeaSource`/`GPEngine` tiếp tục mutate quanh
`VERIFIED_CORES` (toàn pv_reversal) và `CuratedIdeaSource` vẫn seed pv_reversal. Vòng lặp chỉ lọc
**sau khi đã trả tiền sinh**.

**Sửa.**
1. **Nối dây** `on_family_closed` trong `build_closed_loop` → `GPIdeaSource.set_saturated_families()`
   (thêm method) → lọc/không-seed họ đóng **bên trong** `next_batch()` (trước khi sinh), không chỉ
   sau đó. `CombinerIdeaSource`/`CuratedIdeaSource` cũng phải tôn trọng tập đóng.
2. **Đa dạng hoá seed:** `VERIFIED_CORES` hiện 100% pv_reversal → GP luôn quay về đó. Thêm seed
   core từ họ khác (fundamental có `ts_backfill`, momentum có điều kiện `trade_when`), hoặc giảm
   trọng số curated pv_reversal sau khi họ này đóng.
3. Khi *mọi* họ đóng → dừng gọn (`no_more_ideas`) thay vì sinh vô hạn rồi bỏ.

**Acceptance.** Sau khi một họ đóng, log **không** còn spam "Bỏ ý tưởng thuộc họ đã đóng"; thời
gian sinh/ứng-viên-dùng-được giảm mạnh (đo `gen_ms` / số candidate không-bị-bỏ trong SessionSummary).

### C6 — 🟡 MEDIUM: dedup **chưa bất biến theo hằng số/scale**; avoid-list chưa chặn cross-session

**Bằng chứng.** CSV #1 `multiply(4, multiply(-2, X))` vs #2 `multiply(2, multiply(-2, X))` →
`dedup_key` **khác nhau** (e0b19… vs f4568…), cả hai đều được xử lý. Log 17:35 còn **sim lại
đúng** các core pv_reversal mà 11:43 đã thử (avoid-list không giữ qua phiên).

**Gốc trong code.** `CanonicalHasher` chưa **fold hằng số** / bỏ hệ số dương thừa khi kết quả đi
vào `rank`/neutralize (bất biến thang). `VERIFIED_CORES` cũng chứa cả `multiply(2,…)` và
`multiply(1,…)` — sau neutralization là **cùng tín hiệu**.

**Sửa.** (a) Mở rộng `CanonicalHasher`: constant-fold + bỏ nhân-vô-hướng-dương ngoài cùng khi
nằm dưới `rank/scale/neutralize`. (b) Khử trùng `VERIFIED_CORES`. (c) Đảm bảo avoid-list keyed
theo canonical hash **persist cross-session** (repo `avoided_hashes()` — kiểm tra nó thực sự ghi
DB và được nạp đầu phiên; log 17:35 cho thấy nó **không** chặn core của phiên 11:43).

**Acceptance.** `multiply(4,X)` và `multiply(2,X)` cho **cùng** dedup_key; phiên 2 không sim lại
core phiên 1 đã thử.

### C7 — 🟠 HIGH (yield sâu): tín hiệu **chạm trần dưới ngưỡng nộp** kể cả khi hết bug

**Bằng chứng.** Brain sharpe tốt nhất 0.75; ngưỡng thật rất cao (`config/thresholds.py`:
`IS_LADDER_FAIL=1.58`, `IS_LADDER_PASS[2..5]=2.38`). Khoảng cách khổng lồ. Core đơn textbook
trần ~0.7 (đúng như skill/memory).

**Vấn đề phụ — config drift.** `VERIFIED_CORES` chú thích "Sharpe ~1.5+", nhưng phiên này pv_reversal
chỉ ra 0.75. `SIM_DEFAULTS.universe="TOP3000"` trong khi CSV sim ở **TOP1000** (Power Pool theme
override). ⇒ Core "đã kiểm chứng 1.57" có thể ở **universe/neutralization/decay khác** với config
đang chạy — cần soi lại provenance, đừng tin nhãn cũ.

**Sửa.**
1. **Edge đến từ conditioning + combination**, không phải core sạch hơn. `CombinerIdeaSource` đã
   có nhưng **0 combo đạt** — kiểm tra nó thực sự sinh combo (đủ `n_min` sub-signal có PnL local)
   và combo có chạm Brain không. Ưu tiên `trade_when` volume/event-gating trên core tốt nhất mỗi họ.
2. **Truy provenance `VERIFIED_CORES`:** re-sim đúng config từng đạt 1.57 (universe/neut/decay),
   ghi lại; nếu đã suy thoái do crowding → hạ vai trò seed, tập trung họ orthogonal (fundamental
   có `ts_backfill`, options/news alt-data).
3. Áp **Sharpe haircut** cho multiple-testing (skill) để không đuổi theo IS overfit.

**Acceptance.** Có ≥1 alpha combo hoặc conditioned đạt Brain sharpe > core đơn của cùng họ; bảng
provenance cho `VERIFIED_CORES` (config → brain sharpe hiện tại) được lưu.

### C8 — 🟡 MEDIUM: sim 8–19 phút/lần là trần throughput → chỉ giải được bằng C1/C4

Không tối ưu riêng được (sim WQ atomic, `TIMEOUT_SECONDS=1200`). Đòn bẩy thật là **sim ít hơn,
đúng hơn**: sửa C1 (đừng phí lượt vào op-invalid) + C4 (đừng sim ứng viên local mis-ranked). Cân
nhắc chạy sim **song song** nếu quota/tài khoản cho phép (nhiều `/simulations` đồng thời) — kiểm
tra `RateLimiter` và giới hạn account trước.

---

## 3. Bảng ưu tiên

| Ưu tiên | Mã | Việc | Sửa | Chi phí |
|---|---|---|---|---|
| **1** | C1 | pre-check theo live `/operators`; op chỉ-Brain đi thẳng Brain | Bật lại self-corr lever (yield) | Thấp–TB |
| **2** | C3 | Xoá `ts_std`; assert `GP_VOCAB ⊆ WQ_VALID_OPS` | Correctness + throughput | Thấp |
| **3** | C2 | Ghi đúng stage/fail_check cho pre-sim reject | Không còn mù chẩn đoán | Thấp |
| **4** | C5 | Nối `on_family_closed` → generator; đa dạng seed | Throughput (hết đốt 2ph/batch) | TB |
| **5** | C4 | Đóng vòng ρ; abstain theo họ; floor theo ρ | Yield + throughput | TB |
| **6** | C6 | Constant-fold canonical hash; avoid-list cross-session | Bớt sim trùng | Thấp |
| **7** | C7 | Combiner + conditioning; provenance VERIFIED_CORES | Vượt trần Sharpe (yield sâu) | Cao |

**Làm ngay (nửa ngày, mở khoá phần lớn):** C1 + C3 + C2. Ba lỗi correctness này đang âm thầm
vô hiệu hoá đòn bẩy self-corr, sinh operator rác, và giấu bug trong CSV.

## 4. File cần sửa (định vị sẵn)

- `src/simulation/simulator.py` — `simulate()`/`pre_sim_validator`: nguồn whitelist operator (C1);
  trả category reject (C2).
- `src/operators_local.py` + nơi nạp `/operators` cache (`src/data/…`) — tách `LOCAL_COMPUTABLE_OPS`
  vs `WQ_VALID_OPS` (C1).
- `src/gp/…` (từ vựng operator GP) — xoá `ts_std`; startup assert (C3).
- `src/app/closed_loop_adapters.py` — `build_closed_loop`: **truyền `on_family_closed=`** (C5);
  route op-chỉ-Brain qua `_sim_direct` (C1); `LocalTunerRefiner._finalize` gán stage/fail_check
  đúng (C2); abstain theo họ (C4).
- `src/pipeline/closed_loop.py` — `CalibrationTracker`: xuất ρ theo họ, kích hoạt nới floor (C4).
- `src/lang/visitors.py` — `CanonicalHasher`: constant-fold/scale-invariant (C6).
- `config/thresholds.py` — không đổi số; tham chiếu khi nới floor theo ρ (C4).
- `src/reporting/diagnostics.py` — `fail_check_from_reasons` thêm mã `OPERATOR_INVALID`/`FIELD_INVALID` (C2).

## 5. Kiểm chứng (single-variable, theo phong cách của bạn)

1. **Regression test operator:** startup fail-fast + unit test `GP_VOCAB ⊆ WQ_VALID_OPS`; test
   một expr `regression_neut(...)` **không** bị pre-check loại (C1/C3).
2. Chạy 1 phiên ngắn (`--max-ideas` nhỏ) sau C1+C2+C3 → đọc `session_summary`: phải xuất hiện
   `self_corr` thật cho ≥1 alpha, bucket `op_invalid` → 0.
3. Sau C5: log không còn spam "họ đã đóng"; sau C4: không còn local>0.9/brain<0 lọt sim.
4. `./venv/Scripts/python.exe -m pytest` xanh (canonicalize không đổi ngữ nghĩa — C6).
5. **Không auto-submit** (bước 7 QUY_TRINH) — submit vẫn cần bạn đồng ý.

---

### Ghi chú thứ tự đọc log (rút ra từ lần này)
CSV **một mình không đủ** — nó dán nhãn pre-sim reject thành `simmed/LOW_SHARPE` (C2). Luôn đối
chiếu `.log` (`simulator:simulate:254` = pre-check, `closed_loop:run:*` = funnel). Sau khi sửa C2,
CSV sẽ tự đủ để chẩn đoán.
