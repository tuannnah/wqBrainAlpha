# Claude Code — Task thực thi: sửa vòng kín MiniBrain (0 pass → submit-able)

> **Cách dùng:** mở repo `tuannnah/wqBrainAlpha` trong Claude Code, đưa file này vào và làm
> **tuần tự T1 → T7**. Mỗi task có: mục tiêu, file, thay đổi, cách verify, commit. Chẩn đoán
> đầy đủ ở `IMPROVEMENT_SPEC_v2.md` (đọc kèm nếu cần lý do).
>
> **Nguyên tắc bắt buộc:**
> - Trước khi sửa, `grep`/đọc code thật để xác nhận vị trí — số dòng/tên hàm dưới đây suy từ
>   snapshot `main`, có thể lệch. **Đừng hardcode giả định.**
> - Một task = một commit nhỏ, chạy `./venv/Scripts/python.exe -m pytest` trước khi commit.
> - **KHÔNG auto-submit** alpha. **KHÔNG** chạy `closed-loop` thật tốn quota trừ khi tôi bảo;
>   dùng unit test + phiên `--max-ideas` nhỏ để verify.
> - Code/spec bằng English; comment giữ phong cách repo (tiếng Việt) cũng được.

---

## T1 — 🔴 Sửa pre-check loại nhầm operator Brain (`regression_neut`) — BẬT LẠI self-corr lever

**Bug.** `pre_sim_validator` kiểm operator theo registry local (28 op backtest-được). Operator
chỉ-Brain (`regression_neut`, `vector_neut`, `ts_regression`…) không có trong 28 op → bị loại
"Operator không tồn tại" trước khi sim (log `simulator:simulate:254`). ⇒ mọi neutralization chết.

**Việc.**
1. Tìm nguồn whitelist của `pre_sim_validator` (bắt đầu ở `src/simulation/simulator.py` chỗ
   `self.pre_sim_validator`, lần ngược về `main.py`/adapter nơi nó được dựng). Xác định nó đang
   dùng tập operator nào.
2. Tạo/nạp **`WQ_VALID_OPS`** từ cache `/operators` live (đã có theo `QUY_TRINH_SINH_ALPHA.md`
   bước 2 — tìm `src/data/…` chỗ fetch/cache operators). Giữ **`LOCAL_COMPUTABLE_OPS`** (28 op
   trong `src/operators_local.py`) **chỉ** cho gate/backtest local.
3. `pre_sim_validator` (đường Brain) validate operator theo **`WQ_VALID_OPS`**, KHÔNG theo local.
4. Biểu thức dùng operator không local-computable → **đi thẳng Brain** như nhánh alt-data
   (`LocalTunerRefiner._sim_direct` trong `src/app/closed_loop_adapters.py`): mở rộng điều kiện
   rẽ nhánh "sim trực tiếp" để bao cả expr chứa operator chỉ-Brain, không chỉ field ngoài panel.
5. **Verify LIVE**: viết script nhỏ (hoặc test đánh dấu `@pytest.mark.live`) in ra `regression_neut`,
   `vector_neut` có trong `/operators` của account không. Nếu **không có** → dùng `vector_neut`,
   hoặc chuyển hạ self-corr sang **neutralization SETTING** (STATISTICAL/CROWDING) thay vì operator.

**Verify.** Unit test: một expr `regression_neut(multiply(-1, ts_mean(subtract(close, vwap), 10)),
rank(volume))` **không** bị `pre_sim_validator` loại (mock catalog chứa `regression_neut`).

**Commit.** `fix(sim): validate pre-sim operators against live WQ catalog, route Brain-only ops direct`

---

## T2 — 🔴 Xoá operator rác khỏi từ vựng GP (`ts_std` → `ts_std_dev`) + fail-fast

**Bug.** GP sinh `ts_std` (không phải op WQ) → pre-check loại (log dòng 166).

**Việc.**
1. `grep -rn "ts_std\b"` trong `src/gp/` và config từ vựng operator. Thay `ts_std` → `ts_std_dev`
   (kiểm signature). Xoá mọi operator không có trong `WQ_VALID_OPS`.
2. Thêm **startup assertion** (nơi dựng GP/registry): `assert GP_VOCAB <= WQ_VALID_OPS`, nếu lệch
   thì raise + in operator thừa. Với nhánh cần backtest local: `GP_VOCAB_LOCAL <= LOCAL_COMPUTABLE_OPS`.

**Verify.** Unit test `test_gp_vocab_subset_of_wq_operators`; test riêng `ts_std` không còn trong vocab.

**Commit.** `fix(gp): purge invalid operator ts_std; assert GP vocab ⊆ live WQ catalog`

---

## T3 — 🔴 Ghi đúng nhãn instrumentation cho pre-sim reject (đừng gọi là `simmed/LOW_SHARPE`)

**Bug.** Pre-sim reject (operator/field/depth) bị `_finalize` dán `stage_reached="simmed"`,
`fail_check="LOW_SHARPE"`, và **đếm là 1 sim** dù chưa chạm Brain (CSV `sim_ms≈0.7ms`).

**Việc.**
1. `src/simulation/simulator.py`: khi `pre_sim_validator` loại, trả `SimulationResult` mang
   **category** (`presim_reason` + loại: `OPERATOR_INVALID`/`FIELD_INVALID`/`DEPTH`), không chỉ
   `status="error"`.
2. `src/reporting/diagnostics.py::fail_check_from_reasons`: thêm mã `OPERATOR_INVALID`,
   `FIELD_INVALID`.
3. `src/app/closed_loop_adapters.py`: chỗ nhận result pre-sim reject → tạo `IdeaOutcome` với
   `stage_reached="op_invalid"`/`"field_invalid"`, `fail_check` đúng, **`sims_used=0`**
   (không đếm quota Brain). Thêm cột CSV `presim_reason`, `is_brain_sim` (RunAlphaLogger + schema).
4. `SessionSummary`: thêm bucket `op_invalid`/`field_invalid` vào funnel.

**Verify.** Test: pre-sim reject → outcome có `stage_reached="op_invalid"`, `sims_used=0`,
`is_brain_sim=False`. Chạy phiên nhỏ → CSV phân biệt op-invalid vs low-sharpe.

**Commit.** `feat(instrumentation): honest stage/fail_check for pre-sim rejects; add presim_reason,is_brain_sim`

---

## T4 — 🟠 Nối dây `on_family_closed` → generator (hết đốt ~2 phút/batch sinh họ đã đóng)

**Bug.** `ClosedLoop` gọi `on_family_closed` khi đóng họ, nhưng `build_closed_loop` **không truyền**
`on_family_closed=` → generator không biết → cứ sinh pv_reversal rồi bị loại sau khi đã tốn sinh.

**Việc.**
1. `src/app/closed_loop_adapters.py::GPIdeaSource`: thêm `self._saturated: set[str] = set()` +
   method `set_saturated_families(fams)`; trong `next_batch()`/`_run_one_batch()` **lọc bỏ**
   candidate thuộc họ bão hoà **trước khi trả** (dùng `classify_family`). `CombinerIdeaSource`,
   `CuratedIdeaSource` cũng tôn trọng tập này.
2. `build_closed_loop`: truyền `on_family_closed=idea_source.set_saturated_families` vào
   `ClosedLoop(...)` (hiện đang thiếu tham số này ở call cuối hàm).
3. Khi **mọi** họ đóng → generator trả batch rỗng → ClosedLoop dừng `no_more_ideas` (đừng loop vô hạn).

**Verify.** Test: đóng họ `pv_reversal` → batch kế tiếp của GPIdeaSource không chứa expr
`classify_family == pv_reversal`. Phiên nhỏ: log không còn spam "Bỏ ý tưởng thuộc họ đã đóng".

**Commit.** `fix(closedloop): wire on_family_closed to generator; skip saturated families before generation`

---

## T5 — 🟠 Đóng vòng calibration ρ + abstain theo họ (đừng gate local khi ρ không tin)

**Bug.** ρ=0.362 (log dòng 51) nhưng local_floor vẫn chặn theo local sharpe → giết oan + cho lọt
sim 8–19 phút (CSV: local 1.1 → brain −0.3). `CalibrationTracker` chỉ cảnh báo, không ai hành động.

**Việc.**
1. `src/pipeline/closed_loop.py::CalibrationTracker`: expose ρ hiện tại + (nếu khả thi) ρ **theo
   họ** từ `repo.brain_local_sharpe_pairs()` (thêm nhãn family vào cặp).
2. `LocalTunerRefiner` (adapter): khi ρ toàn cục < `CALIBRATION_RHO_BAR` **hoặc** ρ của họ đó thấp
   → **bỏ qua** `min_local_sharpe` floor cho họ đó và route thẳng Brain với **budget nhỏ**
   (abstain, giống alt-data). Dùng hook `min_sharpe` opt-in sẵn có trong `score_local_gate` — đảo
   theo ρ thay vì bật cứng.
3. Classifier: fundamental/ts_corr "tính-được-một-phần" trên panel → xếp **abstain**, không score
   (khớp memory "local panel abstains rather than scores").

**Verify.** Test: ρ<bar → refiner không chặn theo local_floor cho họ ρ thấp. Phiên nhỏ: không còn
ứng viên local>0.9/brain<0 lọt sim ngoài ý muốn (hoặc chỉ do abstain, budget-capped).

**Commit.** `feat(calibration): act on rho — per-family abstain, disable local floor when untrusted`

---

## T6 — 🟡 Canonical dedup bất biến hằng số/scale + avoid-list cross-session

**Bug.** `multiply(4,X)` vs `multiply(2,X)` → dedup_key khác (CSV #1/#2); phiên 17:35 sim lại core
phiên 11:43 (avoid-list không giữ qua phiên).

**Việc.**
1. `src/lang/visitors.py::CanonicalHasher`: constant-fold; bỏ nhân-vô-hướng-dương ngoài cùng khi
   nằm dưới `rank/scale/neutralize` (bất biến thang). Giữ nguyên ngữ nghĩa — thêm test đối chứng.
2. Khử trùng `VERIFIED_CORES` trong `closed_loop_adapters.py` (bỏ biến thể collapse sau neutralize).
3. Kiểm `repo.avoided_hashes()` thực sự **ghi + nạp** cross-session (DB theo email). Sửa nếu không.

**Verify.** Test: `hash(multiply(4,X)) == hash(multiply(2,X))` khi X đi vào rank. Phiên 2 không
sim lại core phiên 1. `pytest` toàn bộ xanh (canonicalize không đổi kết quả backtest).

**Commit.** `fix(dedup): constant-fold canonical hash; persist avoid-list across sessions`

---

## T7 — 🟠 Yield: kiểm Combiner thực chạy + conditioning + provenance VERIFIED_CORES

**Bug.** Core đơn trần ~0.75 Brain sharpe, xa ngưỡng 1.58. `CombinerIdeaSource` có nhưng 0 combo
đạt. `VERIFIED_CORES` nhãn "1.57" nhưng đang sim ở universe/config khác (SIM_DEFAULTS TOP3000 vs
CSV TOP1000).

**Việc (điều tra trước, sửa sau).**
1. Log/đếm trong `CombinerIdeaSource.next_batch`: bao nhiêu sub-signal có PnL local (đủ `n_min`?),
   bao nhiêu combo sinh ra, bao nhiêu chạm Brain. Nếu ~0 → sửa nguồn sub-signal (curated/alt-data
   `pnl` rỗng nên bị loại khỏi `_signals`) để combo thật sự hình thành.
2. Thêm bước **conditioning** `trade_when(volume/event, best_core, exit)` cho core tốt nhất mỗi họ
   (config stage, sau khi core có edge) — theo skill: edge từ conditioning, không từ core sạch hơn.
3. **Provenance:** viết script re-sim từng `VERIFIED_CORES` ở đúng config từng đạt 1.57
   (universe/neutralization/decay), lưu bảng `core → brain_sharpe hiện tại`. Nếu suy thoái do
   crowding → hạ vai trò seed, dồn sang họ orthogonal.

**Verify.** ≥1 combo/conditioned alpha có Brain sharpe > core đơn cùng họ. Bảng provenance được lưu.

**Commit.** `feat(yield): fix combiner signal sourcing; add trade_when conditioning; VERIFIED_CORES provenance`

---

## Thứ tự & Definition of Done

**Làm ngay (mở khoá nhiều nhất):** T1 → T2 → T3 (3 lỗi correctness, ~nửa ngày). Sau đó T4 → T5
(throughput/yield), rồi T6, cuối là T7 (yield sâu).

**Done cho cả đợt:**
- `pytest` xanh; startup assert operator-vocab pass.
- Chạy `closed-loop --max-ideas` nhỏ 1 lần: `session_summary` cho thấy (a) bucket `op_invalid`=0,
  (b) có `self_corr` thật cho ≥1 alpha, (c) không spam "họ đã đóng", (d) không có local>0.9/brain<0
  lọt sim ngoài abstain.
- Không auto-submit; mọi thay đổi ngưỡng vẫn tập trung ở `config/thresholds.py`.
