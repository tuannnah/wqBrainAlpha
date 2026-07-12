# Kế hoạch: Đưa closed-loop tới alpha SUBMIT ĐƯỢC (2026-07-12)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sau 2 phiên chạy thật (07-11: 22 ý tưởng/14 sim/0 pass; 07-12: 13 ý tưởng/9 sim/0 pass)
với TOÀN BỘ fix review 07-11 đã áp, sửa các nút thắt còn lại để phiên kế tiếp ra ≥1 alpha
`failed_checks == []` (submit được).

**Architecture:** Không đập lại engine. 4 đòn bẩy theo thứ tự: (1) sửa Combiner đang chết im lặng
(0 combo mọi batch) — con đường √N duy nhất đưa các tín hiệu ~1.05 đã có vượt ngưỡng ~1.58;
(2) kỷ luật quota — cắt ngân sách GP rác + chặn degenerate trước sim; (3) biến alt-data 1-shot
thành mini-sweep (sign/decay) + multi-simulation để tăng throughput; (4) mở rộng field verify LIVE.

**Tech Stack:** Python 3.12, pytest, SQLAlchemy/SQLite, requests (WQ Brain API), loguru.

## Chẩn đoán nền (bằng chứng — subagent PHẢI đọc trước khi code)

Nguồn: `logs/wq_alpha_2026-07-11.log`, `logs/wq_alpha_2026-07-12.log`,
`logs/alphas_2026-07-12_205508.csv`, `logs/session_summary_2026-07-11_154541.md`,
`docs/tailieu/review20260711/FIX_PLAN_2026-07-11.md`.

1. **Ứng viên sát ngưỡng nhất đã có sẵn:** họ close-vwap Sharpe **1.04–1.07**, self-corr
   **0.40–0.46** (log 07-11 ý tưởng #1–#3) — qua HẾT gate trừ LOW_SHARPE (thang IS-Ladder cần
   ~1.58). Ghép 2–3 tín hiệu ít tương quan cỡ này (Grinold–Kahn √N) là đường khả dĩ nhất tới 1.58.
2. **CombinerIdeaSource luôn `-> 0 combo`** dù `n_db=50` (mọi batch cả 2 phiên). 3 tầng lọc im
   lặng trong `src/pipeline/combine_stage.py::combine_stage`: (a) không dựng được dưới trần depth,
   (b) `scored.verdict.passed == False` — gate local CÓ pool-corr, mà **tín hiệu con từ DB chính
   là thành viên pool** (`good_signals_for_combine` join `PoolPnlModel`) → combo tự-tương-quan với
   chính con mình → nghi rớt gate độc đáo oan; (c) `fitness combo <= best component`.
3. **Alt-data/analyst đi thẳng Brain chỉ được ĐÚNG 1 sim/ý tưởng**
   (`LocalTunerRefiner._sim_direct`, `closed_loop_adapters.py:211`): không flip sign, không sweep
   decay/neut. 07-12: anl4 revision 0.64 rồi vứt; trước đó seed social **sai dấu** (-0.48).
4. **GP vẫn đốt ~50% sim vào rác:** 07-12 các sim GP ra Sharpe -0.13/0.00/0.31/0.15;
   `power(sign(...), 2)` (hằng số, TO=0) vẫn lọt sim. Calibration **ρ=0.308 < bar 0.5** → local
   ranking không tin được, floor local không cứu nổi quota cho GP.
5. **Sim tuần tự 3.5–17 phút/cái**, chưa dùng multi-simulation (không có `multi` nào trong
   `src/simulation/`). 5 seed alt-data 07-12 tốn ~57 phút chờ nối đuôi.
6. **Field guard chặn `days_to_cover`/`shares_short`** (log 07-12 dòng 4–5) — họ short-interest
   chưa bao giờ được thử vì field không có trong catalog cache (chưa verify LIVE dataset nào chứa).

## Global Constraints

- **TDD bắt buộc**: test fail trước → code → test xanh → commit. Mỗi task 1 commit, message tiếng Việt.
- **Chạy test bằng venv**: `./venv/Scripts/python.exe -m pytest -q` (python hệ thống thiếu `lark`).
  Baseline hiện tại: ~1240 passed, 1 fail pre-existing (`tests/test_db_postgres.py`, thiếu psycopg — bỏ qua).
- **Ngưỡng gate KHÔNG hardcode** — mọi số nằm ở `config/thresholds.py`.
- **Chỉ dùng field đã verify LIVE** qua `get_datafields` — không bịa field (cardinal rule #1).
- **Không auto-submit** — submit là hành động không đảo ngược, chỉ user quyết.
- **Code/comment/log tiếng Việt**, giữ nguyên dấu (không viết "nao" thay "não").
- Task nào cần **login Brain thật** (QR terminal) chỉ viết code + script; việc chạy live ghi rõ
  "USER chạy" trong DoD — subagent KHÔNG tự sim/đốt quota.
- File lớn `src/app/closed_loop_adapters.py` (~800 dòng): đọc vùng liên quan trước khi sửa,
  không tái cấu trúc ngoài phạm vi task.

---

### Task 1 — Chẩn đoán offline: vì sao Combiner ra 0 combo (KHÔNG sửa code engine)

**Files:**
- Create: `tools/diag_combiner.py`
- Đọc để hiểu: `src/app/closed_loop_adapters.py:641-756` (CombinerIdeaSource),
  `src/pipeline/combine_stage.py`, `src/generation/combiner.py`,
  `src/storage/repository.py:318-369` (`load_pool`, `good_signals_for_combine`).

**Interfaces:**
- Consumes: DB SQLite thật `wq_alpha_<slug>.db` (đọc `.wq_account` để lấy slug; xem
  `main.py` cách mở DB theo account). Panel local `data/market_yf*` (dò thư mục có
  `returns.parquet`).
- Produces: báo cáo `logs/diag_combiner_<YYYYMMDD>.md` — Task 2 đọc báo cáo này để chọn fix.

**Việc:** Script chạy OFFLINE (không login, không sim) tái hiện đúng đường combiner trên DB thật
và in **lý do rớt ở từng tầng**:

- [ ] **Step 1:** Viết `tools/diag_combiner.py` thực hiện tuần tự và in số liệu mỗi bước:
  1. `repo.good_signals_for_combine(limit=50)` → in số tín hiệu, phân bố fitness, 10 expr đầu.
  2. `select_decorrelated_combos(signals, tau=0.30, n_min=2, n_max=4, max_combos=5)` → in số
     combo thô + kích thước từng combo. Nếu 0: in ma trận |ρ| đôi một của top-10 tín hiệu
     (dùng `pairwise_abs_rho` từ `src/backtest/pool_corr.py`) để chứng minh "tất cả trùng nhau".
  3. Với từng combo thô: `build_combined_expression(...)` → nếu `None` in "RỚT: depth".
  4. `_score_one_full(built.expr, config, data, pool=repo.load_pool())` → in `verdict.passed`
     + `verdict.reasons` (RỚT: gate — ghi rõ reason nào, đặc biệt reason độc đáo/pool-corr).
  5. So `scored.metrics.fitness` với fitness từng sub-expr (RỚT: không vượt trội).
  Dựng `config`/`data`/`registry` đúng cách main.py dựng cho closed-loop (đọc
  `main.py::_run_closed_loop_session` và bắt chước phần dựng ParquetSource + PortfolioConfig).
- [ ] **Step 2:** Chạy: `./venv/Scripts/python.exe tools/diag_combiner.py` → ghi kết quả vào
  `logs/diag_combiner_<YYYYMMDD>.md` (script tự ghi file, in cả stdout).
- [ ] **Step 3:** Kết luận trong báo cáo: tầng nào giết combo (depth / gate-pool / không-vượt-trội /
  greedy-0-combo-vì-tương-quan), kèm số liệu. Nêu rõ fix đề xuất cho Task 2.
- [ ] **Step 4:** Commit: `git add tools/diag_combiner.py logs/diag_combiner_*.md && git commit -m "chẩn đoán: combiner 0-combo — tái hiện offline từng tầng lọc"`

**DoD:** Báo cáo chỉ đích danh tầng lọc giết combo trên DB thật, có số liệu; không sửa engine.

---

### Task 2 — Fix Combiner: instrument lý do rớt + loại tự-so-với-chính-mình + tiêu chí vượt trội theo điểm-nộp

**Files:**
- Modify: `src/pipeline/combine_stage.py`, `src/app/closed_loop_adapters.py:641-756`,
  `src/storage/repository.py:339-369`, `src/generation/combiner.py` (SubSignal)
- Test: `tests/unit/test_combine_stage.py`, `tests/unit/test_closed_loop_adapters.py` (case mới)

**Interfaces:**
- Consumes: báo cáo Task 1 (`logs/diag_combiner_*.md`) — đọc TRƯỚC, fix theo tầng đã chứng minh.
- Produces: `combine_stage(...)` trả thêm thống kê rớt: hàm nhận tham số mới
  `drop_stats: dict[str, int] | None = None` (mutate in-place: khóa
  `"depth" | "gate" | "not_better" | "greedy_empty"`); `SubSignal` thêm field
  `eval_id: int | None = None`; `good_signals_for_combine` trả thêm `evaluation_id`
  (tuple 5 phần tử `(expr, dates, pnl, fitness, eval_id)`); `CombinerIdeaSource.last_stats`
  có thêm khóa `drop_*`.

Ba fix (áp cả 3 — độc lập nhau, mỗi fix 1 vòng TDD; nếu Task 1 chứng minh fix nào vô nghĩa
với dữ liệu thật thì ghi chú trong commit và vẫn giữ instrument):

- [ ] **Step 1 (instrument):** Test: gọi `combine_stage` với score_fn giả luôn fail gate →
  `drop_stats["gate"] == số combo thô`. Implement: đếm tại 3 điểm `continue` + trường hợp
  `select_decorrelated_combos` trả rỗng. `CombinerIdeaSource.next_batch` truyền dict và log
  `logger.info("Combiner drop: depth={} gate={} not_better={} ...")` + gộp vào `last_stats`.
- [ ] **Step 2 (tự-so):** Test: pool chứa đúng PnL của 2 sub-signal (eval_id 1, 2); combo từ
  chúng phải được chấm với pool ĐÃ LOẠI 2 eval_id đó (giả lập `_score_one_full` bắt được pool
  nhận vào). Implement: `good_signals_for_combine` trả `eval_id`; `CombinerIdeaSource._score_fn`
  đổi thành factory `_score_fn_for(combo)` — dựng `pool = {k: v for k, v in full_pool.items()
  if k not in {s.eval_id for s in combo if s.eval_id}}`; `combine_stage` nhận
  `score_fn_factory: Callable[[list[SubSignal]], Callable[[str], _Scored]] | None = None`
  (ưu tiên factory khi có, giữ `score_fn` cũ cho tương thích test hiện hữu).
- [ ] **Step 3 (vượt trội):** Test: combo Sharpe cao/fitness ngang component tốt nhất vẫn được
  nhận khi điểm-nộp `min(sharpe/1.25, fitness/1.0)` vượt component (hằng 1.25/1.0 lấy từ
  `config/thresholds.py` — nếu chưa có hằng điểm-nộp thì thêm `SUBMIT_SHARPE_REF = 1.25`,
  `SUBMIT_FITNESS_REF = 1.0` vào đó, KHÔNG hardcode tại chỗ). Thay so sánh
  `fitness <= best_component` bằng so sánh điểm-nộp.
- [ ] **Step 4:** `./venv/Scripts/python.exe -m pytest tests/unit/test_combine_stage.py tests/unit/test_closed_loop_adapters.py -q` xanh, rồi chạy toàn bộ `-q` xác nhận không vỡ gì.
- [ ] **Step 5:** Chạy lại `tools/diag_combiner.py` (Task 1) → kỳ vọng >0 combo trên DB thật
  (nếu vẫn 0, drop_stats phải nói rõ vì sao — ghi vào báo cáo).
- [ ] **Step 6:** Commit từng fix riêng (3 commit: instrument / tự-so / điểm-nộp).

**DoD:** `diag_combiner` ra ≥1 combo trên DB thật HOẶC drop_stats chứng minh mọi tín hiệu DB
tương quan ≥0.3 đôi một (khi đó ghi rõ: cần tín hiệu orthogonal mới từ Task 6/7 trước). Không
còn 0-combo im lặng.

---

### Task 3 — Cap ngân sách sim GP mỗi phiên (ưu tiên quota cho seed/combiner)

**Files:**
- Modify: `src/pipeline/closed_loop.py` (vòng `run`), `src/app/closed_loop_adapters.py`
  (`GPIdeaSource`, `CuratedIdeaSource`, `AltDataIdeaSource`, `CombinerIdeaSource` — gắn nhãn
  origin), `main.py` (CLI flag `--max-gp-sims`, mặc định 3)
- Test: `tests/unit/test_closed_loop.py` (case mới)

**Interfaces:**
- Consumes: `ShortlistCandidate` (xem định nghĩa trong `src/pipeline/` — đọc trước; nếu là
  dataclass frozen thì thêm field `origin: str = "gp"` có default để không vỡ constructor cũ).
- Produces: `ClosedLoop(..., max_gp_sims: int | None = 3)`; mỗi IdeaSource đặt
  `origin` tương ứng (`"curated" | "alt_data" | "combiner" | "gp"`).

Lý do: 2 phiên gần nhất GP đốt ~10 sim (≈50% quota) → 0 giá trị; nguồn nhiễu khi ρ=0.308.

- [ ] **Step 1:** Test: `ClosedLoop` với `max_gp_sims=1`, refiner giả đếm sim — candidate thứ 2
  origin `"gp"` KHÔNG được sim (outcome stage `"gp_budget"`, `fail_check="GP_BUDGET"`,
  sims_used=0), candidate origin `"curated"` vẫn sim bình thường.
- [ ] **Step 2:** Implement: đếm sim đã dùng bởi outcome có `is_brain_sim=True` và candidate
  origin `"gp"`; chạm trần → bỏ qua refine+sim, ghi outcome trung thực vào session_summary
  (giữ nguyên đường ghi CSV — xem `IdeaOutcome` các field `stage_reached/fail_check`).
- [ ] **Step 3:** Wire `main.py`: `--max-gp-sims` (mặc định 3) → `build_closed_loop` → `ClosedLoop`.
- [ ] **Step 4:** pytest toàn bộ xanh → commit.

**DoD:** Phiên tới GP tối đa 3 sim Brain; funnel CSV ghi rõ ứng viên GP bị chặn vì ngân sách.

---

### Task 4 — Chặn degenerate signal trước sim (position gần hằng số / thiếu hướng giá)

**Files:**
- Đọc trước: commit `8fd2353` (meaningfulness filter hiện có — tìm file bằng
  `git show 8fd2353 --stat`), `src/pipeline/runner.py::_score_one_full`
- Modify: file meaningfulness filter hiện có + `src/app/closed_loop_adapters.py`
  (điểm gọi trước sim trong `LocalTunerRefiner`)
- Test: file test meaningfulness hiện có (thêm case)

**Interfaces:**
- Consumes: AST qua `parse()` + registry; kết quả backtest local (`tr.local_metrics`).
- Produces: filter mở rộng — cùng chữ ký hàm hiện có, thêm rule mới.

Bằng chứng phải chặn (từ log thật — dùng làm test case):
- `power(sign(trade_when(multiply(volume, volume), divide(returns, open), sign(vwap))), 2)` —
  hằng số (sign²), sim thật ra Sharpe 0.00/TO 0.00 (07-12 ý tưởng #11).
- `multiply(-1, multiply(ts_mean(ts_mean(volume, 120), 120), ts_mean(ts_delta(volume, 3), 10)))` —
  chỉ volume, không hướng giá (07-12 #7, Sharpe -0.13).

- [ ] **Step 1:** Test AST-rule: (a) `power(sign(X), k)` với k chẵn → reject "hằng số";
  (b) biểu thức mà TẬP FIELD ⊆ {volume, adv20, sharesout...} (nhóm volume-only — liệt kê từ
  catalog field local, xem `src/data/fields.py`) và không có `returns/close/open/high/low/vwap`
  → reject "không có hướng giá".
- [ ] **Step 2:** Test backtest-rule (rẻ, chỉ đường local-usable): sau backtest local, nếu
  `turnover < 0.005` VÀ `abs(sharpe) < 0.05` → reject "degenerate position" trước khi sim
  (ngưỡng đặt trong `config/thresholds.py`: `DEGENERATE_TURNOVER = 0.005`,
  `DEGENERATE_SHARPE = 0.05`).
- [ ] **Step 3:** Implement 2 rule, wire vào đúng chỗ filter hiện có đang được gọi (giữ nguyên
  luồng, chỉ thêm rule) + điểm sau `self._tune` trong `LocalTunerRefiner.refine_and_sim`.
- [ ] **Step 4:** pytest xanh toàn bộ → commit.

**DoD:** 2 biểu thức bằng chứng trên bị chặn local với lý do rõ, 0 sim; không chặn oan seed
`VERIFIED_CORES`/`ALT_DATA_CORES`/`FUNDAMENTAL_CORES` (viết test khẳng định các seed này qua filter).

---

### Task 5 — Mini-sweep cho đường sim-thẳng: flip sign + 1 biến thể decay (ngân sách có trần)

**Files:**
- Modify: `src/app/closed_loop_adapters.py:211-237` (`_sim_direct`), `main.py`
  (flag `--alt-sweep-budget`, mặc định 2), `config/thresholds.py`
  (`ALT_SWEEP_MIN_ABS_SHARPE = 0.5`)
- Test: `tests/unit/test_closed_loop_adapters.py` (case mới, simulator giả)

**Interfaces:**
- Consumes: `Simulator.simulate(expression, settings) -> SimulationResult`
  (`src/simulation/simulator.py:254`), `SimConfig.with_overrides(...)`
  (`src/simulation/config.py:82`).
- Produces: `LocalTunerRefiner(..., alt_sweep_budget: int = 2)`; `_sim_direct` trả outcome của
  **kết quả tốt nhất** trong ≤ 1 + alt_sweep_budget sim.

Logic (đơn giản, dựa bằng chứng "seed sai dấu" + "1-shot 0.64 rồi vứt"):
1. Sim core như hiện tại (sim #1).
2. Nếu `sharpe <= -ALT_SWEEP_MIN_ABS_SHARPE` → sim `multiply(-1, <expr>)` (flip dấu; nếu expr
   gốc dạng `multiply(-1, X)` thì bóc thành `X` thay vì bọc chồng — dùng parse/AST, không xử lý chuỗi).
3. Nếu `ALT_SWEEP_MIN_ABS_SHARPE <= sharpe` và chưa pass → sim lại best-so-far với
   `decay` khác (4→8 nếu đang 4, ngược lại →4) qua `sim_cfg.with_overrides(decay=...)`.
4. Dừng khi hết budget hoặc `status == "passed"`. Outcome cuối = sim có điểm-nộp cao nhất;
   `sims_used` = tổng sim thật đã đốt (sửa `_finalize` nhận `sims_used: int = 1`).

- [ ] **Step 1:** Test với simulator giả trả kịch bản: (a) sharpe -0.9 → có đúng 2 lần
  `simulate`, lần 2 với expr đã flip, outcome lấy kết quả tốt hơn, `sims_used == 2`;
  (b) sharpe 0.2 → đúng 1 sim (dưới ngưỡng sweep); (c) budget 0 → đúng 1 sim.
- [ ] **Step 2:** Implement theo logic trên (giữ nhánh `presim_reason` như cũ — không sweep
  biểu thức bị pre-filter chặn).
- [ ] **Step 3:** Wire flag CLI + pytest toàn bộ xanh → commit.

**DoD:** Test 3 kịch bản xanh; seed sai dấu giờ tự cứu bằng 1 sim thêm thay vì vứt hypothesis.

---

### Task 6 — Multi-simulation: sim nhiều biểu thức trong 1 lần chờ

**Files:**
- Điều tra trước (BẮT BUỘC, ghi kết quả vào docstring): định dạng multi-sim của WQ Brain —
  POST `/simulations` với body là MẢNG payload (tối đa 10), poll parent → children. Nguồn:
  (a) đọc source wqb-mcp tool `create_multi_simulation` (server đã cài local — tìm trong
  `%APPDATA%/../local` hoặc nơi cài npm/pip của wqb-mcp), (b) `docs/worldquantbrain/docs/`
  (74 file đã tải), (c) forum WQ nếu cần.
- Modify: `src/simulation/simulator.py` (thêm `simulate_many`), `src/app/closed_loop_adapters.py`
  (`AltDataIdeaSource` batch → `_sim_direct_many`)
- Test: `tests/unit/test_simulator.py` (case mới với client giả)

**Interfaces:**
- Consumes: `self.client` (session HTTP đã auth — xem cách `simulate` dùng), pre_sim_validator.
- Produces: `Simulator.simulate_many(jobs: list[tuple[str, dict | None]]) ->
  list[SimulationResult]` — giữ nguyên thứ tự jobs; job bị pre-filter chặn trả
  `SimulationResult(presim_reason=...)` mà KHÔNG chiếm slot trong payload gửi Brain.

- [ ] **Step 1:** Điều tra + ghi lại format request/response thật (payload mẫu, header Location,
  cấu trúc children) vào docstring `simulate_many` — KHÔNG đoán mò.
- [ ] **Step 2:** Test với client giả: POST 1 lần body mảng 3 payload → poll parent trả children
  → GET từng child → 3 `SimulationResult` đúng thứ tự; 1 job pre-filter chặn → chỉ 2 payload
  được POST; lỗi 429 → raise `QuotaExceededError` như đường đơn.
- [ ] **Step 3:** Implement `simulate_many` (tái dùng `_poll`/parse child như `simulate`; timeout
  chung `TIMEOUT_SECONDS` cho parent, mỗi child poll nối tiếp phần dư).
- [ ] **Step 4:** Dùng trong đường alt-data: gom cả batch seed sim-thẳng (5–8 core/phiên hiện
  chạy tuần tự ~1h) thành 1 lần `simulate_many`; sweep Task 5 cũng đi qua `simulate_many` khi
  ≥2 biến thể. Giữ fallback tuần tự khi lỗi multi-sim (log warning, không chết phiên).
- [ ] **Step 5:** pytest toàn bộ xanh → commit (2 commit: simulator / wiring).

**DoD:** Test client-giả xanh; đường alt-data mặc định đi multi-sim; fallback tuần tự có test.
Nghiệm thu live (USER chạy menu-5): 5 seed alt-data xong trong ~1 lần chờ thay vì ~5 lần.

---

### Task 7 — Verify LIVE dataset/field mới + sửa seed short-interest

**Files:**
- Create: `tools/verify_datasets.py`
- Modify: `src/generation/` (file seed chứa `days_to_cover`/`shares_short` — grep để tìm;
  cập nhật field id đúng hoặc gỡ seed nếu account không có dataset)
- Test: test seed hiện có (cập nhật theo field mới)

**Interfaces:**
- Consumes: client đã auth (`main.py` cách dựng), API `get_datasets` + `get_datafields`
  (xem `src/data/fields.py` đang gọi thế nào).
- Produces: `logs/verified_fields_<YYYYMMDD>.json` — bảng `{dataset_id: [field_id, ...]}`
  các field CÓ THẬT cho account/region/universe/delay hiện tại.

- [ ] **Step 1:** Viết `tools/verify_datasets.py`: liệt kê toàn bộ dataset khả dụng
  (USA/TOP3000/delay-1), với mỗi dataset thuộc nhóm quan tâm (short interest, news, earnings,
  insider, analyst, option, sentiment) tải danh sách field + coverage, ghi JSON + in bảng tóm tắt.
  Script chỉ GỌI API đọc (get_*), không sim — an toàn quota.
- [ ] **Step 2:** USER chạy: `./venv/Scripts/python.exe tools/verify_datasets.py` (cần session
  còn hạn). Subagent dừng ở đây nếu không có session — ghi rõ trong báo cáo.
- [ ] **Step 3 (sau khi có JSON):** Đối chiếu seed short-interest hiện tại: nếu tồn tại field id
  khác cho short interest (vd trong dataset khác) → sửa seed dùng field ĐÃ verify + giữ luận
  điểm (docstring hypothesis 4 phần); nếu account không có → xóa seed, ghi chú trong commit.
  Cập nhật test seed tương ứng.
- [ ] **Step 4:** pytest xanh → commit.

**DoD:** Không còn log "Field guard: bỏ qua core..." cho seed có chủ đích; có bảng field verify
LIVE làm nguồn cho seed tương lai.

---

### Task 8 — Báo cáo submit-ready cuối phiên + đường nộp rõ ràng

**Files:**
- Đọc trước: lệnh `submit`/`top` trong `main.py`, `src/pipeline/closed_loop.py::_report`
- Modify: `src/pipeline/closed_loop.py` (_report), `docs/QUY_TRINH_SINH_ALPHA.md` (mục Submit)
- Test: `tests/unit/test_closed_loop.py` (case _report)

**Interfaces:**
- Consumes: DB (`AlphaModel`/`SimulationModel` — alpha `failed_checks == []`, self-corr đã verify).
- Produces: cuối phiên in khối "**SẴN SÀNG NỘP**": danh sách alpha trong DB đạt
  `status=passed AND failed_checks==[] AND self_corr<0.70` kèm lệnh nộp chính xác
  (`./venv/Scripts/python.exe main.py submit --no-dry-run`), phân biệt rạch ròi với
  "Power Pool eligible" (chỉ là structural, KHÔNG nộp được — commit `e27821d`).

- [ ] **Step 1:** Test: repo giả có 1 alpha đạt chuẩn → `_report` in khối SẴN SÀNG NỘP với
  wq_alpha_id; repo không có → in "0 alpha sẵn sàng".
- [ ] **Step 2:** Implement + cập nhật docs (nhắc alpha `rKlkG9O8` Sharpe 1.57/self-corr 0.49
  đang nằm sẵn trong DB — user có thể nộp ngay không cần chạy thêm phiên).
- [ ] **Step 3:** pytest xanh → commit.

**DoD:** Kết thúc mỗi phiên user thấy ngay có gì nộp được và lệnh nộp; hết nhầm lẫn Power Pool.

---

## Ngoài phạm vi (có chủ đích)

- **Vá fidelity panel local (ρ=0.308 → ≥0.5)** — FIX_PLAN 07-11 T6: hoãn. Floor đã tự-tắt khi ρ
  không tin (`a404874`/`ecee333`); hướng của plan này là GIẢM phụ thuộc local (sim-thẳng +
  multi-sim + combiner dùng PnL local chỉ để đo tương quan, không đo mức Sharpe). Mở lại khi
  các đòn bẩy trên đã cạn.
- **Nộp Power Pool**: đã chốt ở `e27821d` — eligible chỉ là structural, không có đường nộp riêng.

## Thứ tự thực thi & nghiệm thu tổng

Thứ tự: **T1 → T2** (đòn bẩy lớn nhất — mở đường √N) → **T3 + T4** (kỷ luật quota, rẻ)
→ **T5** (cứu hypothesis alt-data) → **T6** (throughput) → **T7** (nguồn mới) → **T8** (đường nộp).
T3/T4/T8 độc lập, có thể giao subagent song song với T1/T2 (khác file chính). T5 và T6 đụng
cùng vùng `_sim_direct` — làm tuần tự (T5 trước, T6 refactor sang multi-sim sau).

**Nghiệm thu tổng (USER chạy menu-5 một phiên sau khi merge):**
1. Combiner sinh ≥1 combo được sim Brain (hoặc drop_stats giải thích được vì sao chưa).
2. GP ≤ 3 sim; ≥60% sim Brain thuộc seed/hypothesis/combiner.
3. 0 sim đốt vào biểu thức degenerate/hằng số.
4. Alt-data: mỗi hypothesis |Sharpe|≥0.5 được ≥2 config thử (sweep), không còn 1-shot-rồi-vứt.
5. Mục tiêu chính: **≥1 alpha `failed_checks == []` mới** trong phiên; nếu chưa đạt, đọc
   `session_summary` + CSV để lặp vòng chẩn đoán tiếp theo với dữ liệu mới.
