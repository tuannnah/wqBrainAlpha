# MiniBrain — Progress log

> Append-only development journal. Maintained via the `session-journal` skill.
> Read the `Current state` block + the last few entries at the start of every session;
> append an entry and refresh `Current state` at the end of every session or phase.

## Current state
- **Phase:** Phase 5 — Database ✅ HOÀN TẤT (merged main 98fca96, pushed). Tiếp theo: **Phase 6 — Pool correlation** (local self-corr gate, dùng `load_pool`/`save_pool_pnl` của MiniBrainRepository).
- **Done (Phase 5, 2026-06-25, subagent-driven + opus final review):** 5 model mới trong `src/storage/models.py`
  (`ExpressionModel`/`EvaluationModel`/`PoolPnlModel`/`DeadFieldModel`/`BrainRecordModel`, B11 schema:
  UNIQUE(expression_id,config_json,data_window) + idx_eval_sharpe/expr/status); `MIGRATION_ORDER` mở rộng
  (port Postgres, đúng thứ tự FK); `MiniBrainRepository` (`src/storage/repository.py`: upsert_expression dedup
  theo canonical_hash, record_evaluation lưu CẢ pass+fail+seed[R8], save/load_pool_pnl blob float64/datetime64[D],
  dead_field self-learning, result_cache_get CHỈ hit status=passed, top_n); `ResultCache` (`src/cache/result_cache.py`
  B12 tier3 DB-backed). **Branch thuần additive (844 insert, 0 delete)** — luồng Brain-sim cũ KHÔNG đụng,
  init_db idempotent + AlphaModel cũ còn nguyên (verified). Final review opus: With fixes → ĐÃ fix `load_pool` `.copy()`
  (np.frombuffer trả read-only → crash in-place Phase 6 max_corr). Full suite **753 pass / 1 psycopg tiền-tồn**.
- **Phase 6 dùng được ngay:** `MiniBrainRepository.load_pool()` trả `{evaluation_id: pnl_array}` (đã ghi-được);
  `save_pool_pnl(eval_id, dates, pnl)`. Minor defer: config_json/data_window là opaque key chưa canonical
  (Phase 7/8 wire cache nên `json.dumps(..., sort_keys=True)`); dates_blob lưu nhưng load_pool chưa trả (by-design,
  Phase 6 chỉ cần pnl-by-id). mypy debt trên src/storage là baseline `declarative_base()` legacy (cũ lẫn mới cùng pattern).
- **🎯 NORTH STAR ĐẠT + CỦNG CỐ (2026-06-25): ρ_sharpe=0.823, ρ_fitness=0.922, n=55.** Tiến trình:
  (1) Ban đầu panel S&P500 2015-2025: ρ=0.671 n=42. (2) **Điều tra 13 alpha drop** → root cause
  `EVAL KeyError: 'returns'` (returns là field WQ hợp lệ nhưng MarketData lưu riêng `.returns`, không trong
  `.fields`); **fix** `make_local_scorer` expose returns (commit 5467c3b) → ρ=0.771 n=55. (3) **Mở rộng**
  lịch sử 2015→2010 (467 mã × 3876 ngày) → **ρ=0.823**. RANKING LOCAL ĐÁNG TIN dù data xấp xỉ (S&P500≠TOP3000,
  vwap≈typical price). Lệnh: `main.py calibrate --db-url sqlite:///wq_alpha_phtrang1229_gmail_com.db
  --market-data-dir data/market_yf2010`. Panel: `scripts/fetch_yfinance_panel.py` (default start 2010).
  Brain API xác nhận KHÔNG có bulk OHLCV (Gap#3) → yfinance fallback.
- **Follow-up:** (a) ✅ ĐÃ SỬA TẬN GỐC lỗ hổng `returns`-không-là-field: `MarketData.field()` resolve
  `returns`→`.returns` (field phái sinh) + `MarketData.field_names()` (gồm returns) dùng cho `fields_ok`
  ở `score_local_gate`. Fix MỘT chỗ (Phase 0 core) → hết lỗi ở Evaluator + RefinementLoop gate + calibration;
  gỡ patch tạm trong make_local_scorer. Full suite 723 pass; ρ vẫn 0.823 (central fix ≡ patch cũ). (b) Mở rộng
  universe >S&P500 cần nguồn ticker-kèm-GICS-sector (datahub chỉ có S&P500; 11 alpha group_neutralize cần sector) — CHƯA làm.
- **GROUND-TRUTH XONG:** 55 sim non-null sharpe trong `wq_alpha_phtrang1229_gmail_com.db` (min=-1.62 median=0.62 max=1.27; 12 âm/43 dương). `load_brain_records` đọc đủ 55. Login THẬT qua `WQBrainClient`+`.env`+cookie `.wq_session` (KHÔNG wqb-mcp — mcp trả 400/403). Scripts `gen/persist/run_groundtruth.py` đã commit (retry-timeout + resume).
- **CHẶN ĐO ρ THẬT = Gap#3 (nguồn OHLCV panel):** ĐÃ PROBE Brain API (2026-06-25), KHẲNG ĐỊNH **không có endpoint trả giá trị field bulk**: `/data-fields/close`→200 CHỈ metadata; `/data-fields/close/values` & `/data` →404; `/data-sets`→metadata. Field data chỉ truy cập được TRONG simulation (server-side). => Pull OHLCV qua Brain API KHÔNG khả thi (đúng `fetch_to_parquet` NotImplementedError). Cần fallback: yfinance/stooq (miễn phí, xấp xỉ) HOẶC parquet user cung cấp. Sau khi có panel: `python main.py calibrate --db-url sqlite:///wq_alpha_phtrang1229_gmail_com.db --market-data-dir <parquet>`.
- **Quyết định hướng đi:** Tích hợp MiniBrain vào tool sẵn có (KHÔNG build grenfield). Code mới
  đặt trong `src/` (không phải `minibrain/`), tái dùng login/fetch/DB/sim/AI/submit. Mỗi phase =
  1 nhánh git → merge main → push. **Bỏ đường cũ** (LLM→sim trực tiếp): mọi candidate qua local
  gate trước khi đốt sim (ĐÃ GỠ ở Phase 3 — D9). Spec: `docs/superpowers/specs/2026-06-23-minibrain-into-existing-tool-design.md`.
  Plans: `docs/superpowers/plans/2026-06-24-minibrain-integration-master-plan.md` (P0-P9) +
  `2026-06-24-phase-3-backtester.md` + `2026-06-24-phase-4-metrics-gates.md`.
- **Done Phase 0-4.5** (chi tiết trong Entries Session 02-07): P0 data foundation, P1 parser, P2 evaluator+27 op,
  P3 backtester (MVP), P4 metrics+gates, P4.5 calibration (ρ_sharpe=0.823 đo thật). Tất cả merged main.
- **In progress:** —
- **Next step:** **Phase 6 — Pool correlation** (plan `docs/superpowers/plans/2026-06-24-phase-6-pool-corr.md`):
  `src/backtest/pool_corr.py`; gate tiêu thụ `max_corr`; passing alpha → `save_pool_pnl`; candidate mới tính
  `max|ρ|` vs pool trên ngày aligned, hard gate 0.70. Phase 5 đã cấp sẵn `load_pool`/`save_pool_pnl`. Đây cũng
  là chỗ wire `self_corr` (P4 để dormant=0.0) thành số thật.
- **Blockers / open risks:** (R2/Gap#3) bulk OHLCV chưa giải (chỉ ảnh hưởng calibration mở rộng universe, KHÔNG
  chặn P6). **Minor mở (dọn sau, không chặn):** (P5) config_json/data_window opaque key chưa canonical (Phase 7/8
  cache nên sort_keys=True); mypy debt baseline `declarative_base()` legacy trên src/storage. (P4) `RuntimeWarning:
  Mean of empty slice` `ts_mean`; `filter.py` 2 lỗi mypy pre-existing. (P2) `inf` divide/0, log/0 chưa mask ở
  Evaluator; fidelity WQ `trade_when`/`hump`. Legacy `client.py` 9 lỗi mypy + `test_db_postgres` 1 fail (psycopg).
- **MVP (Phases 1–3) reached:** ✅ YES (Phase 3 xong — parse→eval→build→backtest→equity chạy thông trên dữ liệu thật)
- **Calibration ρ (Spearman, Sharpe):** not measured yet (Phase 4.5)

## Entries
<!-- append-only; newest at the bottom -->

### [2026-06-23] Session 01 — Design spec + skill bundle
- **Phase:** Pre-Phase 0 (planning).
- **Done:** Analyzed the original `goal.md` / `skeleton.md`. Produced the master design spec
  `docs/MINIBRAIN_DESIGN.md` (analysis, full system design, MVP-first phase plan, rules,
  execution order). Authored two Claude Code skills: `minibrain-builder` (architecture +
  phases + conventions + per-phase ritual) and `session-journal` (this log).
- **Decisions:** (1) Added two first-class concerns the original skeleton lacked: a
  **CalibrationHarness** (local vs Brain rank-correlation — the validity check for the whole
  tool, Phase 4.5) and **pool PnL self-correlation** as a hard gate (the real submission
  blocker, not AST hashing). (2) Made `MarketDataSource` a pluggable port because Brain does
  not supply historical data. (3) Simplified the MVP stack to numpy/pandas + joblib + lark +
  sqlite; deferred numba/ray/deap until profiling/scale justifies them. (4) Made the GP
  fitness correlation- and regime-aware (NSGA-II + pool/population correlation penalties) to
  avoid breeding a saturated population. (5) Enforced stage separation: GP searches bare
  cores; neutralization/decay/truncation live in `PortfolioConfig`.
- **In progress:** —
- **Blockers / open risks:** Market-data source not yet identified/wired (see Current state).
- **Next step:** Phase 0 — repo scaffold + data foundation (see Current state).
- **Tests:** None yet (no code).

### [2026-06-24] Session 02 — Phase 0 Data foundation (tích hợp vào tool sẵn có)
- **Phase:** Phase 0 — Data foundation. HOÀN TẤT, merged main.
- **Done:** Brainstorm lại hướng đi (tích hợp thay vì grenfield) → spec + master plan P0-P9 +
  plan Phase 0 chi tiết. Thực thi Phase 0 bằng subagent-driven (11 task TDD): thresholds,
  settings, type aliases, MarketData panel, MarketDataSource port, universe mask, ParquetSource
  adapter (round-trip), market_fetch spike, fixture small_panel. 18 unit test xanh, ruff +
  mypy --strict (module mới) clean. Final review (opus) = Ready to merge YES, 0 Critical/Important.
- **Decisions:** (1) Code mới vào `src/` tái dùng hạ tầng, không tạo package `minibrain/`.
  (2) Mỗi phase 1 nhánh git → merge → push. (3) Bỏ đường cũ, mọi candidate qua local gate (gỡ ở P3).
  (4) Market data kéo từ WQ Brain API (PIT-faithful) — nhưng endpoint bulk chưa rõ (Gap#3), tách
  `_assemble_panel` (thuần, test được) khỏi `fetch_to_parquet` (raise NotImplementedError có chỉ dẫn).
  (5) MarketData validate dates tăng nghiêm ngặt + dtype (review hardening). (6) Cài ruff/mypy/
  pandas-stubs vào venv + requirements.
- **Blockers / open risks:** Gap#3 bulk OHLCV chưa giải (calibration ρ phụ thuộc); legacy client.py
  9 lỗi mypy pre-existing (ngoài phạm vi).
- **Next step:** Phase 1 — Parser: viết plan chi tiết rồi thực thi `src/lang/*`; cuối phase xóa ast_utils.py.
- **Tests:** Xanh. 18 unit test (+ suite cũ không vỡ; 1 fail pre-existing test_db_postgres do thiếu psycopg).

### [2026-06-24] Session 03 — Phase 1 Parser (tầng ngôn ngữ FASTEXPR-subset)
- **Phase:** Phase 1 — Parser. HOÀN TẤT, sẵn sàng merge main.
- **Done:** Thực thi 11 task TDD (subagent-driven): AST sealed hierarchy `Constant/Field/Call` +
  `NodeVisitor` Protocol; `OperatorRegistry` (ArgKind/OpCategory/OperatorSpec/`@register`/
  `default_registry` 6 op placeholder); grammar Lark; parser `parse()`/`parse_expression()`; 5 visitor
  (Depth đếm wrapper, FieldCollector dedup, Serializer round-trip, CanonicalHasher sort-commutative,
  ComplexityVisitor node-count) + hàm thuần `all_subtrees`/`iter_leaves`. Migrate 9 caller
  (complexity, zoo, similarity, novel_ideas, local_select, generator, expr_synth, pre_filter, simulator)
  khỏi `ast_utils` rồi XÓA `src/generation/ast_utils.py` + `tests/test_ast_utils.py`. Verify: 87 unit
  test lang xanh, full suite 590 pass (1 fail pre-existing psycopg), ruff + mypy --strict (src/lang) clean.
- **Decisions:** (1) Registry Phase 1 là **skeleton khai báo** — impl 6 op raise NotImplementedError
  ("Phase 2 sẽ impl"); đủ để parser validate operator/arity, không phải nợ kỹ thuật ẩn. (2) **Deviation
  cần thiết để migration không vỡ:** thêm `parse_expression()` LENIENT (chỉ check cú pháp, bỏ validate
  registry) song song `parse()` STRICT — vì 9 caller legacy dùng operator chưa đăng ký (ts_zscore,
  ts_delta, group_neutralize...); thêm **unary minus** vào grammar (`-N`→`Constant(-N)`, `-expr`→
  `multiply(-1, expr)`) vì codebase legacy có `-rank(x)`/`multiply(-1,x)`. (3) `to_expression` cũ render
  infix `(a + b)` → `Serializer` mới render dạng hàm `add(a, b)` — SEMANTIC tương đương (WQ chấp nhận cả
  hai), test autowrap không lộ khác biệt. (4) `NodeVisitor` dùng TypeVar **covariant** (T_co) cho mypy
  --strict (T chỉ ở return position).
- **In progress:** —
- **Blockers / open risks:** Validate "field tồn tại" hoãn sang Phase 2 (cần MarketData thật).
  `parse_expression` lenient là cửa hậu tạm cho legacy — Phase 2+ nên dần chuyển caller sang `parse`
  strict khi registry đủ operator. Lưu ý git: nhánh `phase-1-parser` có thêm 1 commit docs (3dee855)
  trùng nội dung với commit docs trên main (bf36d78) — merge sẽ auto-resolve (cùng nội dung).
- **Next step:** Phase 2 — Operator Engine (plan `2026-06-24-phase-2-operator-engine.md` đã có sẵn).
- **Tests:** Xanh. 87 unit test lang (8 file) + full suite 590 pass; 1 fail pre-existing test_db_postgres (psycopg).

### [2026-06-24] Session 04 — Phase 2 Operator Engine (Evaluator + 27 operator)
- **Phase:** Phase 2 — Operator Engine. HOÀN TẤT, sẵn sàng merge main.
- **Done:** Thực thi 10 task TDD (subagent-driven-development): 3 implementer subagent (2.1+2.2 engine
  core; 2.3-2.8 sáu file operator; 2.9 wire+integration) + 1 final review subagent (opus). `SubexprCache`
  LRU; `EvalContext`+`Evaluator(NodeVisitor[Panel])` dispatch qua registry + cache canonical-hash + áp
  universe mask sau mỗi Call; 27 operator impl thật (arithmetic 10, cross_sectional 4, timeseries 9,
  group 1, neutralization 2, conditional 2). Golden test mỗi nhóm + integration parse→eval. Verify:
  632 pass / 1 fail pre-existing; ruff + mypy --strict clean (src/engine + src/operators_local). Final
  review opus: READY TO MERGE, 0 Critical/Important, 4 Minor (ghi chú phase sau).
- **Decisions:** (1) Window time-series trailing `[t-d+1, t]` (đủ d quan sát kể cả t); thiếu → NaN —
  chốt tường minh vì spec gốc không nói rõ biên. (2) `scale` dùng `OpCategory.SCALING` + `gp_usable=False`
  (wrapper rescale gross-exposure, rank/sign-preserving) thay vì CROSS_SECTIONAL. (3) GROUP arg biểu diễn
  bằng `Field(name)` trong AST (không thêm node type), `_literal()` đọc `Field.name` làm string. (4) Chạy
  6 file operator TUẦN TỰ trong 1 subagent (không song song) để tránh git-index race trên cùng working
  tree. (5) Golden test so với raw field phải áp universe mask lên `expected` (phản ánh invariant B6
  Evaluator NaN-hóa out-of-universe) — không nới lỏng assertion. (6) Sửa `_apply_universe_mask` nhận
  `Mask` (bool) thay `Panel` cho mypy. (7) Cập nhật test Phase 1 `test_default_registry_has_minimal_phase1_ops`
  bỏ assert placeholder NotImplementedError (Phase 2 ghi đè impl thật vào REGISTRY singleton).
- **In progress:** —
- **Blockers / open risks:** (Minor review, phase sau) (a) `inf` từ divide/0, log/0 KHÔNG bị mask ở
  Evaluator — cần làm sạch ở portfolio Phase 3+. (b) Fidelity WQ `trade_when`/`hump` khớp spec plan
  nhưng có thể lệch WQ thật — rủi ro calibration Phase 4.5. Gap#3 bulk OHLCV vẫn mở.
- **Next step:** Phase 3 — Backtester (MVP): PortfolioBuilder + Backtester delay-1 + equity curve.
- **Tests:** Xanh. 632 pass / 1 fail pre-existing (psycopg). 29 golden + 4 integration + 9 engine unit mới.

### [2026-06-24] Session 05 — Phase 3 Backtester (MVP MILESTONE) + gỡ đường cũ D9
- **Phase:** Phase 3 — Backtester (MVP). HOÀN TẤT, sẵn sàng merge main. **MVP milestone của toàn dự án đạt.**
- **Done:** Thực thi 6 task (subagent-driven): backtest core (config/portfolio/backtester) + integration
  MVP + gate D9 + review. `PortfolioConfig`+`Neutralization`; `PortfolioBuilder.build`
  (decay→neutralize→truncate→scale→delay, chỉ in-universe); `Backtester.run` delay-1 (delay ở portfolio,
  không nhân đôi); `score_local_gate` (parse+eval+pnl). D9: chèn gate BẮT BUỘC vào `RefinementLoop._evaluate`
  trước `simulate`. MVP demo thật trên small_panel: `equity_curve[-1]=-0.0937 sharpe~-3.150`. Verify:
  658 pass / 1 fail pre-existing; loop 54/54; ruff+mypy --strict (src/backtest) clean. Final review opus:
  READY TO MERGE, 0 Critical/Important.
- **Decisions:** (1) **`_truncate` đổi từ 1-pass (plan) sang water-filling LẶP** — vì 1-pass + renorm
  KHÔNG đảm bảo `|w_i|<=cap*gross` sau scale (phản ví dụ rõ); water-filling ghim cap fraction, bảo toàn
  qua `_scale`. User ĐÃ PHÊ DUYỆT sửa correctness này. (2) Sửa test truncate cap 0.10→0.40 vì cap 0.10 với
  4 mã bất khả thi toán học (4×0.10<1.0). (3) `EvalContext(registry=...)` KHÔNG optional → gate/test dùng
  `default_registry()` + `import src.operators_local` (side-effect đăng ký op) — sửa so chữ ký sai trong plan.
  (4) D9: `market_data=None` → gate bỏ qua (bảo toàn hành vi cũ); có data → gate bắt buộc. (5) Alpha MVP
  dùng return-1-ngày thay `open` (fixture không có open).
- **In progress:** —
- **Blockers / open risks:** 2 Minor review (Phase 4 dọn): RuntimeWarning empty-slice; local_gate_fn bind.
  Phase 2 Minor còn mở: inf chưa mask; fidelity WQ trade_when/hump. Gap#3 bulk OHLCV.
- **Next step:** Phase 4 — Metrics + Gates (MetricsCalculator + GateEvaluator + config/thresholds.py).
- **Tests:** Xanh. 658 pass / 1 fail pre-existing (psycopg). Mới: 5 unit backtest + 1 integration MVP +
  gate + loop_local_gate.

### [2026-06-24] Session 06 — Phase 4 Metrics + Gates (subagent-driven, merged main)
- **Phase:** Phase 4 — Metrics + Gates. HOÀN TẤT, merged main + pushed `73c6129..d74d471`.
- **Done:** Thực thi plan `2026-06-24-phase-4-metrics-gates.md` bằng **subagent-driven-development** (theo yêu
  cầu "superpower + subagent"): 6 task, mỗi task 1 implementer (sonnet — KHÔNG haiku vì file tiếng Việt có dấu)
  + task-reviewer + fix-loop khi cần. 4.1 `AlphaMetrics`+`MetricsCalculator`; 4.2 test per_year/concentration;
  4.3 `GateVerdict`+`GateEvaluator`; 4.4 `evaluate_local`; 4.5 integration end-to-end; 4.6 wire
  `score_local_gate`+`self_corr`. Final whole-branch review (opus) = **READY TO MERGE YES, 0 Critical/Important**.
  Demo thật `small_panel`: `sharpe=-0.606 fitness=-0.322 turnover=0.107 concentration=0.099 gate_passed=True`.
  Full suite 689 pass / 1 fail pre-existing (psycopg). ruff + mypy --strict (src/backtest) clean.
- **Decisions:** (1) Tách Task 4.6: implementer chỉ làm Steps 1–9 (wire+test+commit); **final review + merge +
  push do controller** (đúng quy trình SDD, push main là hành động ra ngoài). (2) **Fix correctness ngoài
  brief có chủ đích, được review chấp nhận:** (a) `_turnover` xử lý NaN MỘT phía (mã vào/ra universe) — brief
  cũ chỉ mask both-NaN nên tính thiếu turnover; sửa dùng `nan_to_num` coi NaN = vị thế 0 (WQ-faithful). (b)
  `_weight_concentration` guard hàng all-NaN để hết `RuntimeWarning: All-NaN slice` (pristine output). (c)
  `GateEvaluator` thêm dung sai FP `+1e-9` ở so sánh concentration vì `truncation` mặc định 0.10 == `CAP` 0.10
  + trôi số sau `_scale` → 0.10000000000000005 false-fail; epsilon SỐ HỌC (không phải ngưỡng gate), docstring
  `gates.py` đã phân biệt rõ + test ghim biên. (3) Sửa comment `MAX_DEPTH` (`config/thresholds.py`) cho khớp:
  gate đếm cây TRẦN (nhất quán `pre_filter`), bỏ mệnh đề "gồm wrapper config".
- **In progress:** —
- **Blockers / open risks:** Gap#3 bulk OHLCV vẫn chặn calibration ρ (Phase 4.5). Minor deferred: warning
  `ts_mean` empty-slice (tầng operator); 2 lỗi mypy pre-existing `filter.py` legacy; `self_corr` dormant tới
  Phase 6. Xem `Current state` chi tiết.
- **Next step:** Phase 4.5 — Calibration (CalibrationHarness + spearman_sharpe), phụ thuộc giải Gap#3.
- **Tests:** Xanh. 689 pass / 1 fail pre-existing (psycopg). Mới: test_metrics_local (12) + test_gates (14) +
  test_filter_evaluate_local (3) + integration test_metrics_gates (1) + 2 test mở rộng test_backtest_gate.

### [2026-06-25] Session 07 — Phase 4.5 Calibration (subagent + controller, merged main)
- **Phase:** Phase 4.5 — Calibration. CODE HOÀN TẤT, merged main + pushed `1d21396..4dab94b`.
- **Done:** Thực thi plan `2026-06-24-phase-4.5-calibration.md` (subagent-driven, chuyển sang controller
  tự code từ Task 4.5.4 khi subagent gặp API 402 quota). 4.5.1 `stats.spearman` thuần numpy (no scipy);
  4.5.2 `loader.load_brain_records` (DB AlphaModel⋈SimulationModel⋈SubmissionModel, latest/alpha, lọc
  error/null); 4.5.3 `CalibrationReport`; 4.5.4 `CalibrationHarness` + `make_local_scorer` (config khớp
  ground-truth NONE/decay0/trunc0/delay1 — điều kiện ρ hợp lệ); 4.5.5 CLI `calibrate` (ParquetSource +
  `--market-data-dir` bắt buộc, không in báo cáo giả, verdict vs `CALIBRATION_RHO_BAR`). Final review opus:
  READY=YES, 0 Critical. Full suite 719 pass / 1 pre-existing (psycopg).
- **Decisions:** (1) **Login Brain THẬT** qua `WQBrainClient`+`.env`(phtrang1229)+cookie `.wq_session`, KHÔNG
  qua wqb-mcp (mcp `authenticate` chỉ trả token đọc, `create_simulation` vẫn 400/403). (2) make_local_scorer
  dùng config KHỚP ground-truth (không phải PortfolioConfig mặc định SECTOR/0.10) — bắt được bug `or` vs
  `is not None` cho brain_sharpe=0.0 trong code mẫu brief. (3) CLI lazy-import calibration để tránh E402 (0
  lỗi lint mới; main.py legacy giữ 13 ruff/90 mypy). (4) Fix review: `init_db` cho DB mới (hết
  OperationalError); cảnh báo precondition DB-cùng-config (Important final review) — follow-up: cột
  config_key Phase 5 để lọc.
- **Blockers / open risks:** **ρ DỮ LIỆU THẬT CHƯA ĐO** — ground-truth 50 sim BLOCKED (Brain account
  phtrang1229 API 402 hết quota sau vài sim). Scripts + 60 expr OHLCV-only đã sẵn, resume khi quota mở. Minor
  defer: tiebreak sim_at, n-vs-pair-count, warning ts_mean empty-slice (operator layer).
- **Next step:** Phase 5 — Database (store/repository/cache) HOẶC khi quota Brain mở: chạy ground-truth →
  `calibrate --market-data-dir <parquet>` → đo ρ thật → quyết định có tin ranking local không.
- **Tests:** Xanh. 719 pass / 1 fail pre-existing (psycopg). Mới: test_calibration_stats (9) + _loader (5) +
  _report (2) + _harness (6) + integration (4) + test_calibrate_command (4).

### [2026-06-25] Session 08 — Phase 5 Database (subagent-driven + opus final review, merged main)
- **Phase:** Phase 5 — Database. HOÀN TẤT, merged main + pushed `01b999a..98fca96`.
- **Done:** Thực thi plan `2026-06-24-phase-5-database.md` bằng **subagent-driven-development** (theo yêu cầu user
  "subagent + superpower"): 6 task TDD, mỗi task 1 implementer (sonnet — file tiếng Việt có dấu, KHÔNG haiku) +
  task-reviewer (sonnet) + fix-loop. 5.1 `models.py` +5 model (Expression/Evaluation/PoolPnl/DeadField/BrainRecord,
  B11: UNIQUE(expr_id,config_json,data_window)+idx_eval_sharpe/expr/status); 5.2 `migrate.py` MIGRATION_ORDER đúng
  thứ tự FK (port Postgres); 5.3 `MiniBrainRepository` (upsert dedup canonical_hash, record_evaluation pass+fail+seed,
  save/load_pool_pnl blob, dead_field, result_cache_get chỉ status=passed, top_n) — reviewer probe sâu merge/blob/cache
  đều đúng; 5.4 `ResultCache` (src/cache/, B12 tier3); 5.5 integration parse→visitors thật→repo→cache (`add`
  commutative xác minh trong registry thật). Final whole-branch review (opus) = **With fixes**: 1 Important
  `load_pool` trả mảng read-only (np.frombuffer) → crash in-place Phase 6 → ĐÃ fix `.copy()` + test writeable (e538cb6).
  Branch **thuần additive 844 insert / 0 delete** — luồng Brain-sim cũ KHÔNG đụng, init_db idempotent + AlphaModel cũ
  còn nguyên (verified). Full suite 753 pass / 1 psycopg tiền-tồn.
- **Decisions:** (1) implementer dùng sonnet KHÔNG haiku (file có tiếng Việt có dấu — haiku xóa dấu). (2) Plan
  chứa code đầy đủ nên implementer là transcription+TDD; final review trên opus (model mạnh nhất theo SDD). (3)
  mypy --strict trên src/storage giữ baseline `declarative_base()` legacy (model cũ lẫn mới cùng pattern 2 lỗi/class)
  — KHÔNG migrate sang DeclarativeBase/Mapped trong P5 (ngoài phạm vi); src/cache mypy SẠCH. (4) `DeadFieldModel`
  (`dead_fields_minibrain`) tách khỏi `InvalidFieldModel` cũ (PK-shape khác, hai luồng độc lập). (5) `MiniBrainRepository`
  tách `AlphaRepository` — không sửa class cũ.
- **In progress:** —
- **Blockers / open risks:** Minor defer (không chặn P6): config_json/data_window opaque key chưa canonical (Phase 7/8
  cache nên `json.dumps(sort_keys=True)`); dates_blob lưu nhưng load_pool chưa trả (by-design, P6 chỉ cần pnl-by-id);
  mypy baseline src/storage. Gap#3 bulk OHLCV không ảnh hưởng P6.
- **Next step:** Phase 6 — Pool correlation: `src/backtest/pool_corr.py` + gate `max_corr` 0.70, wire `save_pool_pnl`
  cho alpha pass + `self_corr` thật (P4 dormant). Dùng `load_pool`/`save_pool_pnl` của MiniBrainRepository (Phase 5).
- **Tests:** Xanh. 753 pass / 1 psycopg tiền-tồn. Mới: test_storage_models_minibrain (7) + test_migrate_minibrain (2)
  + test_minibrain_repository (14, gồm fix writeable) + test_result_cache (4) + integration (3).
