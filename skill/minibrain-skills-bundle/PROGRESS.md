# MiniBrain — Progress log

> Append-only development journal. Maintained via the `session-journal` skill.
> Read the `Current state` block + the last few entries at the start of every session;
> append an entry and refresh `Current state` at the end of every session or phase.

## Current state
- **Phase:** Phase 4 — Metrics + Gates ✅ HOÀN TẤT (merged main + pushed `73c6129..d74d471`). Tiếp theo: Phase 4.5 — Calibration.
- **Quyết định hướng đi:** Tích hợp MiniBrain vào tool sẵn có (KHÔNG build grenfield). Code mới
  đặt trong `src/` (không phải `minibrain/`), tái dùng login/fetch/DB/sim/AI/submit. Mỗi phase =
  1 nhánh git → merge main → push. **Bỏ đường cũ** (LLM→sim trực tiếp): mọi candidate qua local
  gate trước khi đốt sim (ĐÃ GỠ ở Phase 3 — D9). Spec: `docs/superpowers/specs/2026-06-23-minibrain-into-existing-tool-design.md`.
  Plans: `docs/superpowers/plans/2026-06-24-minibrain-integration-master-plan.md` (P0-P9) +
  `2026-06-24-phase-3-backtester.md` + `2026-06-24-phase-4-metrics-gates.md`.
- **Done (Phase 3):** `src/backtest/config.py` (`PortfolioConfig` + `Neutralization`, stage separation),
  `src/backtest/portfolio.py` (`PortfolioBuilder.build`: decay→neutralize→truncate→scale→delay; `_truncate`
  dùng **water-filling lặp** — sửa correctness so plan vì 1-pass không cap đúng sau scale), `src/backtest/
  backtester.py` (`Backtester.run` delay-1 `pnl=nansum(w*ret)`, delay áp ở portfolio không nhân đôi),
  `src/backtest/gate.py` (`score_local_gate` cổng local tối thiểu: parse+eval+pnl hữu hạn). **D9:**
  `RefinementLoop._evaluate` chèn `score_local_gate` BẮT BUỘC trước `simulate` khi `market_data` wire;
  `market_data=None` → gate bỏ qua (bảo toàn hành vi cũ, 54 test loop xanh). **MVP demo thật:**
  `equity_curve[-1]=-0.0937, sharpe~-3.150` (random walk fixture, chỉ chứng minh pipeline thông).
  **658 pass / 1 fail pre-existing**; ruff + mypy --strict (src/backtest) clean. Final review opus:
  READY TO MERGE, 0 Critical/Important, 2 Minor (phase sau).
- **Done (Phase 4):** `src/backtest/metrics_local.py` (`AlphaMetrics` frozen/slots + `MetricsCalculator`:
  sharpe/annual_return[simple, KHÔNG CAGR]/turnover/max_drawdown/fitness[floor từ config]/per_year_sharpe
  [qua `data.years()`]/weight_concentration[ngày tệ nhất]). `src/backtest/gates.py` (`GateVerdict` +
  `GateEvaluator`: hard gates depth/fields/self_corr[strict `<`]/concentration tách bạch soft scores
  sharpe/fitness/turnover_band/per_year_min; ngưỡng CHỈ từ `config/thresholds.py`). `src/scoring/filter.py`
  thêm `evaluate_local` (wrap `GateEvaluator`, giữ nguyên `passes`/`blocking_dimensions` sim-Brain).
  `src/backtest/gate.py` rewire `score_local_gate(..., self_corr=0.0)` gọi `MetricsCalculator`+`GateEvaluator`
  thật (tương thích ngược 3 call site RefinementLoop). Integration `tests/integration/test_metrics_gates.py`
  thật trên `small_panel`: **demo `sharpe=-0.606 fitness=-0.322 turnover=0.107 concentration=0.099
  gate_passed=True`**. Full suite **689 pass / 1 fail pre-existing** (psycopg). ruff + mypy --strict (src/backtest)
  clean. Final review opus: **READY TO MERGE = YES, 0 Critical/Important.**
- **In progress:** —
- **Next step:** Phase 4.5 — Calibration: `CalibrationHarness`/`CalibrationReport` + loader `brain_record`
  ground truth; tính `spearman_sharpe` trên ≥~50 alpha đã sim; gate tin cậy ranking theo `CALIBRATION_RHO_BAR`.
  **PHỤ THUỘC** giải Gap#3 (bulk OHLCV) để có data sim thật.
- **Blockers / open risks:** (R2/Gap#3) Gap bulk OHLCV chưa giải (calibration ρ phụ thuộc).
  **Minor đã ghi (dọn sau, không chặn merge):** (1) `RuntimeWarning: Mean of empty slice` từ
  `src/operators_local/timeseries.py` `ts_mean` (lookback đầu panel) — PRE-EXISTING, ngoài phạm vi P4; bọc
  errstate/guard win rỗng ở tầng operator. (2) `src/scoring/filter.py` 2 lỗi mypy --strict PRE-EXISTING
  (`passes`/`blocking_dimensions` untyped `source`) — P4 thêm 0 lỗi mới. (3) gate đếm depth cây TRẦN (không
  cộng wrapper config) — nhất quán `pre_filter`, comment `MAX_DEPTH` đã sửa cho khớp. (4) `self_corr` luôn
  0.0 đến khi Phase 6 wire `PoolCorrelation` (đã document). **Phase 2 Minor còn mở:** `inf` từ divide/0,
  log/0 chưa mask ở Evaluator; fidelity WQ `trade_when`/`hump`. Legacy `client.py` 9 lỗi mypy +
  `test_db_postgres` 1 fail pre-existing (psycopg).
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
