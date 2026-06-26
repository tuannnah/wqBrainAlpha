# MiniBrain — Progress log

> Append-only development journal. Maintained via the `session-journal` skill.
> Read the `Current state` block + the last few entries at the start of every session;
> append an entry and refresh `Current state` at the end of every session or phase.

## Current state
- **Phase:** Phase 7 — GP Engine ✅ HOÀN TẤT 100% (building blocks 7.1-7.6 + integration 7.7-7.9, merged main
  6d9ae8d, pushed). **Tiếp theo: Phase 8 — Short-list + CLI** (`docs/superpowers/plans/2026-06-24-phase-8-cli.md`).
- **Done (Phase 7 building blocks, 2026-06-26, subagent-driven + opus 2-round final review):** 6 task TDD building blocks GP:
  `src/gp/individual.py` (Individual = AST Node + metadata: generation/fitness cache, slots non-frozen vì test gán fitness sau init);
  `src/gp/fitness_vec.py` (FitnessVector 6 chiều `sharpe_deflated/per_year_min_sharpe/turnover_penalty/complexity_penalty/pool_corr_penalty/pop_corr_penalty` + `from_metrics` từ AlphaMetrics; siết Individual.fitness annotation qua TYPE_CHECKING);
  `src/gp/seeds.py` (sinh Node cores từ `families.py.generate_candidates()` + `novel_ideas.NOVEL_ALPHAS` + LLM tùy chọn qua Protocol; **side-effect import `src.operators_local`** bắt buộc để parse validate qua registry);
  `src/gp/init.py` (ramped half-and-half + seeding, depth cap MAX_DEPTH=7, rng inject `np.random.default_rng`);
  `src/gp/variation.py` (typed crossover + point_mutation type-aware WINDOW resample từ `parent_spec.window_choices` / SCALAR perturb Gaussian / GROUP bỏ qua + subtree_mutation / hoist_mutation + dedup canonical_hash; **helper chung `_panel_compatible_subtrees(root, registry)` cho mọi variation operator** đảm bảo typed invariant);
  `src/gp/selection.py` (NSGA-II Pareto front fast non-dominated sort + crowding distance + tie-break ngẫu nhiên qua rng inject; đảo dấu 2 chiều maximize qua `_MAXIMIZE_FIELDS`).
- **B5 stage separation enforced ở registry:** `regression_neut/vector_neut/ts_decay_linear/ts_delay` đặt `gp_usable=False` trong `src/operators_local/` — GP chỉ search BARE SIGNAL CORE, không sinh cây bọc neut/decay/delay (đó là PortfolioConfig Phase 3).
- **Final review opus 2-round:** Round 1 = With fixes (4 Critical typed-invariant + 2 Important — cùng nguyên nhân gốc thiếu helper PANEL-subtree chung, ~50% subtree_mut + 75% hoist + crossover sinh cây type-invalid trên seed kinh điển). Fix 7954468 root cause + 5 stress test 1000-iter `_check_panel_invariant` qua `spec.signature`. Round 2 = **Ready=YES, 0 Critical/Important**. Full suite **827 pass / 1 psycopg tiền-tồn**.
- **Phạm vi đã chốt với user:** chỉ 7.1-7.6 building blocks (plan dừng dở ở Selection — đầu plan tham chiếu 7.7-7.9 nhưng chưa viết). GPEngine (7.7: ghép seeds→init→variation→selection→eval, persist mọi individual, joblib parallel, sub-expr+result cache) + wire RefinementLoop (7.8) + xóa `src/generation/template.py` legacy: defer Phase 8 hoặc viết bổ sung plan 7.7-7.9 sau.
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
- **Done Phase 0-7** (chi tiết Entries Session 02-10): P0 data, P1 parser, P2 evaluator+27 op, P3 backtester (MVP),
  P4 metrics+gates, P4.5 calibration (ρ_sharpe=0.823), P5 database, P6 pool correlation, P7 GP building blocks
  (Individual/FitnessVector/Seeds/Init/Variation/Selection). Tất cả merged main.
- **Done Phase 7.7-7.9 (2026-06-26, inline executing-plans + Opus):** `src/gp/engine.py` (GPEngine: vòng lặp
  tiến hóa μ+λ end-to-end — `run()` ghép seeds→init→`_make_offspring`(crossover/point/subtree/hoist/copy)→eval
  qua Phase 2/3/4/6→`nsga2_select`; `_evaluate_individual` trả (fv,status,reasons,bt) với status passed/
  failed_gate/error; `_persist` upsert+record_evaluation mọi outcome+save_pool_pnl khi pass; `_config_json`
  sort_keys=True cho cache key canonical); `src/gp/seed_adapter.py` (GPSeedGenerator implement Protocol
  `idea_generator.generate_ideas(n)` cho RefinementLoop, dependency rule B1 không import src.llm); CLI
  `main.py generate --method=gp` thay TemplateGenerator (ParquetSource.load → GPEngine.run). Xóa
  `src/generation/template.py` + `tests/test_template.py`.
- **In progress:** —
- **Next step:** **Phase 8 — Short-list + CLI** (plan `docs/superpowers/plans/2026-06-24-phase-8-cli.md`).
- **Blockers / open risks:** (R2/Gap#3) bulk OHLCV chưa giải (chỉ ảnh hưởng calibration mở rộng universe). **Minor
  mở (dọn sau):** (P7.6) test coverage gap NSGA-II không assert `len(fronts)` cho chuỗi dominance, crowding không
  assert phần tử giữa hữu hạn; `_objective` gọi lặp trong crowding (vi mô O(MN²)). (P7.4) `ramped_half_and_half`
  ZeroDivisionError tiềm ẩn `max_depth<min_depth` (không thực tế MAX_DEPTH=7). (P6) `_pairwise_rho` dup-date lossy
  (benign); `evaluate_with_pool` bỏ `worst_id` (P7.7+ nên dùng). (P5) config_json/data_window opaque key chưa
  canonical (P7/8 cache nên sort_keys=True); mypy debt baseline `declarative_base()` legacy trên src/storage.
  (P4) `RuntimeWarning: Mean of empty slice` `ts_mean`; `filter.py` 2 lỗi mypy pre-existing. (P2) `inf` divide/0,
  log/0 chưa mask Evaluator; fidelity WQ `trade_when`/`hump`. Legacy `client.py` 9 lỗi mypy + `test_db_postgres`
  1 fail (psycopg).
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

### [2026-06-25] Session 09 — Phase 6 Pool correlation (subagent-driven + opus final review, merged main)
- **Phase:** Phase 6 — Pool correlation. HOÀN TẤT, merged main + pushed `707d12b..eb311aa`.
- **Done:** Thực thi plan `2026-06-24-phase-6-pool-corr.md` bằng **subagent-driven-development** (theo yêu cầu user
  "subagent + superpower"): 5 task (4.1 code + final review/merge), mỗi task 1 implementer (sonnet) + task-reviewer
  + fix-loop. **6.1** `src/backtest/pool_corr.py` `PoolCorrelation.max_corr` (max|Pearson ρ| align np.intersect1d
  trên dates giao nhau, bỏ qua alpha overlap<2/std=0 — KHÔNG ρ giả; trả (|ρ|, worst_id); KHÔNG import storage/gp/llm
  — dependency rule B1). **6.1-fix Critical** (review opus tìm + sonnet fix `fe977a5`): `_pairwise_rho` `np.argsort`
  cả 2 phía theo dates TRƯỚC `intersect1d`/`searchsorted` — trước đó dates chưa sort → ghép cặp sai âm thầm (ρ
  đáng lẽ 1.0 ra 0.913) hoặc IndexError; regression test `test_unsorted_dates_do_not_corrupt_alignment`. **6.2**
  `src/backtest/gates.py` `GateEvaluator.evaluate_with_pool` (lớp MỎNG: tính self_corr từ `pool_corr.max_corr` rồi
  delegate `evaluate()` cũ — chữ ký cũ byte-identical, ngưỡng 0.70 vẫn chỉ ở `evaluate()`/`SELF_CORR_MAX`). **6.3**
  (ADAPT) `MiniBrainRepository.load_pool` mở rộng trả `dict[int, tuple[(dates, pnl)]]` — giải Minor "dates_blob
  chưa dùng" Phase 5; CHỈ sửa load_pool, KHÔNG đụng save_pool_pnl/model/AlphaRepository; pnl giữ `.copy()` (Phase 5
  in-place fix). **6.4** integration `tests/integration/test_pool_corr_gate.py` end-to-end DB sqlite thật → upsert
  +record_evaluation (FK thật) → save_pool_pnl → load_pool → PoolCorrelation → `evaluate_with_pool`: pool rỗng pass,
  identical hard-fail self_corr, độc lập pass. Full suite **768 pass / 1 psycopg tiền-tồn**. Final review opus:
  **Ready=YES, 0 Critical/Important** (tái lập fix sort + xác minh chuỗi dtype + không caller nào vỡ).
- **Decisions:** (1) Plan Phase 6 viết TRƯỚC Phase 5 — pre-flight phát hiện 2 xung đột, **user duyệt 2 deviation**:
  (a) dùng `MiniBrainRepository` + `PoolPnlModel` Phase 5 (đã có) thay vì thêm vào `AlphaRepository`+tạo model thứ 2
  như plan gốc; (b) chỉ building blocks — KHÔNG wire vào `RefinementLoop` sống (loop hiện dùng `score_local_gate`
  Phase 3, defer wire `evaluate_with_pool` + `save_pool_pnl(if passed)` sang Phase 7/8). (2) Fix sort dates THUỘC
  6.1 (không defer 6.3) — alignment correctness là trách nhiệm đơn vị sở hữu logic; type alias `Dates`/docstring
  không tuyên bố precondition sort.
- **In progress:** —
- **Blockers / open risks:** 2 Minor defer Phase 7/8: `_pairwise_rho` dup-date lossy (benign — PnL thật không trùng
  ngày; có thể dedupe hoặc doc precondition); `evaluate_with_pool` bỏ `worst_id` (P7 nên nhét vào `hard_failures`
  để rejection actionable cho refiner + thêm `if verdict.passed: repo.save_pool_pnl(...)` khi wire loop). Gap#3
  bulk OHLCV không ảnh hưởng P7.
- **Next step:** Phase 7 — GP Engine (plan `2026-06-24-phase-7-gp-engine.md`): seed init typed-tree, typed cross/mut,
  multi-obj fitness gồm pool+pop corr penalty (NSGA-II/sharing), persist mọi individual, joblib parallel, sub-expr
  + result cache. ĐÂY LÀ NƠI WIRE pool corr vào loop sống.
- **Tests:** Xanh. 768 pass / 1 psycopg tiền-tồn. Mới: test_pool_corr (9, gồm fix sort regression) + test_gates_pool_corr
  (4) + chỉnh test_minibrain_repository (+1 mới, sửa 2 cũ cho format tuple) + integration test_pool_corr_gate (1).

### [2026-06-26] Session 10 — Phase 7 GP Engine building blocks (subagent-driven + opus 2-round, merged main)
- **Phase:** Phase 7 — GP Engine BUILDING BLOCKS (7.1-7.6). HOÀN TẤT, merged main + pushed `3e36bc1..2553e08`.
- **Done:** Thực thi plan `2026-06-24-phase-7-gp-engine.md` bằng **subagent-driven-development**: 6 task TDD,
  mỗi task 1 implementer (sonnet) + task-reviewer + fix-loop. **7.1** Individual (slots NON-frozen vì test gán
  fitness sau init); **7.2** FitnessVector 6 chiều + from_metrics; **7.3** Seeds (`generate_candidates()` API thật,
  side-effect import operators_local cho registry); **7.4** Init (ramped half-and-half, deviation `min_depth=1`
  param backward-compat); **7.5** Variation (typed crossover + point_mutation type-aware WINDOW resample từ
  window_choices; fix Important: brief code mẫu perturb Gaussian mọi Constant kể cả WINDOW); **7.6** NSGA-II
  Selection (đảo dấu, liệt kê tay 6 field thay getattr cho mypy). 3 fix giữa task: (a) dấu tiếng Việt seeds.py
  (subagent xóa dấu); (b) Important Task 7.5 WINDOW type-aware; (c) **final fix LỚN sau opus round-1**.
- **Final review opus 2-round:** Round-1 = With fixes. **4 Critical typed-invariant + 2 Important** chỉ lộ ở
  whole-branch gate (mỗi task review riêng đều OK vì chỉ kiểm scope task): (1) `_random_leaf` đặt Constant ở PANEL
  slot; (2) `subtree_mutation` thay không phân biệt vai trò → ~50% type-invalid; (3) `hoist_mutation` nhấc Constant
  lên root → 75% trên seed kinh điển; (4) `crossover` cho phép swap Constant-root vào PANEL. Cùng nguyên nhân gốc:
  thiếu helper PANEL-subtree dùng chung. **Important:** 4 op (regression_neut/vector_neut/ts_decay_linear/ts_delay)
  thiếu `gp_usable=False` → vi phạm B5 stage separation; dấu tiếng Việt thiếu trong test_gp_init/seeds.py.
  **Fix 7954468:** helper `_panel_compatible_subtrees(root, registry)` dùng chung 4 nơi + `_random_leaf` typed
  `kind=PANEL/SCALAR` + `gp_usable=False` cho 4 op + khôi phục dấu test files + 5 stress test 1000-iter
  `_check_panel_invariant` qua spec.signature (RED→GREEN); 2 test cũ encode defect đã flip cho khớp invariant đúng.
  **Round-2 = Ready=YES, 0 Critical/Important.** Suite **827 pass / 1 psycopg tiền-tồn**.
- **Decisions:** (1) Phạm vi Phase 7 = chỉ 7.1-7.6 building blocks (plan dừng dở ở 7.6 Selection). 7.7 GPEngine
  integration + 7.8 wire RefinementLoop + 7.9 defer (cần viết bổ sung plan hoặc gộp Phase 8). (2) Implementer
  sonnet KHÔNG haiku (file tiếng Việt có dấu — haiku xóa). Final review opus 2-round (mạnh nhất theo SDD). (3)
  Fix gộp 6 finding final review thành 1 implementer round (đúng SDD red-flag "ONE fix per finding list, không
  per-finding fixers"). (4) `Individual` slots NON-frozen (brief 7.1 cố ý — test gán fitness/generation sau init).
  (5) `dataclass` `_objective` if/elif liệt kê tay 6 field thay getattr động (mypy strict mất type-narrow trên
  frozen+slots). (6) Side-effect import `operators_local` trong seeds.py có docstring + noqa giải thích — bắt buộc
  cho registry validate parse, không phải hack.
- **In progress:** —
- **Blockers / open risks:** Minor defer (xem Current state). (P5) config_json/data_window opaque key (P7/8 cache).
  (P6) `_pairwise_rho` dup-date lossy; `evaluate_with_pool` bỏ `worst_id` (Phase 8 nên dùng).
- **Next step:** Phase 8 — Short-list + CLI HOẶC bổ sung plan 7.7-7.9. User chốt sau.
- **Tests:** Xanh. 827 pass / 1 psycopg tiền-tồn. Mới (7 file): test_gp_individual (6) + test_gp_fitness_vec (11)
  + test_gp_seeds (6) + test_gp_init (8) + test_gp_variation (15, gồm fix WINDOW) + test_gp_selection (8) +
  test_gp_panel_invariant (5 stress 1000-iter typed invariant). Sửa: test_lang_registry (2 test encode defect
  cũ flip cho khớp invariant B5 đúng).

### [2026-06-26] Session 11 — Phase 7.7-7.9: GPEngine + adapter + CLI (Phase 7 hoàn tất 100%)
- **Phase:** Phase 7 — GP Engine, hoàn tất phần integration (7.7-7.9) sau khi building blocks 7.1-7.6 đã merge.
- **Done:** (7.7) `src/gp/engine.py` — `GPEngine.run()` vòng lặp tiến hóa μ+λ end-to-end: init_population (seed
  cores + ramped) → `_make_offspring` (crossover/point/subtree/hoist/copy theo rate) → đánh giá offspring TRƯỚC
  chọn lọc → `dedup_population` → `nsga2_select`; `_evaluate_individual` (eval→portfolio→backtest→metrics→gate,
  trả (fv,status,reasons,bt)); `_persist` (upsert_expression + record_evaluation mọi outcome pass/fail/seed +
  save_pool_pnl khi pass); `_config_json` sort_keys=True. Test: 10 unit + 2 integration (small_panel + DB thật).
  (7.8) `src/gp/seed_adapter.py` — GPSeedGenerator implement Protocol `idea_generator.generate_ideas(n)` cho
  RefinementLoop (3 test); CLI `main.py generate --method=gp` (ParquetSource.load → GPEngine.run) thay
  TemplateGenerator; xóa `src/generation/template.py` + `tests/test_template.py`. (7.9) merge --no-ff → main
  (6d9ae8d), pushed origin.
- **Decisions:** (1) **Chữ ký building blocks lệch plan** — `init_population(registry,rng,population_size,
  seed_cores,fields,max_depth)`, `crossover(a,b,rng,max_depth)` (KHÔNG nhận registry), `subtree_mutation`/
  `point_mutation` cần `fields: tuple[str,...]`, `all_seed_cores(*,with_llm=...)` keyword-only — adapt run() theo
  file thật, KHÔNG sửa Phase trước. (2) **Đánh giá offspring TRƯỚC `nsga2_select`** (μ+λ chuẩn): selection.py
  assert mọi cá thể có fitness, nên không thể đưa offspring chưa eval vào — lệch mô tả plan Step 10 (plan định
  select rồi mới eval thế hệ cuối), nhưng đúng NSGA-II và đúng assertion. (3) `_persist` tái lập metrics từ bt
  cho passed/failed_gate (redundant 1 lần compute — minor defer như plan ghi).
- **In progress:** —
- **Blockers / open risks:** engine.py ~350 dòng (plan ước <300; phần dôi là docstring tiếng Việt chi tiết —
  chấp nhận). Redundant `pool_corr.max_corr` gọi 2 lần/cá thể (defer). main.py còn 13 lỗi ruff tiền-tồn (E402
  import + F841 ở lệnh sweep-config) — KHÔNG thuộc lệnh generate mới, defer dọn legacy.
- **Next step:** **Phase 8 — Short-list + CLI** (`docs/superpowers/plans/2026-06-24-phase-8-cli.md`).
- **Tests:** Xanh. Full suite 840 pass / 1 psycopg tiền-tồn (`test_db_postgres`). Mới: test_gp_engine (10) +
  test_gp_seed_adapter (3) + test_gp_engine_run integration (2). ruff sạch src/gp; mypy --strict sạch engine.py +
  seed_adapter.py.
