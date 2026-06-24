# MiniBrain — Progress log

> Append-only development journal. Maintained via the `session-journal` skill.
> Read the `Current state` block + the last few entries at the start of every session;
> append an entry and refresh `Current state` at the end of every session or phase.

## Current state
- **Phase:** Phase 1 — Parser ✅ HOÀN TẤT (sẵn sàng merge main). Tiếp theo: Phase 2 — Operator Engine.
- **Quyết định hướng đi:** Tích hợp MiniBrain vào tool sẵn có (KHÔNG build grenfield). Code mới
  đặt trong `src/` (không phải `minibrain/`), tái dùng login/fetch/DB/sim/AI/submit. Mỗi phase =
  1 nhánh git → merge main → push. **Bỏ đường cũ** (LLM→sim trực tiếp): mọi candidate qua local
  gate trước khi đốt sim (gỡ tại Phase 3). Spec: `docs/superpowers/specs/2026-06-23-minibrain-into-existing-tool-design.md`.
  Plans: `docs/superpowers/plans/2026-06-24-minibrain-integration-master-plan.md` (P0-P9) +
  `2026-06-24-phase-1-parser.md`.
- **Done (Phase 1):** `src/lang/ast.py` (Constant/Field/Call frozen+slots + `NodeVisitor` Protocol
  covariant), `src/lang/registry.py` (ArgKind/OpCategory enum, OperatorSpec, OperatorRegistry,
  decorator `@register`, `default_registry()` với 6 op tối thiểu impl placeholder), `src/lang/grammar.lark`
  (field/number/call/`+-*/`/unary-minus), `src/lang/parser.py` (`parse()` strict validate
  operator/arity qua registry + `parse_expression()` lenient cho caller legacy; CLI `python -m
  src.lang.parser`), `src/lang/visitors.py` (DepthVisitor đếm wrapper, FieldCollector, Serializer
  round-trip, CanonicalHasher sort-commutative+normalize-literal, ComplexityVisitor, hàm thuần
  `all_subtrees`/`iter_leaves`). Migrate 9 caller khỏi `ast_utils` + XÓA `src/generation/ast_utils.py`
  + `tests/test_ast_utils.py`. 87 unit test lang xanh; full suite 590 pass; ruff + mypy --strict
  (src/lang) clean.
- **In progress:** —
- **Next step:** Phase 2 — Operator Engine. Viết plan đã có sẵn `docs/superpowers/plans/2026-06-24-phase-2-operator-engine.md`;
  thực thi `src/operators_local/*` + `Evaluator` (AST→signal (T,N)) + golden test; nạp impl thật
  cho 6 op placeholder + bổ sung operator còn lại; áp invariant no-look-ahead + NaN-out-of-universe.
- **Blockers / open risks:** (R2/Gap#3) Gap bulk OHLCV chưa giải (calibration ρ phụ thuộc). Validate
  "field tồn tại trong `available_fields()`" CHƯA làm (thuộc Phase 2 khi có MarketData thật). Legacy
  `src/data/client.py` 9 lỗi mypy pre-existing + `test_db_postgres` 1 fail pre-existing (thiếu psycopg).
- **MVP (Phases 1–3) reached:** no
- **Calibration ρ (Spearman, Sharpe):** not measured yet

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
