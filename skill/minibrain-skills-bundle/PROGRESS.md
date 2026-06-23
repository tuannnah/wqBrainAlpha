# MiniBrain — Progress log

> Append-only development journal. Maintained via the `session-journal` skill.
> Read the `Current state` block + the last few entries at the start of every session;
> append an entry and refresh `Current state` at the end of every session or phase.

## Current state
- **Phase:** Phase 0 — Data foundation ✅ HOÀN TẤT (merged main). Tiếp theo: Phase 1 — Parser.
- **Quyết định hướng đi:** Tích hợp MiniBrain vào tool sẵn có (KHÔNG build grenfield). Code mới
  đặt trong `src/` (không phải `minibrain/`), tái dùng login/fetch/DB/sim/AI/submit. Mỗi phase =
  1 nhánh git → merge main → push. **Bỏ đường cũ** (LLM→sim trực tiếp): mọi candidate qua local
  gate trước khi đốt sim (gỡ tại Phase 3). Spec: `docs/superpowers/specs/2026-06-23-minibrain-into-existing-tool-design.md`.
  Plans: `docs/superpowers/plans/2026-06-24-minibrain-integration-master-plan.md` (P0-P9) +
  `2026-06-24-phase-0-data-foundation.md`.
- **Done (Phase 0):** `config/thresholds.py` (ngưỡng tập trung), `config/settings.py`
  (market_data_dir, global_seed), `src/local_types.py` (Panel/Mask/Dates/Assets),
  `src/data/market_panel.py` (MarketData frozen+slots, validate shape/dates tăng/dtype),
  `src/data/market_source.py` (MarketDataSource Protocol), `src/data/universe.py` (mask per-day +
  sector codes), `src/data/adapters/parquet_source.py` (save/load round-trip), `src/data/market_fetch.py`
  (`_assemble_panel` thuần + `fetch_to_parquet` raise NotImplementedError — Gap#3 chưa giải), fixture
  `small_panel` trong `tests/conftest.py`. 18 unit test xanh, ruff clean, mypy --strict clean (module mới).
- **In progress:** —
- **Next step:** Phase 1 — Parser. Viết plan chi tiết `docs/superpowers/plans/2026-06-24-phase-1-parser.md`
  rồi thực thi: `src/lang/{ast,registry,parser}.py` + `grammar.lark` + visitors; cuối phase migrate
  caller & xóa `src/generation/ast_utils.py`.
- **Blockers / open risks:** (R2/Gap#3) WQ Brain KHÔNG cấp bulk OHLCV sạch — `fetch_to_parquet` còn
  raise NotImplementedError; cần probe API khi có phiên để nạp panel thật (calibration ρ phụ thuộc cái này).
  Tạm thời nạp panel qua `ParquetSource.save()`. Legacy `src/data/client.py` có 9 lỗi mypy --strict
  pre-existing (ngoài phạm vi). Deferred Minor: ParquetSource.load suy dates từ field; noqa save import.
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
