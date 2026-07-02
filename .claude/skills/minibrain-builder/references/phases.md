# MiniBrain phase plan (MVP-first)

Read this at the start of every phase. Implement strictly in order. The **MVP milestone is
Phases 1–3** (parse → evaluate → backtest → Sharpe on real data). Complexity: S < M < L < XL.
Each phase has a **Definition of Done (DoD)** — do not advance until every box is true and
the journal entry is written.

## Phase 0 — Data foundation  (complexity: M)
- **Objective:** repo scaffold + a loadable real market-data panel.
- **Output:** `MarketData` loaded from a small real window via the parquet adapter, with a
  per-day universe mask, returns, and sector groups; `config/settings.py`,
  `config/thresholds.py`, `pyproject.toml`.
- **Depends on:** —
- **DoD:** `MarketData` loads and passes shape/axis-alignment validation; universe mask is
  per-day (no survivorship); returns reconcile against the source; a fixture panel exists
  in `tests/conftest.py`; `mypy --strict` + `ruff` clean.

## Phase 1 — Parser  (complexity: M)
- **Objective:** FASTEXPR-subset string → typed AST + static validation.
- **Output:** `lang/ast.py`, `lang/registry.py` (skeleton), `lang/grammar.lark`,
  `lang/parser.py`; depth/field/serialize visitors.
- **Depends on:** 0
- **DoD:** `parse("rank(ts_mean((close-open)/open,5))")` round-trips through the serializer;
  unknown op / wrong arity / unknown field raise clear errors; depth visitor matches
  hand-computed depths; `python -m minibrain.lang.parser "<expr>"` runs.

## Phase 2 — Operator Engine  (complexity: L)
- **Objective:** implement operators against the registry; evaluate AST → signal panel.
- **Output:** `operators/*` (arithmetic, cross_sectional, timeseries, group,
  neutralization, conditional), `engine/evaluator.py`; golden tests.
- **Depends on:** 1
- **DoD:** every registered operator has an impl + a golden test; evaluator returns a
  `(T,N)` signal with correct NaN propagation and universe masking; no look-ahead (a
  golden test asserts row t depends only on rows ≤ t); `ts_delay`/`ts_rank`/`group_neutralize`
  match WQ semantics (cross-checked against the worldquant-brain skill).

## Phase 3 — Backtester  (complexity: M)  ← MVP
- **Objective:** portfolio construction + PnL (delay-1).
- **Output:** `backtest/config.py` (`PortfolioConfig`), `backtest/portfolio.py`,
  `backtest/backtester.py`.
- **Depends on:** 2
- **DoD:** a hand-written alpha runs end-to-end to an equity curve; weights are
  dollar-neutral and book-normalized; truncation caps per-name weight; delay-1 verified
  (PnL_t uses weights_{t-1}); integration test `parse → eval → portfolio → backtest` green.
  **Demo and review here before proceeding.**

## Phase 4 — Metrics + Gates  (complexity: M)
- **Objective:** full metric table + gate verdict.
- **Output:** `backtest/metrics.py` (`AlphaMetrics`, `MetricsCalculator`),
  `pipeline/gates.py` (`GateEvaluator`).
- **Depends on:** 3
- **DoD:** Sharpe/turnover/returns/drawdown/fitness/per-year/concentration computed;
  fitness uses `max(turnover, 0.125)`; all gate numbers read from `config/thresholds.py`;
  hard gates vs soft scores separated; per-year Sharpe surfaced (not just aggregate).

## Phase 4.5 — Calibration  (complexity: M, highest value-per-line)
- **Objective:** validate local metrics vs Brain on already-simulated alphas.
- **Output:** `validate/calibration.py` (`CalibrationHarness`, `CalibrationReport`); a
  loader for `brain_record` ground truth.
- **Depends on:** 4 (+ a CSV/DB of Brain-simulated alphas)
- **DoD:** `spearman_sharpe` computed on a held-out set of ≥ ~50 alphas; a documented
  minimum-ρ bar gates trust in MiniBrain's ranking; if ρ is below bar, the report points at
  the likely upstream cause (data fidelity / operator semantics). Do not skip this.

## Phase 5 — Database  (complexity: M)
- **Objective:** durable research store + result cache.
- **Output:** `store/db.py`, `store/models.py`, `store/repository.py`,
  `cache/result_cache.py`.
- **Depends on:** 4
- **DoD:** schema created/migrated; `AlphaRepository` upserts expressions and records
  evaluations (pass AND fail); result cache hits skip re-evaluation; dead-field blacklist
  works; seeds recorded.

## Phase 6 — Pool correlation  (complexity: S)
- **Objective:** local self-corr gate wired in.
- **Output:** `backtest/pool_corr.py`; gate consumes `max_corr`; `pool_pnl` populated for
  passed alphas.
- **Depends on:** 5
- **DoD:** passing an alpha stores its PnL vector; a new candidate's `max|ρ|` vs the pool is
  computed on aligned dates and enforced as a hard gate at 0.70; tested against a synthetic
  pair with known correlation.

## Phase 7 — GP Engine  (complexity: XL)
- **Objective:** automated generation at scale, correlation- and regime-aware.
- **Output:** `gp/individual.py`, `gp/seeds.py`, `gp/init.py`, `gp/variation.py`,
  `gp/fitness.py`, `gp/selection.py`, `gp/engine.py`.
- **Depends on:** 5, 6
- **DoD:** seeded init produces valid typed trees within the depth cap; typed
  crossover/mutation never produce invalid trees (or repair them); multi-objective fitness
  includes pool + population correlation penalties; NSGA-II / fitness-sharing keeps the
  population diverse (measured: population self-correlation stays bounded); every evaluated
  individual persisted; parallel via joblib; sub-expression + result caches engaged.

## Phase 8 — Short-list + CLI  (complexity: S)
- **Objective:** ranked, decorrelated candidate export + entrypoints.
- **Output:** `pipeline/shortlist.py`, `pipeline/runner.py`, `cli.py`
  (`score-one`, `generate`, `calibrate`).
- **Depends on:** 7
- **DoD:** `generate` emits a short-list that is internally decorrelated and pool-aware;
  `score-one` returns `(AlphaMetrics, GateVerdict)`; `calibrate` runs the harness.

## Phase 9 — Dashboard / submit assistant  (complexity: M)
- **Objective:** inspect runs; push the short-list.
- **Output:** reporting/export UI; optional Brain submit helper (metadata client only).
- **Depends on:** 8
- **DoD:** runs and metrics are inspectable; the short-list can be exported for Brain
  simulation; no secrets hardcoded.

## Ordering note
The brief's coding order is Parser → Operator Engine → Backtester → Metrics → Database →
GP. This plan follows it, inserting Phase 0 (nothing runs without data) and Phase 4.5
(calibration; the project is unfalsifiable without it). A thin result-cache table may be
pulled forward into Phase 4 if convenient, but the full schema is Phase 5.
