# MiniBrain architecture (condensed)

The full code-level design (class signatures, full schema, mermaid graph) is in
`docs/MINIBRAIN_DESIGN.md` Part B. This file is the always-available digest. When the two
disagree, the master spec wins — update this file to match.

## Contents
1. Layering & dependency rule
2. Data model & the market-source port
3. Expression AST
4. Operator registry
5. Evaluator invariants
6. Portfolio & backtest (stage separation)
7. Metrics, fitness & gates
8. Pool correlation (the killer feature)
9. Calibration harness
10. Database schema (summary)
11. Cache tiers
12. Genetic Programming

## 1. Layering & dependency rule
Layers: `market` (source port + panel) → `lang` (AST, registry, parser) → `operators` →
`engine` (evaluator) → `backtest` (portfolio, backtester, metrics, pool_corr) →
`store` (db, repository) / `gp` (search) → `pipeline` / `validate` (orchestration).

Rule: `lang`, `operators`, `engine`, `backtest` must NOT import from `gp`, `store`, or
`pipeline`. Orchestration is network-agnostic and takes the data source + repository as
injected dependencies, so it is testable with fakes.

## 2. Data model & the market-source port
`MarketData` (frozen, slots): `dates (T,)`, `assets (N,)`, `fields: dict[str, (T,N)]`,
`universe (T,N) bool`, `returns (T,N)`, `groups: dict[str, (T,N) int]`. Out-of-universe
cells are NaN. `Panel = np.float64 (T,N)`.

`MarketDataSource` is a `Protocol` (port). MiniBrain depends on it, never on a concrete
feed. The adapter owns PIT correctness, universe-membership history, and the delay
convention. Ship a real `parquet_source` adapter over whatever feed exists. **Brain's API
gives field/operator METADATA, not the historical values** — the data source is your own.

## 3. Expression AST
Sealed hierarchy: `Constant(value)`, `Field(name)`, `Call(op, args)`; abstract `Node` with
`accept(visitor)` + `children()`. Scalar params (windows, thresholds) are `Constant`
leaves; panel args are sub-trees. Visitors (one class each, single responsibility):
`DepthVisitor`, `FieldCollector`, `CanonicalHasher`, `ComplexityVisitor`, `Serializer`,
`Evaluator`. Canonical hash (commutative-arg sort + literal normalization) drives caching
and dedup.

## 4. Operator registry
One `OperatorRegistry` is the single source of truth for the parser (validate op/arity),
the evaluator (dispatch), and the GP (type-correct construction). `OperatorSpec`: `name`,
`category`, `signature: tuple[ArgKind,...]` (`PANEL`/`WINDOW`/`SCALAR`/`GROUP`), `impl`,
`bounded`, `depth_cost`, `gp_usable`, `window_choices`, `commutative`. Operators
self-register via a `@register` decorator so each `operators/*.py` is independently
importable. `registry.gp_function_set()` returns cores only (config wrappers excluded).

Operator-fidelity notes (see the `worldquant-brain` skill for full semantics): `ts_delay`
not `delay`; `ts_rank` bounded (default), `ts_zscore` unbounded; `ts_backfill` mandatory
for fundamentals, applied close to the raw field; `rank`/`winsorize`/`scale`/`truncate`
are rank-preserving and cannot change correlation; `group_neutralize` is a config wrapper;
`regression_neut`/`vector_neut` are the only decorrelation levers; `trade_when` is the main
conditioning lever; never `hump`/decay a fast signal whose turnover is the alpha.

## 5. Evaluator invariants
Out-of-universe cells stay NaN (never 0); every op NaN-propagates. Cross-sectional ops
operate per-row over in-universe cells only. Time-series ops read only rows ≤ t;
insufficient history → NaN. The sub-expression cache (keyed by canonical node hash) is the
main throughput lever — populations share subtrees heavily — applied before any numba.

## 6. Portfolio & backtest (stage separation enforced here)
The AST is the bare signal core. `PortfolioConfig`: `neutralization`
(NONE/MARKET/SECTOR/INDUSTRY/SUBINDUSTRY), `decay` (0=off), `truncation` (per-name |w|
cap), `scale_book`, `delay` (1). `PortfolioBuilder.build(signal, cfg, data) -> weights`:
optional decay → neutralize (cross-sectional demean / group demean) → truncate →
scale to dollar-neutral book → apply delay. `Backtester.run(weights, data)`:
`pnl_t = nansum(weights_{t-delay} * returns_t)` → daily PnL + equity curve.

## 7. Metrics, fitness & gates
`AlphaMetrics`: `sharpe`, `annual_return`, `turnover`, `max_drawdown`, `fitness`,
`per_year_sharpe: dict[int,float]` (regime robustness, first-class), `weight_concentration`.
`sharpe = mean(pnl)/std(pnl)*sqrt(252)`; `turnover = mean_t sum_i |w_t - w_{t-1}|`;
`fitness = sharpe * sqrt(|annual_return| / max(turnover, 0.125))` — Brain's exact formula
is not public; treat as RELATIVE ranking only. Hard gates: syntax/type/arity, depth ≤ cap,
all fields valid, PnL self-corr < 0.70, concentration ≤ cap. Soft scores (tradable in
search): Sharpe (deflated for multiple testing), Fitness, turnover band, per-year-min
Sharpe, pool correlation. Numbers come only from `config/thresholds.py`.

## 8. Pool correlation (the killer feature)
`PoolCorrelation.max_corr(candidate_pnl, dates) -> (max|ρ|, worst_alpha_id)` over every
passed alpha's stored daily PnL vector (aligned on common dates). It is a LOCAL PROXY for
Brain's self-corr gate — quota-free and high-value, but Brain's checker is authoritative
pre-submission (the proxy can diverge). The GP uses `max_corr` against both the pool and
the current population to keep candidates decorrelated.

## 9. Calibration harness
`CalibrationHarness.run(brain_records) -> CalibrationReport` re-scores already-Brain-
simulated alphas locally and reports `spearman_sharpe` (the headline), `spearman_fitness`,
`self_corr_agreement`, `decile_hit_rate`, and `by_year`. Gate the whole tool on a minimum
ρ before trusting its ranking. This is the validity check for the entire project — without
it MiniBrain is unfalsifiable.

## 10. Database schema (summary)
SQLite via SQLAlchemy Core (ports to Postgres later). Tables: `expression`
(canonical_hash UNIQUE, expr_string, depth, complexity, fields), `evaluation`
(expression_id, config, data_window, metrics, per_year, self_corr_max, status,
fail_reasons, seed; UNIQUE(expr,config,window)), `pool_pnl` (evaluation_id → packed
dates+pnl blobs, for passed alphas), `dead_field` (name PK, reason), `brain_record`
(ground truth for calibration). `AlphaRepository` wraps it with typed methods
(`upsert_expression`, `record_evaluation`, `load_pool`, `add_dead_field`,
`result_cache_get/put`, `top_n`). Persist failures too.

## 11. Cache tiers
(1) Field cache — partitioned Parquet for raw data (immutable, large; never CSV).
(2) Sub-expression cache — in-memory LRU keyed by canonical node hash, scoped to one GP
run; the main throughput win. (3) Result cache — DB-backed
`canonical_hash + config + window → AlphaMetrics`; re-scoring a known alpha is free.
Canonicalization raises hit rates across all three.

## 12. Genetic Programming
Representation: the typed AST. Init: ramped half-and-half **seeded** with economically-
grounded factor-family templates (`gp/seeds.py`) — pure random GP wastes evaluations.
Function set: cores only (`gp_usable`). Multi-objective `FitnessVector`: deflated Sharpe,
per-year-min Sharpe, turnover penalty, complexity penalty, pool-correlation penalty,
population-correlation penalty. Selection: NSGA-II Pareto fronts / fitness-sharing
(correlation-aware) so the population stays diverse. Variation: typed subtree crossover,
point mutation (op/field/window), subtree mutation, hoist mutation (anti-bloat). Depth cap
≈7; validity repair; canonical-hash dedup vs population + DB. Loop: evaluate (joblib
parallel, caches on) → score → select → vary → elitism → next gen; persist every outcome.

GP is appropriate HERE (unlike the online quota-constrained pipeline where it was removed)
because local evaluation is cheap, so GP's sample-inefficiency is acceptable. But the
saturation lesson holds in full: correlation-aware selection is mandatory.
