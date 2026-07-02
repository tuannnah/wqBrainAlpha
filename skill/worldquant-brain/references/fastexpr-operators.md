# FASTEXPR operators

Semantics and pitfalls for the operators that matter most. **Signatures (argument
names/order, optional params) are versioned — confirm the exact signature in the
platform's Operator Explorer before relying on a non-obvious parameter.** This file
describes *what each operator does and when to use it*; the platform is the source of
truth for exact syntax.

## Contents
- Time-series operators
- Cross-sectional operators
- Group operators
- Neutralization / decorrelation operators
- Conditional / turnover operators
- Scaling / cleanup operators
- Common combos
- Pitfalls

## Time-series operators
Operate per-instrument over a trailing window of `d` days.

- `ts_delay(x, d)` — value of `x` `d` days ago. The valid shift operator (NOT `delay`).
- `ts_delta(x, d)` — `x - ts_delay(x, d)`. Change over `d` days.
- `ts_mean(x, d)` — trailing mean.
- `ts_zscore(x, d)` — `(x - ts_mean(x,d)) / ts_std(x,d)`. **Unbounded** — can blow up in
  extreme regimes. Prefer `ts_rank` when regime stability matters.
- `ts_rank(x, d)` — rank of today's value within the trailing `d`-day window, normalized
  to ~[0,1]. **Bounded** — the regime-stable default.
- `ts_decay_linear(x, d)` — linearly weighted moving average, heavier weight on recent
  days. A smoothing/decay wrapper, not a signal core. Consumes a depth level.
- `ts_backfill(x, d)` — fill NaN with the last valid value up to `d` days back.
  **Required for sparse fundamental data.** Apply close to the raw field.
- `ts_regression(y, x, d, ...)` — rolling regression of `y` on `x` over `d` days; the
  return component (residual, slope/beta, intercept, prediction) is selected by a
  parameter. Confirm which return type you're getting before use.

## Cross-sectional operators
Operate across all instruments on a given day.

- `rank(x)` — cross-sectional rank, ~[0,1]. The standard way to make a raw signal
  comparable across names. Rank-preserving: does NOT change correlation structure.
- `winsorize(x, std=N)` — clip cross-sectional outliers at N standard deviations.
  Cleanup only; rank-preserving, so it cannot reduce self-correlation.

## Group operators
- `group_neutralize(x, group)` — subtract the group mean within each group (e.g.
  sector/industry/market), removing group-level exposure. A wrapper; consumes a depth
  level. Removes group *mean* exposure, not arbitrary correlated components.

## Neutralization / decorrelation operators
These are the ONLY operators that reduce self-correlation, because they remove a
correlated *component* rather than monotonically transforming the signal.

- `regression_neut(y, x)` — cross-sectional residual of `y` regressed on `x`. Use to
  strip out a specific crowded exposure (the component your pool is already long).
- `vector_neut(x, y)` — orthogonalize `x` against `y` (remove the projection of `x` onto
  `y`). Use to make a signal orthogonal to an existing/correlated alpha or factor.

When the binding constraint is pool self-correlation, the decorrelation lever is one of
these two against the crowded component. Reaching for `winsorize`/`scale`/`truncate`
here is a category error — they leave correlation unchanged.

## Conditional / turnover operators
- `trade_when(trigger, alpha, exit)` — only take/update the `alpha` position while
  `trigger` holds; flatten when `exit` holds. The primary tool for **event-gating** and
  **volume-gating** — the conditioning that gives crowded factors real edge.
- `hump(x, threshold)` — suppress position changes smaller than `threshold`, reducing
  turnover. Do NOT hump a fast signal whose turnover *is* the alpha.

## Scaling / cleanup operators
- `scale(x, scale=1)` — normalize to a target book (sum of absolute weights). Outermost
  wrapper. Rank/sign preserving.

## Common combos
- Standard wrapper stack (config stage): `scale(ts_decay_linear(group_neutralize(core), d))`
  — note this is already 3 depth levels; keep the core shallow.
- Sparse fundamental signal: `rank(ts_delta(ts_backfill(field, 60), 20))` — backfill
  before any time-series op.
- Volume-gated reversal: `trade_when(volume_condition, reversal_core, exit_condition)`.
- Decorrelate from a crowded factor: `vector_neut(new_core, crowded_factor)` then wrap.

## Pitfalls
- Using `delay` instead of `ts_delay` — invalid operator.
- Forgetting `ts_backfill` on fundamentals — signal is mostly NaN, alpha dies quietly.
- Wrapping a deep multi-field core in the full standard stack — exceeds depth budget;
  the repair often fails silently. Build core first, wrap second.
- Expecting `winsorize`/`scale`/`rank` to reduce self-correlation — they can't.
- Smoothing a high-turnover signal with `ts_decay_linear`/`hump` — kills the edge.
- Assuming an operator's optional parameter default — confirm in Operator Explorer.
