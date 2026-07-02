# MiniBrain conventions

Read before writing code. These are the rules the codebase must follow so it stays
correct, reproducible, and maintainable.

## Code standard
- **Python 3.12.** Use modern syntax: `match`, `X | None`, `type` aliases,
  `@dataclass(frozen=True, slots=True)`, `Protocol`, `enum`.
- **Full type hints** on every public function, method, and attribute. `mypy --strict`
  must pass. Use `numpy.typing` for array shapes via aliases (`Panel`, `Mask`, `Dates`).
- **SOLID.** One responsibility per class. Depend on protocols (`MarketDataSource`,
  `NodeVisitor`), not concretions. The registry + visitor patterns keep evaluation,
  hashing, and validation open for extension and closed for modification.
- **No demo / throwaway code.** No mocks unless a real dependency is genuinely unavailable
  (e.g. the live Brain network). Prefer fakes/fixtures over mocks.
- **Every module runs independently** — importable in isolation, with a focused test or a
  `__main__` demonstrating it (e.g. `python -m minibrain.lang.parser "rank(close)"`).
- **Create all files and folders** in the structure (master spec Part B2); leave no
  dangling imports on package import.
- **Lint/format** with `ruff`; no unused imports.
- **After creating each file, explain its purpose in 1–2 sentences.**

## Correctness invariants (non-negotiable)
- **No look-ahead.** Time-series operators read only rows ≤ t. Insufficient history → NaN,
  never a partial-window guess that peeks forward.
- **No survivorship bias.** Use the per-day universe mask; never backtest on current
  membership. Out-of-universe cells are NaN.
- **Delay-1.** Weights derived from data through close of day t are applied to returns of
  day t+1: `pnl_t = nansum(weights_{t-delay} * returns_t)`.
- **PIT fundamentals.** `ts_backfill` only over point-in-time-valid history; never backfill
  across a reporting boundary you wouldn't have known at t.
- **Stage separation.** Expression search produces bare cores; neutralization / decay /
  truncation / scale / delay are `PortfolioConfig`, applied in the config stage.
- **Determinism.** All randomness flows through an injected seed; record it in the DB.
- **Thresholds in one place.** Only `config/thresholds.py` holds gate numbers.
- **WQ operator fidelity.** Match FASTEXPR semantics (consult the `worldquant-brain` skill);
  do not approximate with naive pandas defaults.

## Testing strategy (per phase)
- **Unit tests** for each module's public surface.
- **Golden tests** for operator semantics: assert exact output panels for small fixed
  inputs, and — where available — reconcile against a handful of Brain-simulated reference
  values. Include a look-ahead test (row t output is unchanged when rows > t are altered).
- **Integration test:** one end-to-end `parse → eval → portfolio → backtest → metrics` run
  on the fixture panel, asserting a stable Sharpe within tolerance.
- Keep the suite green before advancing a phase. Tests use a small **real** fixture panel
  (in `tests/conftest.py`), not synthetic noise, so results are meaningful.

## What "production-ready" means here
A module is done when: it has full type hints and passes `mypy --strict`; it is `ruff`-clean;
it has unit + (where relevant) golden tests; it can be imported and run in isolation; it
honors every correctness invariant above; and its purpose is explained in 1–2 sentences.
