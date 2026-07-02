---
name: minibrain-builder
description: >-
  Engineering knowledge for building MiniBrain, a LOCAL WorldQuant Brain research
  platform that parses FASTEXPR into a typed AST, evaluates it over a market-data panel,
  runs a WQ-faithful portfolio/backtest, computes Sharpe/Turnover/Fitness plus pool
  self-correlation, and generates alphas via genetic programming. Use this skill whenever
  working on the MiniBrain codebase or any of its modules — parser, AST, operator
  registry, evaluator, portfolio/backtester, metrics/gates, calibration harness, database,
  cache, GP engine, CLI — or when implementing, reviewing, refactoring, or debugging any
  MiniBrain phase, even if the user does not name the skill or the platform. It encodes
  the architecture, the MVP-first phase plan, the per-phase build ritual, the coding
  rules, and the correctness invariants (no look-ahead, delay-1, stage separation, local
  vs Brain calibration, pool self-correlation) that are easy to get subtly wrong and that
  silently destroy the tool's value.
---

# MiniBrain builder

MiniBrain is a **local pre-filter** for WorldQuant Brain alpha research — not a clone of
Brain. Its job is to reject 80–95% of weak alphas and flag saturated/correlated ones
*before* a Brain simulation is spent. Build it the way a senior quant engineer ships a
first production version: a thin, correct, end-to-end spine first, proven against reality,
then scaled.

The exhaustive, directly-implementable design lives in **`docs/MINIBRAIN_DESIGN.md`**
(the master spec). This skill is the operating layer: the durable principles, the phase
plan, and the per-phase ritual. Read the master spec for the full class/schema/AST detail;
read the reference files here for the condensed, always-available version.

## North star (what success means)

- **The product is rank-correlation with Brain**, not Brain's exact Sharpe. MiniBrain is
  validated by the calibration harness (local Sharpe vs Brain Sharpe on already-simulated
  alphas). Do not trust MiniBrain's ranking until Spearman ρ on Sharpe clears the agreed
  bar (~0.5–0.6) on a held-out set. If it doesn't, the fix is upstream (data fidelity,
  operator semantics) — not more GP.
- **Pool PnL self-correlation ≤ 0.70 is the real submission blocker**, not raw Sharpe.
  This is exactly what a local backtester can check for free (no quota). It is a
  first-class gate, computed from stored PnL vectors of passed alphas — *not* AST hashing
  (structural similarity ≠ PnL correlation).

## Cardinal rules (violating these silently breaks the tool)

1. **No look-ahead / no survivorship bias.** Per-day universe mask everywhere;
   time-series operators read only rows ≤ t; PIT discipline on fundamentals; delay-1
   (weights at t trade returns at t+1). This is correctness, not a nicety — bias inflates
   local Sharpe and kills correlation with Brain.
2. **WQ operator fidelity.** FASTEXPR semantics ≠ naive pandas (NaN handling, `ts_rank`
   normalization, `group_neutralize`, decay weights, `ts_delay` not `delay`). For operator
   semantics, the self-correlation gate, depth budget, and factor families, **consult the
   `worldquant-brain` skill** — do not reimplement from memory.
3. **Stage separation.** GP searches **bare signal cores**. Neutralization, decay,
   truncation, scale, and delay live in `PortfolioConfig` and are applied in the config
   stage. Never let the generator emit a fully-wrapped expression — it wastes the ≈7-level
   depth budget and confounds attribution.
4. **Only `regression_neut` / `vector_neut` reduce self-correlation.** Every rank-preserving
   transform (`rank`, `winsorize`, `scale`, `truncate`) leaves correlation untouched. Tag
   operators accordingly so gate logic and GP never reach for a cosmetic transform to fix
   correlation.
5. **Thresholds live in one place** (`config/thresholds.py`). Never hardcode a gate number
   (Sharpe floor, 0.70 self-corr, turnover band, concentration cap) at a call site — they
   change per competition.
6. **Determinism.** All randomness flows through an injected seed; record the seed in the
   DB so every run is reproducible.
7. **Persist failures, not just successes.** Failed/invalid evaluations feed the avoid-list
   and the dead-field blacklist. A field rejected by Brain is blacklisted and never
   proposed again.
8. **Correlation-aware GP.** A Sharpe-greedy search breeds a saturated, mutually-correlated
   population — the exact failure that kills submissions. Diversity/novelty pressure (pool
   + population PnL correlation) is wired into the fitness from day one, not bolted on.

## Build workflow — the per-phase ritual (do not skip)

Implement strictly in the phase order (see `references/phases.md`). For **each** phase:

1. **Design** — restate that phase's interfaces/classes from the master spec; note any
   deviation and why. Do not skip the design step.
2. **Implement** — create the files for **that phase only**. Do not code ahead. Do not
   generate the whole repository in one pass.
3. **Explain** — 1–2 sentences per file on its purpose.
4. **Review** — run the phase's unit tests, the golden operator tests, and the
   parse→eval→backtest→metrics integration test. Report results and any open risk
   (especially anything touching look-ahead bias or operator fidelity).
5. **Gate** — only after the review is clean, advance to the next phase.
6. **Journal** — append a progress entry (see the `session-journal` skill) recording what
   was completed, decisions made, and the next step, so the next session can resume.

**MVP milestone = Phases 1–3:** parse a hand-written FASTEXPR alpha → evaluate → backtest →
read a Sharpe and an equity curve on real data. Stop, demo, and review there before
building the trust layer (calibration) and the scale layer (DB, pool-corr, GP).

## Engineering standard (summary)

Python 3.12, full type hints (`mypy --strict` clean), SOLID, depend on protocols not
concretions, every module independently importable and runnable, `ruff`-clean, no demo
code, no mocks unless a real dependency is genuinely unavailable. Full detail in
`references/conventions.md`.

## Reference files (read when relevant)

- `references/architecture.md` — condensed durable design: layering + dependency rule,
  AST, operator registry, evaluator invariants, portfolio/backtest, metrics/fitness, pool
  correlation, calibration, DB schema, cache tiers, GP. Read before designing any module;
  the master spec `docs/MINIBRAIN_DESIGN.md` has the full code-level detail.
- `references/phases.md` — the MVP-first phase plan: objective, output, dependencies,
  complexity, and a Definition of Done per phase. Read at the start of every phase.
- `references/conventions.md` — coding rules, correctness invariants, and the testing
  strategy (unit / golden / integration). Read before writing code.

When platform behavior is uncertain (operator signature, field existence, current
competition threshold), verify against the live WQ platform / the `worldquant-brain` skill
rather than guessing — a wrong assumption costs a wasted simulation or an unsubmittable
alpha.
