---
name: worldquant-brain
description: >-
  Domain knowledge for building, validating, and submitting alpha signals on the
  WorldQuant Brain (WQ Brain) platform using the FASTEXPR language for the TOP3000
  US equity universe (Delay 1). Use this skill whenever the task touches WQ Brain,
  FASTEXPR expressions, alpha generation/refinement, self-correlation or submission
  gates, factor families, or the alpha-engine codebase — even if the user does not
  name the platform explicitly (e.g. "write an expression", "why is my alpha
  rejected", "reduce correlation", "tune neutralization/decay"). Consult this skill
  before writing or editing any FASTEXPR string, any code that generates/scores/
  submits alphas, or any constraint/gate logic, because WQ-specific operator
  semantics, field-naming rules, depth limits, and the self-correlation gate are
  easy to get subtly wrong and silently produce bad or unsubmittable alphas.
---

# WorldQuant Brain alpha engineering

This skill encodes hard-won platform knowledge so alphas are *valid*, *economically
grounded*, and *submittable* on the first pass instead of burning simulation quota
on garbage. The platform punishes subtle mistakes silently: an expression can be
syntactically fine, simulate cleanly, and still be unsubmittable (self-corr) or
structurally crippled (depth budget). The rules below exist to prevent exactly those
silent failures.

## Cardinal rules (violating these wastes quota or produces dead alphas)

1. **Verify every field name before using it.** Field names are platform-specific and
   change across datasets. Never invent a field. Check it exists in the field
   repository / Data Explorer first. In code, fields must pass through the validated
   field set; a hallucinated field gets rejected by WQ and should be blacklisted so it
   is never proposed again.

2. **`ts_delay`, not `delay`.** The valid time-shift operator is `ts_delay(x, d)`.
   `delay` is not a valid FASTEXPR operator here. Same family discipline applies to
   all `ts_*` operators.

3. **`ts_backfill` is mandatory for fundamental data.** Fundamental fields are sparse
   (reported quarterly, NaN between reports). Without `ts_backfill(field, d)` the
   signal is mostly NaN and the alpha is dead. Price/volume fields generally don't
   need it.

4. **Self-correlation ≤ 0.70 is a HARD submission gate, and only neutralization
   operators can fix it.** `regression_neut` / `vector_neut` are the *only* levers that
   surgically remove a correlated component. Every rank-preserving transform
   (`winsorize`, `scale`, `truncate`, `rank`) leaves correlation untouched. Treat pool
   self-correlation as a first-class objective, not an afterthought. See
   `references/constraints-and-gates.md`.

5. **Mind the depth budget (≈7 levels).** The standard wrapper stack
   `scale(ts_decay_linear(group_neutralize(...)))` already consumes 3 levels, leaving
   only ~4 for the signal core. A multi-field conditional core will not fit underneath
   the full wrapper. **Build and test the bare signal core first; apply
   neutralization/decay/truncation as a separate configuration stage.** See rule 7.

6. **Hypothesis before formula.** Never emit an expression without an economic
   rationale (observation → mechanism → expected effect → specification). Zero-shot
   formulas collapse onto crowded reversal/textbook factors. Label which parts are
   literature-grounded vs. engineering adaptation. See `references/factor-families.md`.

7. **Expression search and configuration search are separate stages.** First establish
   that the *signal core* has edge. Only then tune neutralization, decay, truncation,
   and universe. Mixing the two confounds what is actually driving performance and
   wastes depth budget during the search phase.

## Workflow for producing a new alpha

1. **State the hypothesis** (4 parts: observation, theoretical basis, economic
   mechanism, specification). Confirm the data fields exist.
2. **Write the bare signal core** in FASTEXPR — no wrappers yet. Keep within depth so a
   conditional/multi-field core stays legal once wrapped later.
3. **Check originality cheaply** before simulating: structural similarity (AST) against
   the reference zoo / already-passed alphas. Near-duplicates of a saturated family
   will fail the correlation gate later — don't spend a simulation on them.
4. **Simulate the core**, read the full year-by-year table (not just aggregate
   metrics). Change one variable at a time when iterating.
5. **Configuration stage:** add neutralization/decay/truncation. If pool self-corr is
   the binding dimension, reach for `regression_neut` / `vector_neut` against the
   crowded component — nothing else moves it.
6. **Pre-submission gates:** verify self-corr against WQ's *actual* checker (a local
   proxy can diverge), per-year Sharpe (regime fragility hides in aggregates), weight
   concentration, turnover, and margin. See `references/constraints-and-gates.md`.

## Signal construction defaults (regime-stable choices)

- Prefer **bounded** signals (`ts_rank`) over **unbounded** (`ts_zscore`) — bounded
  signals don't amplify wrong-way bets in extreme regimes (e.g. 2021 meme-stock).
- **Do not smooth fast signals.** If high turnover *is* the alpha, smoothing destroys
  returns faster than it cuts turnover.
- Crowded textbook factors rarely clear threshold standalone in TOP3000 US. Edge comes
  from **non-obvious conditioning** (volume-gating via `trade_when`, range/position
  context), not from a cleaner version of a known factor.

## Reference files (read when relevant)

- `references/fastexpr-operators.md` — operator semantics, signatures, common combos,
  and pitfalls. Read before composing or debugging any non-trivial expression.
- `references/constraints-and-gates.md` — self-correlation, depth, submission
  thresholds, pre-sim and OOS/regime gates, and which operator fixes which gate.
- `references/factor-families.md` — economic rationale library, which families are
  saturated in a crowded TOP3000 US pool, and regime complementarity.
- `references/codebase-conventions.md` — conventions of the alpha-engine pipeline
  (stage separation, correlation as first-class, self-learning field blacklist, soft
  vs. hard gates) and how to investigate the repo instead of hardcoding assumptions.

When platform behavior (operator signature, current competition threshold, field
existence) is uncertain, **verify against the live platform (Operator/Data Explorer)
rather than guessing** — these are versioned and the cost of a wrong assumption is a
wasted simulation or an unsubmittable alpha.
