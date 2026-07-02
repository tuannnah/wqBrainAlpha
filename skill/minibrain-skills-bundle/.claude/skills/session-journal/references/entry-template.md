# PROGRESS.md template & example

## File skeleton
A fresh `PROGRESS.md` should look like this — a live `Current state` block at the top, then
an append-only `Entries` section (newest first or oldest first; pick one and keep it
consistent — this project uses **newest at the bottom** so the log reads chronologically).

```markdown
# MiniBrain — Progress log

## Current state
- **Phase:** <e.g. Phase 2 — Operator Engine (in progress)>
- **Done:** <one-line summary of completed phases/milestones>
- **In progress:** <what's mid-flight>
- **Next step:** <the single next action>
- **Blockers / open risks:** <or "none">
- **MVP (Phases 1–3) reached:** <yes/no>
- **Calibration ρ (Spearman, Sharpe):** <value or "not measured yet">

## Entries
<!-- append-only; newest at the bottom -->
```

## Entry template
```markdown
### [YYYY-MM-DD] Session NN — <short title>
- **Phase:** <phase number/name and where within it>
- **Done:** <what was completed and verified this session>
- **Decisions:** <choices made and the reason; deviations from the design + why>
- **In progress:** <started but not finished>
- **Blockers / open risks:** <anything blocking, or risks to watch>
- **Next step:** <the single most important thing to do next>
- **Tests:** <suite status — green/red, and what was added>
```

## Filled-in example
```markdown
### [2026-06-30] Session 04 — Operator engine: time-series ops + evaluator
- **Phase:** Phase 2 — Operator Engine (≈70% done)
- **Done:** Implemented arithmetic, cross_sectional, and timeseries operators with golden
  tests. Evaluator walks the AST and masks out-of-universe cells to NaN. Added a look-ahead
  test (altering rows > t leaves row t unchanged) — passing.
- **Decisions:** ts_rank normalized to (0,1] over the trailing window incl. today, to match
  WQ (confirmed against the worldquant-brain skill). Chose an LRU sub-expression cache keyed
  by canonical node hash over memoizing in the visitor, so the GP can share it later.
- **In progress:** group.py and neutralization.py (regression_neut / vector_neut) not yet
  implemented; group_neutralize stubbed.
- **Blockers / open risks:** No Brain reference values yet for golden reconciliation —
  operator fidelity is asserted against hand-computed expectations only. Flag for Phase 4.5.
- **Next step:** Implement group + neutralization operators, then the Phase 2 integration
  test (parse → eval on the fixture panel).
- **Tests:** Green. 41 unit + 12 golden. mypy --strict and ruff clean.
```
