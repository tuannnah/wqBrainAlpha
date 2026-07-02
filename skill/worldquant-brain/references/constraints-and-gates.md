# Constraints & gates

Which checks block submission, why, and which lever fixes each. The dominant structural
blocker in a mature pool is **self-correlation**, so it is treated first.

## Self-correlation ≤ 0.70 (the dominant gate)
- A submitted alpha must correlate ≤ 0.70 with the user's existing pool. This is a hard
  cutoff, not a soft penalty.
- **Verify against WQ's actual correlation checker, not just a local proxy.** A
  post-simulation proxy and the platform's real checker can diverge; trusting the proxy
  alone produces alphas that pass locally and get rejected on submission.
- **Only `regression_neut` / `vector_neut` reduce it.** All rank-preserving transforms
  (`winsorize`, `scale`, `truncate`, `rank`) leave correlation unchanged. If correlation
  is the binding dimension, the refinement must map to a neutralization operator against
  the crowded component — not to a cosmetic transform.
- Within a **saturated family**, decorrelation tweaks systematically fail above the
  cutoff: improving signal quality and reducing self-correlation are often the same lever
  pulled in opposite directions. The escape is a *different* family or genuinely
  orthogonal conditioning, not another within-family variant.

## Depth limit (≈7 levels)
- The expression tree has a maximum depth (≈7). The standard wrapper stack
  `scale → ts_decay_linear → group_neutralize` consumes 3, leaving ~4 for the core.
- A multi-field conditional core (e.g. `trade_when` over a multi-term signal) easily
  exceeds the remainder once wrapped.
- **Mitigation:** treat wrappers as a separate configuration stage; search expression
  cores at full depth budget, then add only the wrappers that earn their level.
- Watch for **silent repair failure**: when a depth error occurs, a naive "extract the
  rejected field" repair step can return empty hints and loop without progress. Depth
  errors need a structural fix (drop a wrapper / flatten the core), not a field swap.

## Submission thresholds (verify current competition rules)
Typical gates — **confirm exact numbers against the current IQC/Challenge rules, since
they change per competition:**
- **Sharpe** above a floor (historically ~1.25 for submission). Standalone textbook
  factors in crowded TOP3000 US realistically cannot reach this alone.
- **Fitness** above a floor.
- **Turnover** within a band (too low = no trading edge; too high = cost/instability).
- **Weight concentration** capped — reject alphas whose book is dominated by a few names.
- **Margin / margin improvement** — enforce as a hard gate pre-submission where required.
- **Sub-universe / drawdown** checks depending on competition.

Keep the live thresholds in one place in the codebase so they are easy to update when a
new competition opens.

## Pre-simulation gates (save quota)
- **Originality (AST similarity)** against the reference zoo (Alpha101 + already-passed
  alphas) filters obvious structural duplicates *before* paying for a simulation. Note
  this is a *structural* check, not a PnL-correlation check — it can disagree with the
  real self-corr gate, so it is a cheap pre-filter, not a substitute for the real check.
- **Syntax / type / arity** pre-filter is local and cheap — always run before simulating.

## OOS and regime gates
- **No OOS data → regime-fragility gates are unreliable.** An aggregate Sharpe can hide a
  signal that only works in one regime.
- **Per-year Sharpe** is a first-class scoring dimension: inspect year-by-year, not just
  the aggregate. Historically weak years for specific families include 2019, 2020, 2023;
  reversal alphas were fragile in 2021–2022.
- Apply a **Sharpe haircut (deflation)** for multiple testing to counter in-sample
  overfitting introduced by greedy LLM refinement.

## Which gate → which lever (quick map)
- High self-corr → `regression_neut` / `vector_neut` against the crowded component.
- Depth error → drop a wrapper or flatten the core (NOT a field swap).
- Weak/unstable per-year Sharpe → change family or add conditioning; don't over-tune IS.
- High weight concentration → reconsider the cross-sectional transform / universe.
- Turnover out of band → `hump`/decay to lower it (but never on a fast signal whose
  turnover is the edge), or rethink signal speed.
