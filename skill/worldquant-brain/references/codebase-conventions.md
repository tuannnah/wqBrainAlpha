# Codebase & pipeline conventions

Conventions of the alpha-engine pipeline. **Do not hardcode assumptions about the repo
from this file — investigate the actual code first** (grep for the symbol, read the
module). This file states the design *principles* the code is meant to follow so changes
stay consistent with them.

## Architecture principles
- **Seven-layer pipeline:** data → generation (LLM / template) → pre-simulation gates →
  simulation → scoring → refinement loop → submission manager.
- **Shared infrastructure** (platform client, field/operator repositories, cache DB,
  local pre-filter, simulator) is reused across search strategies. Scoring,
  decorrelation/zoo, and the offline generate-and-select pipeline are shared — do not
  delete them when removing a search strategy.
- **Pure orchestration** (K-pass loop, simulation budget, multi-direction) is kept
  network-agnostic and testable with fakes. Keep it that way: orchestration takes
  callbacks and must not know about the network.

## Stage separation (enforce in code)
- **Expression search ≠ configuration search.** Establish core edge first; tune
  neutralization/decay/truncation/universe second. Don't let the generator emit a fully
  wrapped expression during the search phase — it wastes depth budget and confounds
  attribution.
- **Single-variable iteration.** When refining, change one parameter at a time and report
  the full year-by-year table before moving on. Flag any change that moves multiple
  variables at once as ambiguous.

## Correlation as a first-class citizen
- The pool self-correlation function is always wired in (not behind an optional flag),
  because self-corr is the dominant submission blocker.
- The refinement loop must be able to treat **pool self-correlation as a blocking
  dimension** and map that dimension's hint to a neutralization operator
  (`regression_neut` / `vector_neut`), since those are the only effective levers.
- Beware the greedy "fix the weakest dimension" strategy: improving the weakest dimension
  can move against a learned trade-off (e.g. fitness vs. correlation). Prefer a
  trade-off-aware step (or the MCTS variant) when greedy and the correlation gate
  conflict.

## Gates: soft vs. hard
- Prefer **soft penalties** for most constraints so the search can trade off against
  them; reserve **hard gates** for obvious failures (syntax, dead field) and true
  submission blockers (self-corr, and where required weight concentration / margin).
- Note the tension: the AST-originality threshold is *structural*, not PnL-based, so it
  sits awkwardly with the soft-penalty principle — keep it as a cheap pre-filter, and let
  the real PnL self-corr gate be authoritative.

## Self-learning dead zones
- A field rejected by the platform is blacklisted: removed from the validated field set
  for the session AND injected into the generation prompt so the LLM is told not to
  propose it again. Persist failures (not just successes) to feed an avoid-list.

## Generation steering (RAG over MCP for batch)
- For a batch pipeline, prefer **direct API/library calls** (HTTP client for the
  platform REST API, a SQL client for the persistent outcomes DB) over an interactive
  Memory MCP — MCP is optimized for LLM-in-the-loop interactive workflows.
- **Inject context into the prompt** (query the DB for saturated families and dead fields
  before each generation batch) rather than relying on model memory. Steer generation
  toward alternative datasets when price/volume is exhausted.

## How to investigate before changing code
- `grep` for the symbol across `src/`, `tests/`, `scripts/`, and the entrypoint before
  assuming where something lives or what depends on it.
- Read the field repository / Operator repository to confirm a field or operator exists —
  don't trust memory.
- Run the local pre-filter and the test suite before and after a change; keep the suite
  green and avoid dangling imports.
