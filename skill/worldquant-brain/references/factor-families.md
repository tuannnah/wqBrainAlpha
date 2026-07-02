# Factor families & economic rationale

Guidance for choosing *what* to express, not just how. The goal is edge that survives the
self-correlation gate in a crowded TOP3000 US pool — which means avoiding saturated
families and finding non-obvious conditioning.

## Hypothesis-first discipline
Every alpha starts as a 4-part hypothesis before any formula:
1. **Observation** — the empirical regularity.
2. **Theoretical basis** — the literature or market-structure grounding.
3. **Economic mechanism** — *why* it should produce returns (whose behavior, what
   friction).
4. **Specification** — the fields and operators that capture it.

Label literature-grounded elements vs. engineering adaptations explicitly. A formula with
no mechanism is a curve-fit and will not generalize out of sample.

## Saturation map (a mature TOP3000 US pool)
These families tend to be crowded; within-family variants usually fail the ≤0.70 gate:
- **Intraday reversal cluster** — vwap/open relationships, close-position-in-range,
  volume-anomaly gating. Heavily mined; treat as exhausted unless conditioning is novel.
- **Implied volatility / options signals** — also crowded.

When a line of experimentation hits a hard ceiling inside a saturated family, the
productive move is to **switch families**, not to generate another decorrelation tweak.
Recognize exhaustion early rather than iterating within a failing family.

## Structurally weak standalone families (in TOP3000 US)
- **Quality / profitability** (gross-profits-to-assets, Novy-Marx) — economically real
  but a low standalone Sharpe ceiling (~0.7), below typical submission thresholds. Useful
  as a *complement* or under conditioning, not as a standalone submission.
- More generally, **crowded textbook factors rarely clear threshold standalone.** Past
  successes came from non-obvious conditioning (e.g. volume-gating), not cleaner textbook
  factors.

## Regime complementarity
Different families carry the portfolio in different regimes:
- Reversal alphas are strong in mean-reverting regimes but fragile in trending/extreme
  ones (e.g. 2021 meme-stock, parts of 2022).
- Quality/value factors tend to work precisely when reversal is weak (e.g. 2022).
- Because the pool needs decorrelated, regime-complementary alphas — not more of the
  same — family diversification is itself a correlation-management strategy.

## Where edge actually comes from
- **Non-obvious conditioning** beats factor cleanliness. `trade_when`-style volume/event
  gating turns a tired factor into something orthogonal to the crowd.
- **Bounded signals** (`ts_rank`) over **unbounded** (`ts_zscore`) for regime stability.
- **Don't smooth fast signals** — if turnover is the alpha, smoothing destroys it.
- **Alternative datasets** (options/news/social/analyst/graph) are a source of genuine
  orthogonality versus a price/volume-saturated pool — steer generation toward them when
  the price/volume space is exhausted.

## Useful references (for grounding, not copying)
- "101 Formulaic Alphas" (Kakushadze) — vocabulary of formulaic structures.
- "Navigating the Alpha Jungle" (arXiv 2505.11122) — LLM-driven mining + refinement loop
  patterns aligned with this platform.
- AlphaAgent (KDD 2025), AlphaForge, AlphaGen — alternative search formulations (note:
  mostly different universes/objectives; adapt, don't transplant).
