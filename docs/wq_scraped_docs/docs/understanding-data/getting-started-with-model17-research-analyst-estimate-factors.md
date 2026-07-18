# Understanding Data / Getting Started with Model17 (Research Analyst Estimate Factors)

<https://api.worldquantbrain.com/tutorial-pages/getting-started-model17-iqc2026s2>

Every publicly traded company has a small army of professional analysts watching it. These analysts are employed by banks and research firms. They study the company, build financial models, and publish their conclusions: *buy this stock, sell that one, here is what we think earnings will be next quarter*. **Model17** collects all of that activity in one place and turns it into clean, ready-to-use daily signals.

Think of this dataset as a live scoreboard of what Wall Street's professional opinion-makers are thinking and, more importantly, *changing their minds about*.

This deepdive covers the **Model17** dataset for the USA region.

# Dataset Highlight

The **Model17** dataset is classified under the **Model** category > **Estimates Models** subcategory.

* **Data Type:** MATRIX only
* **Delay:** 1 (43 Fields)
* **Universes:** TOP3000, TOP2000, TOP1000, TOP500, TOP200, TOPSP500
* **Coverage:** Around 42%

The ~42% coverage is not a data quality problem - it reflects reality. Smaller, less-followed companies simply do not have many analysts writing research on them. The dataset covers stocks where analyst opinion is active and meaningful. When you build signals from model17, about half the universe will be missing. That is expected behavior.

# Dataset Feature

The 43 fields in model17 organize into **five themes**. Each theme captures one dimension of the analyst-consensus picture. Knowing which theme a field belongs to tells you what question it is answering.

## Composite Rank Signals

The dataset ships four ready-made composite signals. They are built by blending estimates, revisions, recommendations, and technical indicators into a single score.

* **mdl17\_score** - an overall buy/sell/neutral sentiment score summarising the full analyst picture for a stock.
* **mdl17\_dynamicfocusrank** - a short-term rank signal focused on the **1-2 week** horizon. It reflects how strong the current buy or sell signal is over the near term.
* **mdl17\_score\_equityfocusrank** - a medium-term rank signal built for **2-3 month** holding periods, blending fundamental and technical inputs.
* **mdl17\_fnd\_focusrank** - a long-term rank signal for **1-2 year** investment horizons, emphasising fundamental value.

These are the highest-level signals in the dataset. Using rank or z-score on them is a good starting point. The tradeoff is that what goes into them is opaque. If you want to understand *why* a stock scores high, you need to look at the underlying theme fields below.

## Analyst Recommendations

When an analyst publishes research, the conclusion is a **recommendation**: buy, hold, or sell. These fields track what the current mix of recommendations looks like across all analysts covering the stock.

* **mdl17\_est\_z\_buyrec\_pct** - percentage of analysts currently saying *buy*.
* **mdl17\_est\_z\_sellrec\_pct** - percentage of analysts currently saying *sell*.
* **mdl17\_est\_z\_netrec\_pct** - buy recommendations minus sell recommendations, as a share of all recommendations. This is the cleaner single number. A score of +60% means 60% more analysts are bullish than bearish, scaled to the total.

Each of these has a **score\_** twin (e.g., **mdl17\_score\_buyrec\_pct**). The **est\_z\_** fields are closer to the raw data. The **score\_** fields apply additional processing. For most alphas either works - start with one and switch if the results are similar.

Recommendation data is slow-moving. Analysts do not reverse their views every week. These fields reward long-window operators and penalise short-window noise-chasing.

## Price Target Activity

Analysts also publish a **price target** - the price they think the stock should be worth. This is separate from their buy/sell label. Model17 tracks how those targets are moving.

* **mdl17\_est\_z\_uptarget\_pct** - percentage of analysts who raised their price target recently.
* **mdl17\_est\_z\_downtarget\_pct** - percentage of analysts who cut their price target recently.
* **mdl17\_est\_z\_nettarget\_pct** - target increases minus target decreases over the past four weeks, as a percentage of all price targets.

Price target changes are a faster-moving signal than outright recommendations. An analyst who is not ready to flip from buy to sell may still quietly lower their target. That quiet downward revision, aggregated across the analyst community, can sometimes precede a formal recommendation change.

The newer fields **pct\_price\_target\_increases\_recent\_period**, **pct\_price\_target\_decreases\_recent\_period**, and **net\_pct\_price\_target\_change\_recent\_period** are direct counterparts with cleaner naming. They measure the same concept.

## Earnings Estimates, Revisions, and Dispersion

This is the richest theme in the dataset in terms of field variety.

* **mdl17\_est\_z\_earningsrevision** - change in the average earnings estimate over the past month, scaled by the stock's price. This tells you not just which direction estimates moved, but how large that move was relative to the stock.
* **mdl17\_est\_z\_netearningsrevision** - percentage of analysts raising their FY1 (next fiscal year) earnings estimate, minus those lowering it. A positive number means the crowd is becoming more optimistic. A negative number means they are pulling back.
* **mdl17\_est\_z\_dtstsespe** - the **standard deviation** of all FY1 earnings estimates. This measures disagreement. When analysts largely agree on earnings, the number is small. When they are far apart, the number is large. High disagreement means the market is uncertain about what this company will earn.
* **mdl17\_score\_earningstorpedo** - the difference between what analysts expect the company to earn *next year* and what it actually earned *over the last four quarters*. A large positive value means analysts are projecting a big jump in earnings relative to recent history. The name comes from the idea that stocks can be "torpedoed" when that expected growth fails to materialise.

## Earnings Surprise and Long-Term Growth

* **mdl17\_est\_z\_earningssurprise** - the current quarter earnings surprise. This is the gap between what the company actually reported and what analysts had predicted. The field captures the magnitude of that gap.
* **mdl17\_est\_z\_epsgrowthest** - the median analyst forecast for long-term earnings growth over the next full business cycle (typically 3-5 years). This is a slow-moving, structural signal.
* **mdl17\_est\_z\_coverage** - the number of analysts currently covering the stock. This is as much a measure of market attention as it is of data quality. More analysts means more scrutiny and more rapid pricing of public information.

# Usage Advice

* **Experiment with neutralization.** Sector neutralization is a reasonable starting point, but model datasets can vary a lot depending on the subcategory. Try sector, industry, and subindustry to see what works best for the specific field you are using.
* **The est\_z\_ and score\_ twins often overlap.** Most key metrics publish in two flavours. The **est\_z\_** version is closer to the raw data. The **score\_** version applies additional model processing. In practice, start with either. If one gives weaker results, try the other before doing more complex engineering.
* **Coverage gaps are structural, not random.** A missing value in model17 usually means the stock has little or no analyst following, not that the data feed broke. Be careful when building signals that involve both high-coverage and low-coverage stocks in the same expression - the meaning of "no data" differs from "low score".
* **These are weekly snapshots, not tick-by-tick.** The underlying data updates weekly. Short-window operators like **ts\_delta(..., 1)** or **ts\_zscore(..., 5)** mostly capture noise on a dataset that refreshes this slowly. Prefer windows of 20 days or more. Use **ts\_backfill** to carry the last valid reading forward through days when the field has not yet updated.
* **Use ts\_backfill on estimate and revision fields.** A 5-10 day backfill is appropriate. Earnings estimates and recommendations do not change daily - the most recent reading is valid until the next one arrives.
* **Prefer group operators over raw rank.** Because model17 fields are pre-scored, the raw value is already meaningful. But **group\_rank(x, industry)** or **group\_zscore(x, sector)** remove level differences between industries or sectors before the ranking step, which is generally cleaner than a raw **rank(x)**.
