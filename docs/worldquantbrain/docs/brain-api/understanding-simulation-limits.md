# BRAIN API / ❗Understanding simulation limits

<https://api.worldquantbrain.com/tutorial-pages/understanding-simulation-limits>

The simulation limit encourages high yield research, ensure optimal system performance and fair usage for all consultants. Here's what you need to know:

# What counts as a simulation?

All successful simulations are counted, including child simulations in multi-simulations and simulations of previously existing Alphas. Duplicate Alphas that you simulate **are not** removed from the count.

# Tracking simulations on the Platform

The [**Alphas page**](https://platform.worldquantbrain.com/alphas/unsubmitted) with Today or Yesterday filter on Date Created can help estimate your simulation count. However, if you simulate the same Alpha on different days, the simulation will count, but the Alpha will only appear under the date it was first simulated. **Simulating the same Alpha on different days consumes your simulation quota, even though you do not see Alpha on Alphas page for Today.**

# Tracking simulations using BRAIN API

## Simulation limit details in API response

When posting a simulation, the response headers provide:

* **X-Ratelimit-Limit**: Total daily simulation limit.
* **X-Ratelimit-Remaining**: Remaining simulations for the day.
* **X-Ratelimit-Reset**: Time (in seconds) until the limit resets.

Example:  
simulate\_response.headers["x-ratelimit-limit"],  
simulate\_response.headers["x-ratelimit-remaining"],  
simulate\_response.headers["x-ratelimit-reset"]

# Managing simulation limits

To avoid exceeding the daily limit:

* Plan simulations effectively to avoid wasting your daily quota.
* Monitor your simulation count and pause or stop when nearing the limit.
* If using the API, implement logic to handle limits, such as pausing simulations when the count reaches the limit and resuming after the reset.
* On the Platform interface, a warning message will appear when less than 1,000 simulations remain: **You are 987 simulations away from breaching the daily simulation limit. If you exceed this limit, you will not be able to run additional simulations until tomorrow (EST time zone). Please plan accordingly.**

# How to use simulations effectively?

## Analyze data fields

To maximize the effectiveness of your simulations, it's essential to analyze the data fields available on the Platform and use the information provided on the [**Data page**](https://platform.worldquantbrain.com/data). The Data page includes useful details such as:

* **Coverage**: The percentage of instruments in the universe covered by the data field.
* **Date Coverage**: The percentage of days with data available over a 10-year period. Ensure sufficient historical data for meaningful back testing.
* **Description**: Explains the context of the data field, helping you understand its potential use cases. It can also be used to identify similar fields. Fields with similar descriptions might represent overlapping or related data. You can use tools like LLMs (Large Language Models) to find similar data field efficiently. Avoid technical fields such as **timestamps, dates, or symbol identifiers** (e.g., ISIN, CUSIP) that **are unlikely to contribute to Alpha generation**.
* **Alpha Count**: The number of Alphas that have used this data field. A low Alpha count might indicate untapped potential, while a high count could suggest the data field is well-explored.
  + **User Count**: The number of users who have utilized this data field. This can indicate how popular or challenging the data field is.

## Use BRAIN Labs for deeper insights

* **Visualize data fields:** Use BRAIN Labs to explore data fields in detail. Analyze daily coverage, turnover, typical values, and correlations with returns.
* **Experiment with data fields and Python:** In BRAIN Labs, you can recreate operators using Python to test how they affect the data field and its correlation with returns. This process can help you identify promising transformations and refine your approach.

## Utilize Alpha statistics

When simulating an Alpha, review additional PnLs and statistics:

* **Risk-handled performance:** Simulates using SLOW\_AND\_FAST neutralization.
* **Investability-constrained performance:** Simulates using MaxTrade ON.

## Optimize search space

![simulation_cutoff.png](https://api.worldquantbrain.com/content/images/bXVASTINaSSJaBjn07neVE3DXjw=/449/original/simulation_cutoff.png)

To make the most of your simulations, it's important to restrict your search space initially and focus on testing one element at a time. Once a signal is found, you can expand and refine your approach by testing similar elements

### Select data fields

* If you are working with a large dataset, analyze the descriptions of data fields and exclude irrelevant fields (e.g., timestamps, dates, or symbol identifiers like ISIN or CUSIP).
* Select only one data field from a group of similar fields to test for an initial signal. For example, if you have fields like implied\_volatility\_call\_30, implied\_volatility\_call\_120, and implied\_volatility\_call\_360, start with just one (e.g., implied\_volatility\_call\_30).
* Once a signal is found, you can experiment with substituting similar fields to improve the Alpha.

## Select operators

* Operators can be grouped into categories, such as:
  + **Aggregational**: ts\_mean, ts\_median, ts\_sum.
  + **Delta-based**: ts\_delta, ts\_av\_diff, ts\_zscore.
  + **Group-based**: group\_zscore, group\_rank.
* Start by testing only one operator from a group (e.g., ts\_mean for aggregational operators).
* If a signal is found, try replacing it with similar operators from the same group (e.g., switch from ts\_mean to ts\_median or ts\_stddev) to explore variations.

### Select parameters

* For time series operators, start with a single timeframe.
  + For **fast** signals, use shorter timeframes (e.g., 5, 10 or 21 days).
  + For **slow** signals, use longer timeframes (e.g., 63, 121 or 252 days).
* Once a signal is found, experiment with other timeframes to see if the signal improves.

### Select neutralization

Applies to both -- in-code neutralization and neutralization in settings.

* **Market**: For single-country regions.
* **Country**: For multi-country regions.
* **Industry**: For industry-specific neutralization.
* Combined neutralization in code: Country + Industry, for multi-country regions.
* **Risk** neutralization:
  + If you see a signal in the risk-handled performance (e.g., SLOW\_AND\_FAST neutralization), you can experiment with similar settings like SLOW, FAST, CROWDING, or REVERSION\_AND\_MOMENTUM. These are variations of SLOW\_AND\_FAST and may yield better results.
  + STATISTICAL neutralization is different, so should be tested separately.
* Begin by testing one neutralization type (e.g., market neutralization).
* If a signal is found, try other neutralization settings (e.g., industry, subindustry, sector) to refine the Alpha further.

## Tips

* Run 10-50 simulations initially to test the search space. If no signal is found, edit the template before running the full search space.
* Simulate daily to avoid wasting your simulation quota.
