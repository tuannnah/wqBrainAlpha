# Examples / ⭐ Alpha Examples for Bronze Users 🥉

<https://api.worldquantbrain.com/tutorial-pages/sample-alpha-concepts>

# Valuation based on cash flow

**Hypothesis**

A lower EV/CF usually suggests the company is becoming cheaper relative to its cash-generating ability; a higher multiple suggests it’s getting more expensive.

**Implementation**

Use ts\_zscore to standardize the chang of the ratio and group\_rank to control the turnover.

**Hint to Improve Alpha**

There are various types of cash flow, and switching the type used in the metric may improve its performance.

```text
# Simulation settings: instrumentType=EQUITY, region=USA, universe=TOP3000, delay=1, decay=0, neutralization=INDUSTRY, truncation=0.08, pasteurization=ON, unitHandling=VERIFY, nanHandling=OFF, language=FASTEXPR, maxTrade=OFF, maxPosition=OFF
group_rank(-ts_zscore(enterprise_value/cashflow, 63),industry)
```

# Overpriced stocks

**Hypothesis**

When analyst price target estimates (est\_ptp) and free cashflow estimates (est\_fcf) move highly in sync over the past month (high positive correlation), it may signal that the market has already fully priced in the cash flow expectations into price targets — leaving little room for further upside.

**Implementation**

Using est\_ptp to capture price estimate and est\_fcf to capture free cash flow and calculate the dynamics between them with ts\_corr.

**Hint to Improve Alpha**

The window of 1 year might be too long to react on the price correction. Try shorter window.

```text
# Simulation settings: instrumentType=EQUITY, region=USA, universe=TOP3000, delay=1, decay=0, neutralization=MARKET, truncation=0.08, pasteurization=ON, unitHandling=VERIFY, nanHandling=OFF, language=FASTEXPR, maxTrade=OFF, maxPosition=OFF
-ts_corr(est_ptp,est_fcf,252)
```

# Volatility arbitrage

**Hypothesis**

Higher volatility is often observed during bearish markets, while lower volatility is typically seen during bullish markets. A lower Parkinson's volatility coupled with a higher implied volatility may suggest that there could be a stronger bullish sentiment for the stock in the future.

**Implementation**

Long the stock if its implied volatility significantly exceeds its historical volatility and short the opposite

**Hint to Improve Alpha**

Can you use ts\_backfill to avoid missing data on certain days?

```text
# Simulation settings: instrumentType=EQUITY, region=USA, universe=TOP200, delay=1, decay=0, neutralization=SECTOR, truncation=0.08, pasteurization=ON, unitHandling=VERIFY, nanHandling=OFF, language=FASTEXPR, maxTrade=OFF, maxPosition=OFF
implied_volatility_call_120/parkinson_volatility_120
```
