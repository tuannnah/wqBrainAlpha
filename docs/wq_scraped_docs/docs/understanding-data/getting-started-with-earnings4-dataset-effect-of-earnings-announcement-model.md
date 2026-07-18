# Understanding Data / Getting Started with earnings4 dataset (Effect of Earnings Announcement Model)

<https://api.worldquantbrain.com/tutorial-pages/getting-started-earnings4-dataset>

Most volatility datasets answer one question: "what is implied volatility today?" **Earnings4** is interesting because it answers a much harder one. *\*Of the volatility the options market is pricing right now, how much is for the next earnings event, and how much is everything else?\**

The dataset breaks the option price into two pieces. The earnings-effect part, and the non-earnings part. It does this with a 12-event history of how the stock has moved on past releases. Plus a model forecast of how it could move on the next one.

Most fields in earnings4 come in pairs. A series *\*with\** the earnings effect in it, and one with the earnings effect cleanly removed (the `xern` suffix). The good Alphas almost always live in the *\*gap\** between the two. Some fields only update around the earnings event itself, so they sit flat or missing between releases and need careful handling

# Dataset Highlight

The **Earnings4** dataset is classified under the **Earnings** category > **Earnings Estimates** subcategory.

* **Data Type:** MATRIX and VECTOR
* **Delays:** 0 (125 Fields) and 1 (375 Fields)
* **Universes:** TOP3000, TOP2000, TOP1000, TOP500, TOP200, TOPSP500
* **Coverage:** Around 75% Every VECTOR-typed field must be reduced (converted to Matrix) before any cross-sectional or time-series operator can be used. The operator we use throughout this note is vec\_avg(...) - this operator changes the multiple data points of the vector to a single number per stock per day for other operators to be utilized.

# Dataset Feature

The 375 fields in earnings4 may sound overwhelming until you realise they group into **eight conceptual families**. Recognising the family a field belongs to tells you which operators are likely to work and which are likely to fight the data.

## Per-Event Earnings Move History

Fields **ern4\_ernmv1** through **ern4\_ernmv12** carry the stock's actual percent move on each of the last 12 earnings releases, as a decimal fraction. They are indexed from most recent (1) to oldest (12). **ern4\_ernmv1** is, on its own, the most used field in the dataset. It is the cleanest post-earnings drift proxy you can get. Today's signal is just the size of the last earnings reaction. Each **ernmvN** updates as a step function and is flat otherwise. So windowed operators like **ts\_delta** and ts\_zscore on these series are really tracking one thing - *whether a new earnings event has just landed*. That is more useful than it sounds.

## Aggregate Earnings-Move Statistics

**ern4\_absavgernmv** is the absolute average earnings-day move across the last 12 events. **ern4\_ernmvstdev** is its standard deviation. Together they trace the stock's "earnings volatility fingerprint" - how much, on average, and how erratically, the price reacts when results come out. These are the slow companions to the per-event series. They are best used as denominators or normalisers. Not as standalone signals.

## Constant-Maturity Implied Volatility

**ern4\_10div**, **ern4\_30div**, **ern4\_60div**, **ern4\_90div** are constant-maturity implied volatilities interpolated to fixed calendar tenors (10, 30, 60, 90 days), annualised.] The dataset *also* publishes the **earnings-effect-stripped** companion **ern4\_30dexerniv** - "30-day constant-maturity implied volatility with earnings effect removed". The pair (**30div, 30dexerniv**) is one of the most loaded objects in the dataset. Their difference is, by construction, the share of the 30-day implied volatility that comes from the next earnings event.

## Term Structure (Monthly ATM IV)

**ern4\_m1atmiv**, **ern4\_m2atmiv**, **ern4\_m3atmiv**, **ern4\_m4atmiv** are at-the-money implied volatilities (annualised, decimal) for the first, second, third, and fourth listed expirations. The companion **ern4\_m1dtex** (days to expiration for the front month) tells you how many days are left on the m1atmiv chain. The same indexing logic carries through **m2/m3/m4**. If you need a *constant-maturity* anchor instead of a chain-position one, use **ern4\_10div** or **ern4\_30div** - they interpolate cleanly to a fixed tenor. These series are *not* constant-maturity. Each one tracks the option chain's *slot*, not a fixed horizon. When the front-month option expires, the chain shifts down by one slot. What the dataset will call m2atmiv tomorrow is the IV of what was the month-3 chain today. Because of this rolling, a short-window operator like **ts\_delta(vec\_avg(ern4\_m2atmiv), 5)** will pick up a big "drop" every month that is not a real volatility move - it is just the field starting to read a different option. Use these fields as a *snapshot* instead. Rank them across stocks on the same day, or take a ratio against a fixed-tenor anchor like **ern4\_30div**.

## Realised Volatility (With and Without Earnings)

**ern4\_10dclshv, ern4\_20dclshv, ern4\_90dclshv, ern4\_120dclshv, ern4\_1000dclshv** and their open-range tick versions (**ern4\_\*dorhv**). Every long-window historical volatility has an **xern** twin (**ern4\_500dclshvxern, ern4\_1000dorhvxern**, etc.). It is the same calculation with earnings-day returns excluded. **The gap between matched HV and HVxern is, in plain terms, how much of this stock's past realised volatility came from earnings days alone.** That is a number that fundamental-volatility datasets cannot provide you.

## Vol-Surface Shape

**ern4\_slope** is the put-call slope at the interpolated 28-day tenor. It is the cleanest single measure of where downside protection is being demanded versus upside. Note that the dataset gives you slope but not a separate skew or convexity field. The second derivative of the surface is not published here. Slope at the 28-day point is one of the more crowded fields in the dataset. It tends to need pairing with another signal to be useful.

## The Forecast Family - the Defining Strength of Earnings4

This is the family that makes earnings4 distinct from any other volatility dataset on the Platform.

* **ern4\_fcsterneffct** - "forecasted earnings effect on implied volatility (percent)". The model's forward-looking estimate of how much the next earnings release could move IV.
* **ern4\_erneffct1** - "the earn effect for earnings date number 1". The realised earnings effect on IV for the most recent past release.
* **ern4\_fairvol90d** - "the 90-day IV plus the average earnings percent move applied to the first earnings month". Read this as a fair-value reference that *adds* the historical earnings premium on top of the spot 90-day IV.
* **ern4\_fairxieevol90d** - "the 90-day IV minus the implied earnings effect plus the average earnings percent move applied to the first earnings month". In effect: strip out what the market is *currently* pricing for the next event, and substitute the long-run historical average instead.
* **ern4\_fairmth2xieevol90d** - the same construction as **fairxieevol90d** but applied to the **second** earnings month forward rather than the first.
* **ern4\_impernmv90d** - "the percent move on earnings needed to make FairXieeVol90d equal to the first earnings month implied". This is the **market-implied percent move on the next earnings release**, expressed cleanly.
* **ern4\_m1fcaststrapx**, **ern4\_m2fcaststrapx** - forecasted near-ATM straddle prices (in USD) for the month-1 (front) and month-2 listed expirations. These fields are dense and continuous (no backfill needed) but they update slowly. They revise on the model's recalibration cycle, not tick-by-tick. A short-window **ts\_delta** on a forecast field is mostly noise. The signal lives in the gap between *forecast* and *implied*.

## Option Market Microstructure & Event Timing

A handful of useful fields sit here. **ern4\_avg20doptvolu** is the 20-day average option volume - one of the few field-level proxies for "is there good interest in this option right now". **ern4\_m1dtex** gives the days to expiration of the listed options - a calendar quirk, but useful as a cycle clock. **ern4\_m1lostrike** and its siblings are the straddle/strangle strike pair the dataset prices off. They *step* whenever the model rolls to a new strike pair. That makes big single-day jumps that look like signal and are not. The event-timing pair is the most interesting:

* **ern4\_ernmnth** - the listed-expiration month number (1, 2, 3, ...) right *after* the next earnings date. It tells you, in chain-slot terms, where the earnings event sits in the option calendar.
* **ern4\_nexterntod** - **time-of-day code for the next earnings release**: 0900 = Before Market Open, 1200 = During, 1630 = After Market Close, 2359 = Unknown/Unscheduled.

# Usage Advice

* **Industry neutralization is the default.** Earnings dynamics affect stock price differently depending on the industry, so neutralising at the industry level isolates the cross-sectional signal cleanly.
* **Mind the gap between "level" fields and "step" fields.** Fields like ern4\_30div, ern4\_fcsterneffct, ern4\_slope are continuous - daily changes are real. Fields like ern4\_ernmv1, ern4\_ernmnth, ern4\_nexterntod, ern4\_m1lostrike are step-functions. They change only on set events (an earnings release, a strike roll, a calendar tick). On step fields, what ts\_delta is really catching is *the event*, not a slow move. That is often just what you want.
* **Use xern pairs.** Any time you build a signal from a volatility level, ask if the right thing to ratio it against is its own xern twin. The gap vec\_avg(ern4\_30div) - vec\_avg(ern4\_30dexerniv) is, by construction, the implied earnings effect for the front month. The same idea shows up across the dataset.
* **Backfill the sparse fields, not the dense ones.** ern4\_fcsterneffct, ern4\_erneffct1, and the per-event ernmvN series gain a lot from ts\_backfill(..., 5). The continuous IV and HV series do not need it. Backfilling a dense field is harmless but pointless.
* **Beware ts\_corr between two earnings4 fields.** Many fields are mechanically derived from each other. fairvol90d is a transform of 90div and the per-event move history. The strike pair is bound by the stock price. Correlations between such fields capture the *construction*, not any cross-asset behaviour. If you want a residual signal, prefer ts\_regression or a standardised spread.
* **The forecast family is a low-turnover, high-margin corner.** Signals built on ern4\_fcsterneffct and ern4\_erneffct1 inherit the slow update cycle of the underlying field. That holds the signal in place for longer than a typical IV (Implied Volatility) or HV (Historical Volatility) series could.
