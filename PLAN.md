# Plan: What Else Can Be Done to Make the Barometer Predict Better

Written on Oct 30, 2026. Here is how it came about: eight independent "perspectives" (leverage, options, on-chain data, attention and sentiment, prediction markets, direction methods, risk methods, portfolio, and an audit of our code) submitted 52 ideas. Each group was evaluated by a separate skeptic with access to research (evidence, chances of passing our protocol, biggest risks). The decisive numbers were recalculated from scratch on our own data, and the data sources were verified by actually downloading them. The ideas that could not be confirmed are listed at the end, along with the reasons.

## The Direct Answer

**We will not improve price direction with more data.** This is not a conclusion from a single test, but from three independent rounds (a scanner of 8,500 tests, 13 pre-registered hypotheses, and now 52 ideas reviewed by skeptics). The only thing supported by research is a weak trend (our H7), and its strength is declining year by year. On top of that, we uncovered an important fact about our own setup: our threshold for admitting a method into the forecast (a 2% improvement in pinball loss) mathematically requires a predictable shift of 0.2 standard deviations, which means an R² of around 4%. A signal the size of H7 (0.13 standard deviations) provides only 0.84%. This means that a "50% chance of an increase" is partly a feature of our evaluation metric, not just the market itself. A separate track for direction (point 11) could show an accuracy of 52% to 54%, but proving this out of sample will take years (the statistical power of the test over the held-out 2 years is around 15%). Realistically, 50% will stay on the dashboard for a long time, and that is honest.

**Range, risk, and probabilities can be improved, and cheaply.** Volatility is the only thing this system genuinely predicts (a 0.50 correlation between our forecast and what actually happened a week later). Three things were confirmed three times today (by the agent, the skeptic, and myself): the 90 to 180-day bands are too narrow during calm periods; "touching a level" is calculated using daily closes, even though the dashboard promises "at least once along the way"; and we are missing a "calm or stormy week" product, which has measurable predictive power.

**The portfolio needs a more honest description before we add anything to it.** The active portfolio beats regular dollar-cost averaging (DCA) in only 48% of two-year windows since 2014 (median 0.994). A shallower drawdown in 97% of windows (by an average of 7.9 percentage points) is mainly due to holding less Bitcoin: a fixed 65% allocation gives a similar drawdown. The "+47% edge over DCA since 2018" depends heavily on the choice of the trend window (28 days: 1.47; 112 days: 0.64). The "risk" strategy on its own loses to DCA (0.78).

**Two new multi-year historical datasets that we did not have are available for free:** Binance positioning data since September 2020 (at 5-minute intervals) and Deribit options trades since 2018. They are worth adding, but as crash risk signals and for calibrating bands, not as directional forecasts; for direction, we expect them to be rejected.

## What I Fixed Today (Bugs Found by the Audit)

1. **The CME leveraged funds positioning column (`cot_lev`)** contained Italian Wikipedia page view numbers in 88% of its rows (a leftover from column mixing on Oct 29). It was rebuilt from a fresh CFTC download, dated to the publication day (Friday, 3 days after Tuesday's cutoff), with a range guard.
2. **Unfinished daily candle.** Bitstamp was also returning the current, unfinished day, so live forecasts and the journal were calculated using an intraday price (the "7-day" horizon was actually around 7.6 days). Now we only fetch closed UTC days, and the dashboard labels the price with the closing date.

## Implementation Status

- Oct 30, 2026: items 1 to 3 implemented (volatility regime-dependent bands for 7/30/90 days, "Calm or stormy?", touches based on daily highs and lows).
- Oct 30, 2026: items 7 to 9 implemented (Binance leverage and hypotheses H15 to H17, Polymarket and Kalshi benchmark, Deribit options history since 2018 for H13). Benchmark result: the prediction market predicts touches better than our model (Brier score of 0.131 vs 0.160 over the past 2 years).
- Oct 31, 2026: items 10 to 13 implemented. The HAR-RV/HARQ race on 1/3/7-day bands: none passed (edge <= 0.4%, p > 0.18). Batch 6 (volatility): H18 attention spike and H21 fee share confirmed (weakly, correlation 0.11 and 0.15), H19 exchange flows work in reverse, H20 news tone rejected. The direction track did not pass (Brier score edge -0.07% before, +0.33% after, p-values 0.55 and 0.36), so 50% remains. Taxes: the active portfolio's advantage over dollar-cost averaging since 2018 is 1.13 after taxes and interest (1.12 before).
- Oct 30, 2026: items 4 to 6 implemented (daily step for 1/3/7 days, HLN correction, "inconclusive" verdict, minimum detectable effect, H7/H10 retirement monitor and rule, H14; portfolio robustness block; log of all methods).

## Plan in Priority Order (Gain × Chance of Passing Protocol) / Effort

Effort: 1 = hours, 3 = one to two days, 5 = one week. "Chance" is the skeptic's assessment of whether the idea will pass our protocol (pre-registered hypothesis, out-of-sample test, 2-year evaluation).

| # | What | Gain | Effort | Chance |
|---|---|---|---|---|
| 1 | Volatility regime-dependent bands (H-day mean-reverting variance) + calibration audit across regimes | range | 2 | 0.45 to 0.6 |
| 2 | "Calm or stormy week", probability of a day with a move >= 5% / >= 10%, drawdown probability | new, testable product | 2 | 0.7 |
| 3 | Touch calculated using daily highs and lows, with sampling correction | honesty + risk | 2 | 0.55 |
| 4 | Appropriately sized statistics: daily steps for 1 to 7 days, small-sample correction, "inconclusive" verdict, minimum detectable effect at every gate | reliability | 2 | 0.8 |
| 5 | Portfolio honesty: all start dates, confidence interval, "blind" portfolio with identical exposure, sigma without lookahead bias, fragility table, 1/N alongside Hedge | reliability | 2 to 3 | 0.5 to 0.75 |
| 6 | Future logging: quantiles of every method daily, append-only, with code version commit hash | reliability | 1 | 0.6 |
| 7 | Leverage history from Binance (since 2020-09) as a tail-risk signal; archive of funding rates instead of cloud-blocked Bybit | risk | 3 | 0.2 to 0.3 |
| 8 | External benchmark for touches: resolved Polymarket prediction markets (since 2024-02) and Kalshi log | reliability | 2 to 3 | 0.4 |
| 9 | Deribit options history (since 2018): 25-delta skew, term structure; H13 testable now, not in 2027 | risk (options) | 4 | 0.15 (H13), 0.3 (options-implied distribution) |
| 10 | Honest rerun of the volatility race using 5-minute data (HARQ, proper benchmark) | range (1 to 7 days) | 3 | 0.4 |
| 11 | Joint test batch of "variance regressors": attention, exchange outflows, news sentiment (GDELT), fee share | range | 2 | 0.3 |
| 12 | Separate track for direction: probability of an increase calculated directly and scored with Brier score | direction | 2 to 3 | 0.15 |
| 13 | Taxes (19%, annual settlement) and interest on cash in portfolio, minimal version | portfolio reliability | 2 | 0.45 |

### 1. Volatility Regime-Dependent Bands

Today, the H-day band is simply daily volatility × √H. During calm periods, volatility typically reverts to the mean, so the 90 to 180-day band is too narrow; during turbulent periods, it is too wide. My figures (since 2015, 80% band coverage in the calm / medium / stormy tertile): 30 days: 67% / 77% / 86%; 90 days: 52% / 74% / 85%; 180 days: 48% / 73% / 82%. H-day mean-reverting variance (mean-reversion parameter from our GJR-GARCH, half-life of 23 days) evens out the spread to 1.21 / 1.25 / 1.26 over 180 days.

Skeptic's caveats that we will take into account: calculate the "baseline" level from a rolling 2 to 3-year window (Bitcoin's volatility has been declining for a decade; during the 2024 to 2026 evaluation period the effect is already small); primary horizons are 7 and 30 days, while 90 and 180 days are descriptive only; implement either this method or regime-dependent band multipliers (not both, as they overlap). First, conduct an audit: a "where the bands fail" table (by volatility regime, HMM regime, year), as this is risk-free diagnostics.

Pre-registered test: "standardization using mean-reverting variance reduces the pinball loss of the baseline distribution at 7 and 30 days (DM, HAC) and brings 80% coverage to within ±5 percentage points in each tertile, both in the pseudo-live test and in the out-of-sample evaluation".

### 2. Calm or Stormy Week

We publish alongside the honest 50/50: (a) the probability that volatility next week will be above the 1-year median; (b) the probability of at least one day with a move of >= 5% and >= 10% within 7 and 30 days (derived from already simulated paths); (c) the probability of a 10% or 20% drawdown in 30 days. Replications: 0.50 correlation; Brier score better by 6.5% compared to climatology and by 3.6% to 5.1% compared to "it will be like last week" (persistence), including in the evaluation period. The benchmark on the dashboard will be this tougher version (persistence).

### 3. Touch Based on Daily Highs and Lows

The dashboard asks "will it touch the level at least once along the way", but we calculate it using daily closes. The difference: +10% in 30 days happens in 57.9% of windows based on daily highs, compared to 50.7% based on closes; -10% in 7 days: 18.6% versus 11.7%. The fix: jointly sample triplets (close, high, low) or use a Brownian bridge (Broadie, Glasserman, Kou 1997), with protection against single-exchange wick anomalies (cross-checking with 5-minute Binance data). If the new model does not maintain its predictive accuracy in the evaluation period, we stay with daily closes and reword the dashboard.

### 4. Right-Sized Statistics

Tests for 1 and 3 days use 579 out of about 4,050 available days (weekly stepping for all horizons). The evaluation for 90 and 180 days is based on 91/79 weekly points, which amounts to roughly 7 and 3 independent observations, yet it displays "failed" as a certainty. Changes: daily step for 1/3/7 days (refit weekly, evaluate daily, HAC standard errors with lag H+1), Harvey-Leybourne-Newbold correction for small samples, an "inconclusive" verdict when the effective sample size is under 20, and at every gate the sentence "the smallest effect this test could detect", with a clear rule: a test too weak to detect an effect still means "we do not use it". In addition, a dated retirement rule for H7/H10 (decay of the effect across two consecutive annual reviews).

### 5. Portfolio Honesty

Robustness block in `sim.py` and on the dashboard: performance across every start date since 2014 (2 and 4-year windows), confidence intervals from block bootstrapping, sigma computed strictly out of sample (currently volatility parameters are fit on the full history, including the evaluation period), a fragility table for the trend window, 1/N alongside Hedge (they are virtually identical: 1.15 vs 1.14), a monitor for H7 decay, and pre-registration of H14 = continuation of H7 on data after Oct 29, 2026. Most importantly: comparison against a "blind" portfolio with the same average BTC exposure, as that is the honest benchmark for drawdowns. The dashboard will state plainly: the active portfolio means lower exposure during bad times, not greater wealth.

### 6. Future Journal Logging

Every day, we append the quantiles of every method (not just the published ones), with the commit hash from GitHub Actions; nothing is overwritten. Every day without this is evidence lost forever. Establish the admission rule for a method to enter the forecast with a proper threshold (the skeptic showed that a 6% to 14% interval for tails with 52 rows would reject a valid model 63% of the time).

### 7. Leverage History from Binance

The `data.binance.vision` archive (verified): open interest, top trader and global long/short ratios, buyer vs seller volume ratio, every 5 minutes since Sep 1, 2020 (around 2,200 small files; we keep only daily aggregates); perpetual contract premium since Jan 1, 2020; funding rates since Jan 1, 2020 (replacing Bybit, which is blocked in cloud environments, and OKX with only 36 days of history). There is no historical liquidation data.

Pre-registered hypothesis (risk, not direction): "open interest / BTC supply in the top decile (known at D+2) -> more frequent touches of -10% within 30 days and more breaches below the P10 band" (one-sided, block bootstrap, not a binomial test because episodes cluster). Risks: Binance market share is shrinking, and since 2024 a portion of large positions represents hedged ETF basis arbitrage, so "high open interest" means something different today than in 2021. Liquidation cascades from 5-minute drops in open interest used only as event labels (practitioner data points toward price continuation rather than a bounce). Basis/carry (CME via Yahoo + Binance premium) later, also as a tail-risk test; Deribit before 2021 consisted only of thin quarterly futures.

### 8. External Benchmark for Touches

Polymarket has resolved monthly "will BTC touch X" markets since February 2024 with daily price histories (settled on Coinbase 1-minute candles). This is the only immediate, independent check for the only model that has demonstrated edge. From this local machine the address does not resolve in DNS (local filtering), but from GitHub's servers it will likely work; this needs verification. With around 33 contract-months, this serves as an honesty report ("the market says 19%, we say 23%") rather than a strict pass/fail test. Kalshi (31 resolved monthly markets since August 2026, no API key required) should be logged every 6 hours and displayed alongside our own figure, with a disclaimer regarding the bid-ask spread.

### 9. Deribit Options History

Options trade archive (verified: since January 2018 with implied volatility for every trade, dense since 2019/2020; 550 expired futures contracts since 2016). A one-time download requires thousands of requests and several hours locally. From this: daily 25-delta skew, term structure, net demand. This allows H13 (already registered) to be resolved across roughly 2,400 days instead of waiting until 2027. Honest expectation: rejection (the skeptic confirmed that the cited paper on Bitcoin investigates the effect of order flow on option prices, not underlying spot price forecasting; an SMU 2026 PhD dissertation finds an effect that weakens after 2024). Only after that should the option-implied distribution (Breeden-Litzenberger) be tested as a candidate for 30-day bands and touches, evaluated against the current DVOL mixture, not against "no change".

### 10. Volatility Race from 5-Minute Data, Repeated Fairly

Our previous test evaluated models against a noisy target (daily squared return r²). When the benchmark is actual 5-minute realized volatility, HAR-RV wins by 8% to 9% (QLIKE 0.273 vs 0.296), and the 30/70 blend by 9%. However, 1-day band coverage barely changes (0.840 vs 0.832). Therefore, the acceptance criterion must be the pinball loss of the finalized 1/3/7-day bands, not QLIKE. HARQ variant (with realized quarticity), expanding refits annually, with the year 2017 excluded.

### 11. Variance Regressors, One Joint Series

Candidates (all without API keys): Wikipedia attention spikes (we have this), exchange outflows and turnover relative to reserves (Coin Metrics `FlowOutExNtv`, currently unused), Bitcoin news sentiment from GDELT (verified: daily history since 2017; the article count endpoint failed to respond today), fee share in miner revenues (we have the columns). Each tested as a variance hypothesis, accepted only if it improves QLIKE on out-of-sample data. Expected gain is small: public attention follows price moves that the volatility model already captures.

### 12. A Separate Track for Direction

Probability of an upward move calculated directly (probit regression on signals that passed screening, with isotonic calibration), evaluated via Brier score against 0.5 and a sign test with HAC standard errors, run daily rather than weekly. Only to be done after item 4. Honest expectation: 52% to 54% hit rate, test power on the 2-year holdout around 15%, meaning an "inconclusive" verdict for years; 50% stays on the dashboard, accompanied by a note showing actual live accuracy. The trend family (position relative to the 200-day moving average) as one unified primary hypothesis, as it is the exact same underlying factor as H7.

### 13. Taxes and Interest in the Portfolio

Minimal version, only after item 5: 19% tax on annual net income from virtual currencies (costs accounted on a cash basis, excess costs carried forward to the next tax year, pursuant to Polish PIT law Art. 30b(1a); not FIFO), cash interest at the NBP base rate (published by BIS without an API key) minus the 19% capital gains tax (Belka tax). Display alongside pre-tax performance. Preliminary calculation by the agent: the 1.14 edge over DCA drops to roughly 1.10 to 1.115.

## Record Starting Today (No Historical Data Available, so Testing Will Be Possible in One to Two Years)

- Raw rows of the full Deribit options volatility smile on every update (we already fetch the summary, we just need to persist it): evaluation will take years.
- Kalshi: monthly "touch / no touch" markets (item 8).
- Short-term and long-term holder supply (BGeometrics, no API key required, but only 4 years of history, rolling window causes older data to be lost; personal-use-only license, so no publishing on the dashboard without clarification).
- Strategy premium to net asset value of held bitcoins (mNAV) from data we already have; the GBTC arbitrage leg has been obsolete since January 2024.
- Effective dates and public comment deadlines for crypto regulations from the Federal Register (no API key required): not for forecasting, but for the upcoming events timeline.

## Not Worth Doing (Tested or Evaluated, with Reasons)

- Bitfinex margin funding positions since 2016: lack of empirical evidence; the exchange lost market relevance after 2019, making the rule "same sign across both halves of the history" nearly impossible to satisfy.
- App store rankings (Coinbase in the App Store), StockTwits, CoinGecko sentiment votes, GitHub stars: same "attention and sentiment" family that failed in our tests; peak events occur once per cycle, requiring a decade to gather sufficient proof.
- Google Trends stitched from 90-day windows since 2011: stitching compounds sampling error; academic literature confirms that attention lags price.
- Miner flows, coin-days destroyed, HODL waves: chart lore with no out-of-sample evidence; a 4-year data window yields only 1 to 3 independent observations.
- CPI/NFP release days as a volatility multiplier: the effect on Bitcoin is minor and regime-dependent (NY Fed 2023: Bitcoin does not react systematically to macroeconomic announcements); requires a FRED API key.
- Mempool congestion as a volatility regressor: since 2023, transaction fees have been driven by Ordinals and Runes independently of price action.
- A fourth portfolio strategy based on implied volatility (DVOL): sigma already incorporates 30% DVOL; in cryptocurrencies, the leverage effect is inverted (Baur and Dimpfl 2018); 3.6 years of data cannot distinguish it from the current setup.
- Regime-dependent momentum + meta-labeling: requires a correlation difference larger than the entire momentum effect itself.
- Order flow and liquidations as a directional signal: predictive value decays in seconds and minutes; as for bounces after liquidation cascades, empirical data from practitioners directly contradicts this hypothesis.
- FOMC statements and SEC filings as signals: no predictive power; Federal Register useful only as a calendar.
- On-chain data revision log: currently, no live signal depends on on-chain data; revisit if one is adopted (and then log a hash of the entire column, because Coin Metrics rewrites full history whenever new exchange wallet clusters are discovered).

## The Biggest Bet

Items 1 to 3 combined: bands that adjust to the market regime, touch probabilities calculated the way the dashboard promises, and "calm or stormy week". These are the only areas where three independent calculations showed the same measurable improvement today, and all three are inexpensive to build.

## Key References Checked Today

- Diebold, Hickman, Inoue, Schuermann (1998), Converting 1-day volatility to h-day volatility: scaling by √h is worse than you think, Risk 11.
- Broadie, Glasserman, Kou (1997), A continuity correction for discrete barrier options, Mathematical Finance 7(4).
- Christoffersen (1998), Evaluating interval forecasts, International Economic Review 39(4).
- Harvey, Leybourne, Newbold (1997), Testing the equality of prediction mean squared errors, International Journal of Forecasting 13.
- Patton (2011), Volatility forecast comparison using imperfect volatility proxies, Journal of Econometrics 160; Bollerslev, Patton, Quaedvlieg (2016), Exploiting the errors, Journal of Econometrics 192.
- Schmeling, Schrimpf, Todorov (2023), Crypto carry, BIS Working Paper 1087: high futures basis predicts crashes, not lower average returns.
- Cederburg, O'Doherty, Wang, Yan (2020), On the performance of volatility-managed portfolios, JFE 138: in real time, volatility management underperforms in 72 out of 103 strategies.
- Baur, Dimpfl (2018), Asymmetric volatility in cryptocurrencies, Economics Letters 173.
- Benigno, Rosa (2023), The Bitcoin-Macro Disconnect, NY Fed Staff Report 1052.
- Page, Clemen (2013), Do prediction markets produce well-calibrated probability forecasts?, Economic Journal 123.
- Hoang, Baur (2020), Forecasting bitcoin volatility: evidence from the options market, Journal of Futures Markets.
- Data sources verified via live download: data.binance.vision (metrics since 2020-09-01, premiumIndexKlines and fundingRate since 2020-01), history.deribit.com (options trades since 2018-01, 550 expired contracts), api.elections.kalshi.com (KXBTCMAXMON/MINMON), api.gdeltproject.org (timelinetone since 2017), api.bitcoin-data.com (4-year window), api.blockchain.info (mempool since 2016-06), publicreporting.cftc.gov.
