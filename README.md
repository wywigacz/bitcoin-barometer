# Bitcoin Barometer: an honest, self-testing Bitcoin forecast

**Live dashboard: [wywigacz.github.io/bitcoin-predictor](https://wywigacz.github.io/bitcoin-predictor/)**: refreshed automatically every 6 hours.

[![Update forecasts](https://github.com/wywigacz/bitcoin-predictor/actions/workflows/update.yml/badge.svg)](https://github.com/wywigacz/bitcoin-predictor/actions/workflows/update.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![No API keys](https://img.shields.io/badge/API%20keys-none-success)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[![Bitcoin Barometer dashboard](docs/preview.png)](https://wywigacz.github.io/bitcoin-predictor/)

Most Bitcoin "price predictions" give you one confident number. The Bitcoin Barometer does the opposite: it forecasts the
**whole range of likely prices** (7 to 180 days ahead), explains it in plain English like a weather report, and keeps
**testing itself in public** so you can see exactly how far to trust it.

It is a one-person research project in quantitative finance, time-series forecasting and forecast evaluation, built
with free public data only and run entirely on GitHub Actions.

## What makes it different

- **Probabilistic, not point forecasts.** Quantile bands (P10 to P90) for 7, 30, 90 and 180 days, the chance of touching
  key price levels, and the chance of a calm or stormy week.
- **Pre-registered hypotheses.** 21 ideas (ETF flows, whale wallets, leverage, options skew, Wikipedia attention, news tone...)
  were written down *before* looking at results, with Bonferroni correction and a split-half stability check.
- **A 2-year holdout exam.** Models are chosen only on data before Oct 29, 2024 and must then beat a "no change" benchmark
  on the unseen last two years (Diebold-Mariano test with HAC and Harvey-Leybourne-Newbold corrections).
- **Hundreds of data sources, zero API keys.** On-chain metrics, Deribit options, Binance leverage and 5-minute candles,
  CFTC positioning, macro and liquidity data, ~250 global market series, Wikipedia attention in 12 languages, GDELT news tone.
- **An external judge.** Our touch probabilities are scored against real-money prediction markets (Polymarket, Kalshi).
- **A virtual trader with a public journal.** Every ~6 hours it decides to buy, sell or wait with a virtual 10,000 PLN
  (+10,000 PLN a month) and writes down what it sees, what it does and when it would change its mind:
  [read the trader journal](https://wywigacz.github.io/bitcoin-predictor/journal.html).
- **Append-only forecast ledger.** Every forecast is logged with the commit hash and scored once its horizon passes.

## Key findings so far

Honest results matter more than impressive ones:

| Question | Result |
|---|---|
| Can we predict the **direction** of Bitcoin's price? | No reliable edge. Of 14 pre-registered direction hypotheses only weak momentum passed, and it adds nothing out of sample; a dedicated direction track also failed, so the dashboard says 50/50. |
| Can we predict **how volatile** the next week will be? | Yes. Volatility-event forecasts beat the "same as last time" benchmark both before and after the holdout cut-off. |
| Do volatility-aware bands help? | Yes at 7, 30 and 90 days: mean-reverting volatility scaling beat the square-root-of-time rule. |
| Who predicts price **touches** better, us or the market? | The market. Polymarket's Brier score is 0.131 vs our 0.160 over the last two years. |
| Does active allocation beat regular buying (DCA)? | Not convincingly. It wins in about half of all 2-year windows; the 90% confidence interval includes zero. |

The full research roadmap and its outcomes are in [PLAN.md](PLAN.md).

## Tech stack

Python (pandas, NumPy, SciPy, scikit-learn), GARCH / HAR / HARQ volatility models, hidden Markov models, quantile
gradient boosting, ridge regression, kNN analogues, walk-forward validation, pinball loss and Brier scores,
Monte Carlo path simulation, a static HTML dashboard with inline SVG and Chart.js, and GitHub Actions + GitHub Pages
for fully automated, serverless operation.

## Local run

```
python run.py            # incremental update + analysis + dashboard (docs/index.html)
python run.py --full     # full history download from scratch
```

Requires: pandas, numpy, scipy, scikit-learn, requests (`pip install -r requirements.txt`). No API keys.

## Architecture: what each module does

| Module | Task |
|---|---|
| `sources.py` | price, on-chain (MVRV), miners, sentiment, macro, stablecoins, funding, Deribit (DVOL, put/call), CFTC, Upbit (kimchi), Wikipedia in 11 languages |
| `world.py` | about 250 global series: exchanges and currencies of 20+ countries, commodities and energy (TTF gas, coal, gasoline), BTC mining companies, Wikipedia attention (about 130 topics: people, institutions, bubbles, climate, technologies), weather in mining regions, sunspots, earthquakes, the moon, calendar and "wild cards" (Mercury retrograde calculated from Kepler orbits, Friday the 13th, World Cup/Olympics/Super Bowl, Ramadan, Golden Week, Thanksgiving, daylight saving time) |
| `extra.py` | dollar liquidity (NY Fed: reverse repo, SOFR, EFFR, SOMA; US Treasury: TGA; net liquidity), CFTC fund positions in S&P, Nasdaq, 10-year Treasuries, yen, euro, VIX, gold and oil, DefiLlama (TVL, DEX, USDT, USDC), Lightning (mempool.space), CoinMetrics (BTC on exchanges, inflows), GFZ space weather (Kp, Ap, F10.7), Hacker News (monthly mentions, shifted by one month so as not to peek at the future) |
| `whales.py` | large players: daily flows into Bitcoin ETFs (TFTC, CC BY 4.0), Strategy transactions (Bitbo, change calculated from balance differences), Hyperliquid funding rate and position snapshots of the 150 best and 150 worst Hyperliquid traders every 6 hours (`data/hl_snapshots.csv`, recorded since Oct 29, 2026); country wallet balances from mempool.space: USA (Silk Road address), Germany (7 BKA addresses, total 49,859 BTC consistent with reports), El Salvador (official address list from bitcoin.gob.sv); transaction history in `data/entities_cache.json` |
| `touch.py` | chance of touching price levels in 7/30/90 days (based on daily highs and lows if that definition passed the test; otherwise based on closes): simulation of 20,000 paths from full historical days (close, high, low, standardized); 4 variants (volatility mean reversion, FHS with GJR-GARCH, 4-year drift, both) compared by a backtest from 2016 vs historical frequency computed only from the past (Brier score split into reliability and resolution); the best one goes to the dashboard |
| `sim.py` | virtual, actively managed portfolio (10,000 PLN + 10,000 PLN every month): daily decision on BTC allocation from 3 strategies (regular, volatility control, 28-day trend) weighted by the Hedge algorithm over the last 365 days; 5 percentage point band, 0.2% fee; append-only live journal (`data/sim_live.csv`, from Oct 28, 2026); historical test from 2018 and exam from Oct 29, 2024 vs regular buying; `sim.taxed`: result after Polish tax (19% on annual crypto income, costs on a cash basis with excess carryforward, payment on Apr 30) and with interest on cash (NBP rate from BIS minus 1 percentage point, after Belka tax; `data/nbp_rate.csv`) |
| 2-year exam | `ensemble.holdout`, `touch.calibrate`, `sim.run`: method/variant selection only on data before Oct 29, 2024, one test on the last 2 years; only the winner that passed the exam enters the forecast (beats "no change", DM p < 0.10) |
| method journal | `store.update_ledger`: from Oct 30, 2026 daily quantiles of each method for each horizon (`data/ledger.csv`), with commit number, append-only; after the deadline the result is appended (80% coverage, pinball); `forecast_log.csv` is also append-only |
| test scope | `ensemble._step`: horizons 1/3/7 days checked daily (from Oct 30, 2026), longer horizons weekly; `_dm` with the Harvey-Leybourne-Newbold correction; `_mde` = minimum detectable edge; exam "inconclusive" at < 20 independent observations; `confirm.health` = monitor and retirement rule for confirmed signals, H14 = H7 only on data after Oct 30, 2026 |
| `lev.py` | leverage from the Binance archive (data.binance.vision, no key): open interest and long/short ratios (from Sep 2020), perpetual contract premium and funding rate (from Jan 2020); `data/lev.csv`; hypotheses H15-H17 (risk of a 10% drop in 30 days) |
| `opthist.py` | Deribit options history reconstructed from the trade archive (history.deribit.com, from 2018): 25-delta skew, at-the-money volatility, put/call ratio; 3 samples of 1,000 trades per day; `data/opt_hist.csv`; feeds H13 |
| `judge.py` | judge from prediction markets: resolved monthly touch markets on Polymarket (from 2024, `data/poly_markets.csv`; when DNS is blocked via Google DNS and curl) versus our model using data from the same day; live Kalshi journal (`data/judge_live.csv`) |
| `volev.py` | "calm or stormy": chance of a stormy week (7-day volatility above the yearly median), a day with a move ≥5% within 7 days and ≥10% within 30 days; logistic regression on the volatility forecast, walk-forward weekly from 2015, compared with a constant frequency and with "like recently"; the edge is confirmed only if it beats "like recently" before and after Oct 29, 2024 |
| band scaling | `ensemble.horizon_sigma` / `scale_test`: volatility over H days mean-reverting to the 2-year norm (persistence from GJR-GARCH) instead of σ·√H; enabled per horizon only when the baseline distribution does better before Oct 29, 2024 (DM p < 0.10) and in the exam; band coverage audit across volatility terciles |
| `rv.py` | realized volatility from 5-minute Binance BTCUSDT candles (data.binance.vision, from Sep 2017): RV, semivariances, bipower, jumps; `data/rv_daily.csv`. In `ensemble.vol_models`, candidates HAR-RV and HAR with semivariance and jumps, evaluated on held-out 30% of data (QLIKE); used only if they improve the forecast there; from Oct 31, 2026 also realized quarticity RQ and `ensemble.rv_race`: HAR-RV and HARQ (forecast directly for 1/3/7 days, retrained yearly) versus the current scale, judge = pinball of final bands, accepted only at DM p < 0.10 before Oct 29, 2024 and with an edge after that date |
| `confirm.py` | separate confirmation test for hypotheses written in advance on Oct 29, 2026: batch 1 (H1-H6, large players, Bonferroni 0.05/6) and batch 2 (H7-H10, 1 to 4 week trend, attention, signal combination, 0.05/4); Spearman, n/horizon, one-sided, consistent sign in both halves; confirmed signals form the "conf" method in the ensemble (same significance gate as the rest); batch 6 (H18-H21, Oct 31, 2026) covers volatility regressors: target = log(realized 7-day variance / EWMA), Wikipedia attention jump, exchange flows relative to stock (Coin Metrics), GDELT news tone (`extra.gdelt_tone`), fee share of miner revenue |
| `trader.py` | virtual trader from Oct 31, 2026: takes over the portfolio from `sim.py` and on every cloud update (about every 6 hours) decides at the current price (Bitstamp, NBP rate): buy, sell or wait, and for how much; the same strategies and Hedge, trend from the current price, 5 percentage point band; append-only journal (`data/trader_log.csv`) with entries "what I see / what I do and why / when I will change my mind", page `docs/journal.html`, settlement of each trade after 1 and 7 days; saves locally only with `BTC_TRADER=1` |
| `dirtrack.py` | separate direction track: chance of a rise in 7 days from logistic regression on confirmed signals, retrained weekly; Brier versus a constant frequency of rises (DM with HLN), replaces 50% only at p < 0.10 before and after Oct 29, 2024 |
| `store.py` | data guard: removes impossible values (for example Coinbase price outside 0.8 to 1.25 of Bitstamp price, DVOL outside 10 to 400), after fixing mixed-up columns in Oct 2026 |
| `events.py`, `events_data.py` | event database: manual, calendar (FOMC, options), hacks (DefiLlama), automatically detected market shocks; move attribution and event study |
| `worldlog.py` | world journal: what happened each day since 2011 (Wikipedia Current events), split into categories and about 20 keyword groups (war, banks, sanctions, elections, disasters, crypto...) -> series for the scanner |
| `news.py` | news headlines about Bitcoin from days with large moves (GDELT, with cache and patient request rate) |
| `model.py` | 15 signals (MVRV, 200W, Puell, macro...) -> composite indicator |
| `regularity.py` | Hurst (DFA), permutation entropy, variance ratio, volatility clustering, Hill index, EVT (GPD), Markov chain, Kalman filter, periodogram |
| `scan.py` | scanner of hundreds of hypotheses with multiple-testing correction (FDR) + stability + out-of-sample test |
| `ensemble.py` | volatility (EWMA + GJR-GARCH + HAR), HMM, power law; models: naive, composite indicator, ridge, kNN analogs, quantile boosting, world scanner; ensemble with a significance gate |
| `report.py`, `template_*.html`, `chart.js` | HTML dashboard (one file), including an interactive regular-deposit simulator (history + scenarios; educational, no advice and no transactions) |

## Reliability rules

- **Walk-forward**: every model learns only on data known on that day; scanner feature selection happens inside walk-forward.
- **Benchmark "no change"** is calibrated the same way as the models; comparison in pinball loss, Diebold-Mariano test.
- **Significance gate**: a model enters the forecast only at p ≤ 0.10 (full weight at p ≤ 0.03). Today none pass it, so the center of the forecast = current price, and the band comes from volatility.
- **Band calibration**: ~80% of outcomes should fall within P10-P90 (checked in the test).
- Audit: random features and a shuffled target give no edge, while a deliberate future leak gives +34% (the pipeline detects a real signal).
- Forecast log (`data/forecast_log.csv`) compares each forecast with the actual price after the horizon passes.

## Limitations

- Known weaknesses: GARCH/HAR parameters and volatility mixture weights chosen on the full sample (mild leakage, affects scale, not direction); signal thresholds come from published studies.
- Missing free sources: ETF flows, Google Trends, global M2 (FRED is sometimes unavailable, then the signal is skipped).
- Halving cycles are only 4 observations.

## Local schedule (optional)

```
python run.py --install-schedule   # Windows task every 6 hours
python run.py --remove-schedule
```

## About the author

Built by **Przemysław Wywigacz** ([@wywigacz](https://github.com/wywigacz)), founder of **Life Automation**, working at
the intersection of automation, data and AI-assisted engineering. This project is a public, end-to-end example of how I
approach forecasting problems: collect broad free data, state hypotheses up front, validate out of sample, and report
negative results as clearly as positive ones.

If you find it useful, a star on the repository helps others find it. Ideas, issues and pull requests are welcome.

## Disclaimer

This is an educational research project. Nothing here is investment advice, and the trading portfolio is purely
virtual: no real money and no real orders are involved.

## License

[MIT](LICENSE) © 2026 Przemysław Wywigacz
