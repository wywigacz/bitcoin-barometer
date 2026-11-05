"""Settings for the BTC forecasting system. Signal weights can be edited here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
DAILY_CSV = DATA_DIR / "daily.csv"
LOG_CSV = DATA_DIR / "forecast_log.csv"
LATEST_JSON = OUT_DIR / "latest.json"
DASHBOARD_HTML = ROOT / "docs" / "index.html"

HORIZONS = [7, 30, 90, 180]          # days
HALVINGS = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
NEXT_HALVING = "2028-04-17"          # estimate

# Model
EWMA_ALPHA = 0.03                    # volatility: EWMA of squared returns (half-life ~23 days)
SLOPE_RIDGE = 0.15                   # shrinkage of the regression slope
DRIFT_SHRINK = 0.0                   # share of BTC's historical drift (0 = we do not extrapolate past gains)
MIN_TRAIN = 300                      # min. samples needed to fit
WF_STEP = 7                          # walk-forward step (days)
WF_START = 1500                      # walk-forward start index (~4 years of history)
MIN_IC = 0.05                        # out-of-sample correlation threshold for turning on the signal "tilt"
MIN_SKILL = 0.0                      # threshold for MSE improvement vs the naive forecast
MIN_WEIGHT = 2.5                     # minimum total weight of available signals

# Weights by the "reliability levels" from the research notes (0 turns a signal off)
WEIGHTS = {
    "mvrv": 1.0,        # MVRV = price / Realized Price: the best bottom indicator
    "ma200w": 1.0,      # price vs the 200-week average
    "mayer": 0.4,       # price vs the 200-day average
    "puell": 0.5,       # miner revenue vs its yearly average
    "hashribbon": 0.4,  # miner capitulation (30d vs 60d hash rate)
    "fng": 0.3,         # Fear & Greed (read contrarian)
    "funding": 0.4,     # perpetual futures funding rate (leverage)
    "stable": 0.3,      # stablecoin supply growth (30d)
    "m2": 0.5,          # M2 growth with a ~70-day lag (if FRED is available)
    "dxy": 0.2,         # change in the dollar index (30d)
    "ndx": 0.4,         # Nasdaq momentum (90d), BTC as a risk asset
    "vix": 0.3,         # VIX
    "rates": 0.3,       # change in short-term rates (13-week bills, 180d)
    "pi": 0.2,          # Pi Cycle Top (only a warning of a top)
    "halving": 0.6,     # phase of the halving cycle
}
