"""Realized volatility from 5-minute data (Binance BTCUSDT, data.binance.vision archive, since September 2017).

For each day (UTC) we compute from 288 five-minute returns:
  rv     - realized variance (sum of squared returns),
  rs_neg - downside semivariance (falls only), rs_pos - upside semivariance (rises only),
  bv     - bipower variance (robust to jumps), jump = max(rv - bv, 0) - the part of the variance from sudden jumps.
Output: data/rv_daily.csv (small daily data). Raw archives are not saved; we only fetch the missing days."""
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

import config as C

OUT = C.DATA_DIR / "rv_daily.csv"
BASE = "https://data.binance.vision/data/spot"
START = pd.Timestamp("2017-09-01")
MIN_BARS = 250


def _klines(url):
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4])
    ts = df[0].astype("int64")
    ts = np.where(ts > 1e14, ts // 1000, ts)                 # since 2025 Binance gives microseconds
    return pd.Series(df[4].astype(float).to_numpy(), index=pd.to_datetime(ts, unit="ms"))


def _daily(px):
    lr = np.log(px).diff().dropna()
    lr = lr[np.isfinite(lr)]
    g = lr.groupby(lr.index.normalize())
    out = pd.DataFrame({
        "rv": g.apply(lambda x: float((x ** 2).sum())),
        "rs_neg": g.apply(lambda x: float((x[x < 0] ** 2).sum())),
        "rs_pos": g.apply(lambda x: float((x[x > 0] ** 2).sum())),
        "bv": g.apply(lambda x: float(np.pi / 2 * (x.abs() * x.abs().shift(1)).sum())),
        "rq": g.apply(lambda x: float(len(x) / 3 * (x ** 4).sum())),    # realized quarticity (for HARQ)
        "bars": g.size(),
    })
    out["jump"] = (out["rv"] - out["bv"]).clip(lower=0)
    return out[out["bars"] >= MIN_BARS]


def update(verbose=True):
    old = pd.read_csv(OUT, index_col=0, parse_dates=True) if OUT.exists() else pd.DataFrame()
    have = set(old.index[old["rq"].notna()].normalize()) if len(old) and "rq" in old else set()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    parts, t0 = [], time.time()
    this_month = today.replace(day=1)
    for m in pd.date_range(START, this_month - pd.offsets.MonthBegin(1), freq="MS"):
        days = pd.date_range(m, m + pd.offsets.MonthEnd(0))
        if sum(d in have for d in days) >= len(days) - 3:   # month already computed (single days can be empty because of exchange outages)
            continue
        px = _klines(f"{BASE}/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{m:%Y-%m}.zip")
        if px is not None:
            parts.append(_daily(px))
        time.sleep(0.2)
    # current month: daily files (the archive publishes them with about a one-day delay)
    for d in pd.date_range(this_month, today - pd.Timedelta(days=1)):
        if d in have:
            continue
        px = _klines(f"{BASE}/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{d:%Y-%m-%d}.zip")
        if px is not None:
            parts.append(_daily(px))
        time.sleep(0.1)
    new = pd.concat(parts) if parts else pd.DataFrame()
    merged = new.combine_first(old) if len(old) else new
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.index.name = "date"
    if len(merged):
        merged.to_csv(OUT, float_format="%.6g")
    if verbose:
        print(f"  [OK   ] Binance 5 min (realized volatility): {len(new)} new days, {len(merged)} in total, {time.time() - t0:.0f}s", flush=True)
    return merged


def load():
    return pd.read_csv(OUT, index_col=0, parse_dates=True) if OUT.exists() else None
