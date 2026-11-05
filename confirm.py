"""Confirmation test for big-player moves: a few hypotheses written down IN ADVANCE (2026-10-29), before anyone looked at the results.

Why separate from the scanner: the scanner runs ~8,500 tests, so a real but weak effect would drown in the correction for the number of tries.
Here there are only 6 tries, each with its direction predicted in advance, so the bar is gentler and the result is still honest:
  1. BTC return counted from the close of the day AFTER the day the data became public (no peeking into the future),
  2. rank correlation (Spearman), with the number of observations reduced for overlapping returns (n / horizon),
  3. one-sided test in the predicted direction, Bonferroni correction: p < 0.05 / 6,
  4. the same direction in both halves of the sample.
A hypothesis that passes everything joins the model ensemble as an extra feature. We do not change hypotheses or thresholds after the fact."""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as tdist

import config as C
import whales as WH

ALPHA = 0.05
MIN_DAYS = 250
HYP = [
    {"id": "H1", "col": "etf3", "h": 7, "sign": +1, "text": "Large money inflow into ETFs (sum of 3 trading days) → higher BTC return over the next week"},
    {"id": "H2", "col": "etfstreak", "h": 7, "sign": +1, "text": "A long run of days with ETF inflows (not outflows) → higher return over the next week"},
    {"id": "H3", "col": "mstr30", "h": 30, "sign": +1, "text": "Large purchases by Strategy in the last 30 days → higher return over the next 30 days"},
    {"id": "H4", "col": "hl_smart", "h": 7, "sign": +1, "text": "Best Hyperliquid traders positioned for a rise → higher return over the next week"},
    {"id": "H5", "col": "hl_spread", "h": 7, "sign": +1, "text": "Best traders more bullish than the worst ones → higher return over the next week"},
    {"id": "H6", "col": "hl_fund7", "h": 7, "sign": -1, "text": "High fee for betting on a rise on Hyperliquid (crowding) → lower return over the next week"},
]
for _h in HYP:
    _h["batch"] = 1
# Batch 2 (written 2026-10-29, before looking at results): the best-documented effects in Bitcoin research
# (Liu and Tsyvinski 2021: 1-4 week momentum and investor attention) plus a combination of weak signals into one.
HYP += [
    {"id": "H7", "col": "mom28", "h": 7, "sign": +1, "batch": 2, "text": "Price rise over the last 4 weeks → higher return over the next week (momentum)"},
    {"id": "H8", "col": "mom7", "h": 7, "sign": +1, "batch": 2, "text": "Price rise over the last week → higher return over the next week (short momentum)"},
    {"id": "H9", "col": "attn", "h": 7, "sign": +1, "batch": 2, "text": "Rising interest in Bitcoin on Wikipedia → higher return over the next week"},
    {"id": "H10", "col": "combo", "h": 7, "sign": +1, "batch": 2, "text": "Combination of weak signals (momentum, attention, ETF inflow) → higher return over the next week"},
]
# Batch 3 (written 2026-10-29, after the Deep Research reports, before looking at results): stablecoins and the options market.
HYP += [
    {"id": "H11", "col": "peg7", "h": 7, "sign": -1, "batch": 3, "text": "USDT priced above 1 dollar (flight to “digital cash”) → lower BTC return over the next week"},
    {"id": "H12", "col": "vrp", "h": 30, "sign": +1, "batch": 3, "text": "Options expensive relative to actual swings (volatility risk premium) → higher return over the next 30 days"},
    {"id": "H13", "col": "rr25", "h": 1, "sign": +1, "batch": 3, "text": "Options betting on a rise pricier than those betting on a fall (25-delta risk reversal) → higher return the next day"},
]


def _z(s):
    return ((s - s.expanding(365).mean()) / s.expanding(365).std()).clip(-4, 4)


def features(idx, df=None):
    """Daily features aligned to the BTC index and shifted by 1 day (the value from day D is used from day D+1)."""
    F = pd.DataFrame(index=idx)
    if df is not None:
        lc = np.log(df["close"])
        F["mom28"] = lc - lc.shift(28)
        F["mom7"] = lc - lc.shift(7)
        if "wiki_en" in df:
            v = df["wiki_en"].where(df["wiki_en"] > 0)
            F["attn"] = np.log(v.rolling(7, min_periods=5).mean() / v.rolling(90, min_periods=60).mean())
        if "dvol" in df:                                     # options-implied variance minus 30-day realized variance (both annualized)
            r2 = (lc.diff() ** 2).rolling(30, min_periods=25).mean() * 365
            F["vrp"] = (df["dvol"] / 100) ** 2 - r2
    if not WH.OUT.exists():
        return F.shift(1)
    w = pd.read_csv(WH.OUT, index_col=0, parse_dates=True)
    if "w_etf_flow" in w:
        f = w["w_etf_flow"].dropna()
        f3 = f.rolling(3).sum() / 1e6                                         # USD millions, trading days
        sgn = np.sign(f)
        streak = sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount().add(1) * sgn
        F["etf3"] = f3.reindex(idx).ffill(limit=4)
        F["etfstreak"] = streak.reindex(idx).ffill(limit=4)
    if "w_mstr_btc" in w:
        b = w["w_mstr_btc"].reindex(pd.date_range("2020-08-01", idx[-1])).fillna(0)
        F["mstr30"] = (b.rolling(30).sum() / 21e6 * 100).reindex(idx)        # % of all bitcoins
    if "w_hl_smart" in w:
        F["hl_smart"] = w["w_hl_smart"].reindex(idx)
        if "w_hl_dumb" in w:
            F["hl_spread"] = (w["w_hl_smart"] - w["w_hl_dumb"]).reindex(idx)
    if "w_hl_fund" in w:
        F["hl_fund7"] = w["w_hl_fund"].rolling(7, min_periods=5).mean().reindex(idx)
    if "w_usdt_px" in w:
        F["peg7"] = ((w["w_usdt_px"] - 1) * 100).rolling(7, min_periods=5).mean().reindex(idx)
    try:                                                         # H13: options skew rebuilt from Deribit trades (since 2018, consistent over time)
        import opthist
        O = opthist.load()
        if O is not None and O["rr25"].notna().sum() > 300:
            F["rr25"] = O["rr25"].reindex(idx)
    except Exception:
        pass
    if "rr25" not in F and "w_rr25" in w:
        F["rr25"] = w["w_rr25"].reindex(idx)
    try:                                                         # batch 6: volatility regressors
        if df is not None:
            wk = [c for c in df.columns if c.startswith("wiki_")]
            if wk:
                tot = df[wk].sum(axis=1, min_count=6).rolling(7, min_periods=5).mean()
                F["attn_jump"] = np.log(tot / tot.shift(7))
            if {"fees", "minerrev"} <= set(df.columns):
                F["feeshare7"] = (df["fees"] / df["minerrev"]).rolling(7, min_periods=5).mean()
        import extra as XT
        if XT.OUT.exists():
            ex = pd.read_csv(XT.OUT, index_col=0, parse_dates=True)
            if {"x_exin_btc", "x_exout_btc", "x_exsupply"} <= set(ex.columns):
                F["exflow_int"] = ((ex["x_exin_btc"] - ex["x_exout_btc"]).abs().rolling(7, min_periods=5).sum() / ex["x_exsupply"]).reindex(idx)
            if "x_gdelt_tone" in ex:
                F["tone7"] = ex["x_gdelt_tone"].rolling(7, min_periods=4).mean().reindex(idx)
    except Exception:
        pass
    try:                                                         # batch 5: leverage from the Binance archive
        import lev
        Lv = lev.load()
        if Lv is not None and df is not None:
            if "oi_btc" in Lv and "supply" in df:
                F["oi_sup"] = (Lv["oi_btc"].reindex(idx) / df["supply"].ffill()) * 100
            if "premium" in Lv:
                F["basis7"] = Lv["premium"].reindex(idx).rolling(7, min_periods=5).mean()
            if "top_ls_pos" in Lv:
                F["topls7"] = Lv["top_ls_pos"].reindex(idx).rolling(7, min_periods=5).mean()
    except Exception:
        pass
    parts = [_z(F[c]) for c in ("mom28", "attn", "etf3") if c in F]
    if parts:
        F["combo"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return F.shift(1)


def _spear(x, y):
    a, b = rankdata(x), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _p_hac(x, y, h, sign):
    """Robustness test: slope of the regression of return on signal rank, Newey-West standard error with window h
    (overlapping returns). For information only: the hypothesis status is decided by the rules written in advance."""
    xr = (rankdata(x) - 0.5) / len(x)
    xr = (xr - xr.mean()) / xr.std()
    yc = y - y.mean()
    b = float((xr * yc).sum() / (xr * xr).sum())
    u = (yc - b * xr) * xr
    n = len(u)
    S = (u * u).sum()
    for k in range(1, h + 1):
        S += 2 * (1 - k / (h + 1)) * (u[k:] * u[:-k]).sum()
    se = np.sqrt(S) / (xr * xr).sum()
    return float(1 - tdist.cdf(sign * b / se, n - 2)) if se > 0 else 1.0


# Batch 4 (written 2026-10-30): H14 = H7 tested ONLY on data after the registration date. The H7 result comes from the history
# we picked it on; the only clean evidence is the future. Verdict only at n/h >= 250 (that is years).
HYP += [
    {"id": "H14", "col": "mom28", "h": 7, "sign": +1, "batch": 4, "from": "2026-10-30", "min_n": 250 * 7,
     "text": "H7 continued on new data (from Oct 30, 2026): rise over 4 weeks → higher return over the next week"},
]

# Batch 5 (written 2026-10-30, before looking at the data): leverage from the Binance archive (since 2020) as crash risk, not direction.
# Target: whether within 30 days the price drops at least once (daily low) by 10% or more (in line with “Crypto carry”, BIS 2023).
HYP += [
    {"id": "H15", "col": "oi_sup", "h": 30, "sign": +1, "batch": 5, "target": "drop10",
     "text": "Many open futures positions (relative to BTC supply) → more frequent 10% drop within 30 days"},
    {"id": "H16", "col": "basis7", "h": 30, "sign": +1, "batch": 5, "target": "drop10",
     "text": "High premium of perpetual futures over the price (crowd chasing leverage) → more frequent 10% drop within 30 days"},
    {"id": "H17", "col": "topls7", "h": 30, "sign": +1, "batch": 5, "target": "drop10",
     "text": "Top traders heavily positioned for a rise (long/short ratio) → more frequent 10% drop within 30 days"},
]

# Batch 6 (written 2026-10-31, before looking at results): VOLATILITY regressors, not direction. Target: whether over the next 7 days
# swings will be bigger than past volatility alone predicts (log of the ratio: 7-day realized variance / forecast).
# A confirmed regressor would be a candidate for the volatility model (where it would still have to improve QLIKE on held-out data).
HYP += [
    {"id": "H18", "col": "attn_jump", "h": 7, "sign": +1, "batch": 6, "target": "volratio",
     "text": "Sudden jump in interest in Bitcoin on Wikipedia (12 languages) → bigger swings over the next week than the model predicts"},
    {"id": "H19", "col": "exflow_int", "h": 7, "sign": +1, "batch": 6, "target": "volratio",
     "text": "Large flows into and out of exchanges (relative to exchange reserves) → bigger swings over the next week"},
    {"id": "H20", "col": "tone7", "h": 7, "sign": -1, "batch": 6, "target": "volratio",
     "text": "More negative tone of Bitcoin news (GDELT) → bigger swings over the next week"},
    {"id": "H21", "col": "feeshare7", "h": 7, "sign": +1, "batch": 6, "target": "volratio",
     "text": "High share of fees in miner revenue (congested network) → bigger swings over the next week"},
]

# Retirement rule (written 2026-10-30): a confirmed signal stops being used when, at two consecutive yearly reviews
# (Oct 29, from 2027), over the last 3 years both p (Newey-West, one-sided) > 0.10 and the correlation in the predicted direction < 0.03.
RETIRE_FIRST = pd.Timestamp("2027-10-29")


def _window_stats(x, y, end, h, sign, years=3):
    m = x.notna() & y.notna() & (x.index > end - pd.DateOffset(years=years)) & (x.index <= end - pd.Timedelta(days=h))
    if m.sum() < 200:
        return None
    xv, yv = x[m].to_numpy(float), y[m].to_numpy(float)
    return {"rho": _spear(xv, yv) * sign, "p_hac": _p_hac(xv, yv, h, sign), "n": int(m.sum())}


def health(x, y, h, sign, today):
    """Health of a confirmed signal: the result over the last 3 years and the yearly reviews under the retirement rule."""
    now = _window_stats(x, y, today, h, sign)
    checks, fails, retired = [], 0, False
    c = RETIRE_FIRST
    while c <= today:
        s = _window_stats(x, y, c, h, sign)
        bad = s is not None and s["p_hac"] > 0.10 and s["rho"] < 0.03
        fails = fails + 1 if bad else 0
        retired = retired or fails >= 2
        checks.append({"date": str(c.date()), **(s or {}), "fail": bool(bad)})
        c = c + pd.DateOffset(years=1)
    return {"now": now, "checks": checks, "retired": retired, "next_check": str(c.date())}


def run(df):
    F = features(df.index, df)
    lr = np.log(df["close"])
    today = df.index[-1]
    rows = []
    for H in HYP:
        k = sum(1 for x in HYP if x["batch"] == H["batch"])            # correction for the number of tries within the batch
        r = dict(H)
        r["alpha"] = ALPHA / k
        x = F[H["col"]] if H["col"] in F else pd.Series(np.nan, index=df.index)
        if H.get("from"):
            x = x.where(x.index >= pd.Timestamp(H["from"]))
        if H.get("target") == "volratio":                       # realized h-day variance / forecast from the past (EWMA)
            r2 = lr.diff() ** 2
            ew = r2.ewm(alpha=C.EWMA_ALPHA, adjust=False, min_periods=30).mean()
            y = np.log(r2[::-1].rolling(H["h"]).mean()[::-1].shift(-1) / ew)
        elif H.get("target") == "drop10":                       # event: lowest price within h days ≤ 90% of today's close
            low = np.log(df["low"] if "low" in df else df["close"])
            fmin = low[::-1].rolling(H["h"]).min()[::-1].shift(-1)
            y = (fmin - lr <= np.log(0.9)).astype(float).where(fmin.notna())
        else:
            y = lr.shift(-H["h"]) - lr
        m = x.notna() & y.notna()
        n = int(m.sum())
        n_eff0 = max(n / H["h"], 5)
        r.update({"n": n, "now": None if x.dropna().empty else float(x.dropna().iloc[-1]),
                  "first": None if x.dropna().empty else x.dropna().index[0].strftime("%Y-%m-%d"),
                  # smallest correlation the test would detect 4 times out of 5 with the current number of independent observations
                  "mde_rho": float((norm.ppf(1 - ALPHA / k) + norm.ppf(0.8)) / np.sqrt(max(n_eff0 if n >= MIN_DAYS else 250, 5)))})
        if n < H.get("min_n", MIN_DAYS):
            r.update({"status": "collect", "r": None, "p": None})
            rows.append(r)
            continue
        xv, yv = x[m].to_numpy(float), y[m].to_numpy(float)
        rho = _spear(xv, yv)
        n_eff = max(n / H["h"], 5)
        tt = rho * np.sqrt((n_eff - 2) / max(1 - rho * rho, 1e-9))
        p = float(1 - tdist.cdf(H["sign"] * tt, n_eff - 2))
        half = n // 2
        ra, rb = _spear(xv[:half], yv[:half]), _spear(xv[half:], yv[half:])
        stable = H["sign"] * ra > 0 and H["sign"] * rb > 0
        q = pd.qcut(pd.Series(xv).rank(method="first"), 3, labels=False)
        ret = pd.Series(yv) * 100 if H.get("target") == "drop10" else (np.exp(pd.Series(yv)) - 1) * 100   # for events: frequency in %
        ok = p < ALPHA / k and stable
        r.update({"r": rho, "p": p, "r_a": ra, "r_b": rb, "stable": bool(stable), "n_eff": n_eff, "p_hac": _p_hac(xv, yv, H["h"], H["sign"]),
                  # for volatility use the median: single blow-ups distort the mean of variance ratios
                  "ret_lo": float((ret[q == 0].median() if H.get("target") == "volratio" else ret[q == 0].mean())),
                  "ret_hi": float((ret[q == 2].median() if H.get("target") == "volratio" else ret[q == 2].mean())),
                  "status": "confirmed" if ok else ("reverse" if H["sign"] * rho < 0 and (1 - p) < ALPHA / k else "rejected")})
        if r["status"] == "confirmed":
            r["health"] = health(x, y, H["h"], H["sign"], today)
            if r["health"]["retired"]:
                r["status"] = "retired"
        rows.append(r)
    # only return signals feed the direction; confirmed volatility and event regressors are kept separately
    conf = list(dict.fromkeys(r["col"] for r in rows if r["status"] == "confirmed" and not r.get("target")))
    vconf = list(dict.fromkeys(r["col"] for r in rows if r["status"] == "confirmed" and r.get("target") == "volratio"))
    return {"rows": rows, "confirmed": conf, "vol_confirmed": vconf, "features": F, "alpha_each": ALPHA / sum(1 for x in HYP if x["batch"] == 1)}
