"""Broad tracking of stocks and bonds: every S&P 500 company + ~110 ETFs (countries, sectors, styles, bonds, commodities).

From this we compute:
  * "under the hood" market measures (breadth, dispersion, average correlation, yield curve, credit spread, ETF ratios),
  * a network of links with BTC (what it moves with in the same weeks),
  * analogues of the link structure: when the set of BTC correlations with markets looked like today.
Raw prices (a large file) are not stored in the repository; in the cloud they are downloaded afresh at every update."""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import config as C

RAW = C.DATA_DIR / "markets.csv.gz"
FEAT = C.DATA_DIR / "market_features.csv"
SP500 = C.DATA_DIR / "sp500.csv"
UA = {"User-Agent": "Mozilla/5.0 (btc-predictor)"}

ETF = {
    # bonds
    "SHY": "US Treasuries 1-3 years", "IEI": "US Treasuries 3-7 years", "IEF": "US Treasuries 7-10 years", "TLT": "US Treasuries 20+ years",
    "TIP": "US inflation-linked bonds", "AGG": "US bonds (whole market)", "LQD": "Corporate bonds (investment grade)",
    "HYG": "Junk bonds (high risk)", "JNK": "Junk bonds (SPDR)", "EMB": "Emerging market bonds (USD)",
    "BNDX": "Non-US bonds", "BWX": "World government bonds", "MUB": "US municipal bonds", "MBB": "US mortgage bonds",
    "BKLN": "Leveraged loans", "FLOT": "Floating-rate bonds",
    # broad market and styles
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "US small caps", "DIA": "Dow Jones", "RSP": "S&P 500 equal weight", "MDY": "US mid caps",
    "IWD": "Value stocks", "IWF": "Growth stocks", "MTUM": "Momentum", "QUAL": "Quality", "USMV": "Low volatility",
    "SPHB": "High beta", "SPLV": "S&P low volatility",
    # sectors
    "XLB": "Materials (sector)", "XLC": "Communication", "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Consumer staples", "XLRE": "Real estate", "XLU": "Utilities", "XLV": "Health care", "XLY": "Consumer discretionary",
    # themes
    "SMH": "Semiconductors", "SOXX": "Semiconductors (iShares)", "IGV": "Software", "ARKK": "Innovation (ARK)", "KWEB": "Chinese internet",
    "IBB": "Biotech", "XBI": "Biotech (small)", "ITB": "Homebuilders", "KRE": "Regional banks", "KBE": "Banks",
    "XRT": "Retail", "JETS": "Airlines", "ICLN": "Clean energy", "TAN": "Solar", "URA": "Uranium",
    "LIT": "Lithium and batteries", "GDX": "Gold miners", "SIL": "Silver miners", "BLOK": "Blockchain companies", "WGMI": "Bitcoin miners",
    # countries
    "EWJ": "Japan", "EWZ": "Brazil", "EWG": "Germany", "EWU": "United Kingdom", "EWQ": "France", "EWI": "Italy", "EWP": "Spain",
    "EWC": "Canada", "EWA": "Australia", "EWY": "South Korea", "EWT": "Taiwan", "EWH": "Hong Kong", "FXI": "China (large caps)", "MCHI": "China (MSCI)",
    "INDA": "India", "EWW": "Mexico", "TUR": "Turkey", "EZA": "South Africa", "EIS": "Israel", "KSA": "Saudi Arabia", "ARGT": "Argentina",
    "VNM": "Vietnam", "EPOL": "Poland", "EWS": "Singapore", "EIDO": "Indonesia", "THD": "Thailand", "EEM": "Emerging markets", "VGK": "Europe",
    # commodities and currencies
    "GLD": "Gold", "SLV": "Silver", "USO": "Oil", "UNG": "Natural gas", "DBC": "Commodity basket", "DBA": "Agriculture", "CPER": "Copper",
    "UUP": "US dollar", "FXE": "Euro", "FXY": "Yen", "FXF": "Swiss franc",
}
RATIOS = {  # ratios that often describe the mood (risk assets vs safe havens)
    "credit": ("HYG", "IEF", "Risk appetite in bonds (junk vs Treasuries)"),
    "term": ("TLT", "SHY", "Long vs short bonds"),
    "infl": ("TIP", "IEF", "Inflation expectations (TIPS vs Treasuries)"),
    "small": ("IWM", "SPY", "Small vs large caps"),
    "growth": ("IWF", "IWD", "Growth vs value"),
    "beta": ("SPHB", "SPLV", "High beta vs low volatility"),
    "em": ("EEM", "SPY", "Emerging markets vs US"),
    "semis": ("SMH", "SPY", "Semiconductors vs the market"),
    "banks": ("KRE", "SPY", "Regional banks vs the market"),
    "homes": ("ITB", "SPY", "Homebuilders vs the market"),
    "cugold": ("CPER", "GLD", "Copper vs gold (economic cycle)"),
    "miners": ("GDX", "GLD", "Gold miners vs gold"),
    "discr": ("XLY", "XLP", "Discretionary vs staples"),
    "china": ("KWEB", "SPY", "Chinese internet vs US"),
}
LEVELS = {"breadth200": "Share of S&P 500 companies above their 200-day average", "breadth50": "Share of companies above their 50-day average",
          "dispersion": "Dispersion of company returns (30 days)", "avgcorr": "How closely stocks move together (60-day average correlation)",
          "updown": "Share of companies up over the week", "curve": "Yield curve (10 years minus 3 months)"}


INTL = {  # index: (Wikipedia page, symbol column, converter to Yahoo symbol, country)
    "DAX": ("DAX", "Ticker", lambda t: t, "Germany"), "Euro Stoxx 50": ("Euro_Stoxx_50", "Ticker", lambda t: t, "Eurozone"),
    "CAC 40": ("CAC_40", "Ticker", lambda t: t, "France"), "AEX": ("AEX_index", "Ticker", lambda t: t, "Netherlands"),
    "IBEX 35": ("IBEX_35", "Ticker", lambda t: t, "Spain"), "FTSE 100": ("FTSE_100_Index", "Ticker", lambda t: t.replace(".", "-").rstrip("-") + ".L", "United Kingdom"),
    "Hang Seng": ("Hang_Seng_Index", "Ticker", lambda t: t.split(":")[-1].strip().zfill(4) + ".HK", "Hong Kong"),
    "TSX 60": ("S%26P/TSX_60", "Symbol", lambda t: t.replace(".", "-") + ".TO", "Canada"), "Nifty 50": ("NIFTY_50", "Symbol", lambda t: t + ".NS", "India"),
    "ASX 50": ("S%26P/ASX_50", "Symbol", lambda t: t + ".AX", "Australia"), "SMI": ("Swiss_Market_Index", "Ticker", lambda t: t + ".SW", "Switzerland"),
    "KOSPI 200": ("KOSPI_200", "Symbol", lambda t: str(t).zfill(6) + ".KS", "South Korea"), "Bovespa": ("List_of_companies_listed_on_B3", "Ticker", lambda t: t + ".SA", "Brazil"),
}
INTL_CSV = C.DATA_DIR / "intl.csv"
NAME_COLS = ("Company", "Name", "Company name")


def intl_list():
    rows = []
    for idx, (page, col, fx, country) in INTL.items():
        try:
            r = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers={"User-Agent": "btc-predictor/1.0 (research; wywigacz@users.noreply.github.com)"}, timeout=40)
            tabs = [t for t in pd.read_html(io.StringIO(r.text)) if col in map(str, t.columns)]
            t = max(tabs, key=len)
            nc = next((c for c in NAME_COLS if c in t.columns), None)
            for _, row in t.iterrows():
                sym = str(row[col]).strip()
                if not sym or sym == "nan":
                    continue
                rows.append({"Symbol": fx(sym), "Name": str(row[nc]) if nc else sym, "Index": idx, "Country": country})
        except Exception:
            continue
    df = pd.DataFrame(rows).drop_duplicates("Symbol")
    if len(df) > 200:
        df.to_csv(INTL_CSV, index=False)
        return df
    return pd.read_csv(INTL_CSV) if INTL_CSV.exists() else df


def sp500_list():
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers={"User-Agent": "btc-predictor/1.0 (research; wywigacz@users.noreply.github.com)"}, timeout=40)
        t = pd.read_html(io.StringIO(r.text))[0][["Symbol", "Security", "GICS Sector"]]
        t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
        t.to_csv(SP500, index=False)
        return t
    except Exception:
        return pd.read_csv(SP500) if SP500.exists() else pd.DataFrame(columns=["Symbol", "Security", "GICS Sector"])


def _yahoo(sym, p1, p2):
    for i in range(3):
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}?period1={p1}&period2={p2}&interval=1d",
                             headers=UA, timeout=30)
            if r.status_code == 200:
                res = r.json()["chart"]["result"][0]
                adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or res["indicators"]["quote"][0]["close"]
                s = pd.Series({pd.Timestamp(int(t), unit="s").normalize(): c for t, c in zip(res["timestamp"], adj) if c is not None}, dtype=float)
                return sym, s[~s.index.duplicated(keep="last")].sort_index()
            if r.status_code == 404:
                return sym, None
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return sym, None


def update(workers=8, verbose=True):
    sp = sp500_list()
    it = intl_list()
    syms = list(dict.fromkeys(list(ETF) + sp["Symbol"].tolist() + it["Symbol"].tolist()))
    p1, p2 = 1313625600, int(time.time()) + 86400
    t0 = time.time()
    out, fails = {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, s in ex.map(lambda x: _yahoo(x, p1, p2), syms):
            if s is not None and len(s) > 200:
                out[sym] = s
            else:
                fails.append(sym)
    px = pd.DataFrame(out).sort_index()
    px = px.loc["2011-08-01":]
    C.DATA_DIR.mkdir(exist_ok=True)
    px.to_csv(RAW, float_format="%.6g", compression="gzip")
    if verbose:
        print(f"  markets: {px.shape[1]} series ({len(fails)} missing), {time.time() - t0:.0f}s", flush=True)
    return px, sp


def load():
    if not RAW.exists():
        return None
    return pd.read_csv(RAW, index_col=0, parse_dates=True, compression="gzip")


def features(px, sp):
    """"Under the hood" market measures, day by day (using only the past)."""
    stocks = [s for s in sp["Symbol"] if s in px.columns]
    S = px[stocks]
    f = pd.DataFrame(index=px.index)
    f["breadth200"] = (S > S.rolling(200, min_periods=180).mean()).where(S.rolling(200, min_periods=180).mean().notna()).mean(axis=1) * 100
    f["breadth50"] = (S > S.rolling(50, min_periods=45).mean()).where(S.rolling(50, min_periods=45).mean().notna()).mean(axis=1) * 100
    lr = np.log(S).diff()
    f["dispersion"] = np.log(S / S.shift(21)).std(axis=1)
    f["updown"] = (S / S.shift(5) > 1).where(S.shift(5).notna()).mean(axis=1) * 100
    # average correlation from the variance of an equal-weight portfolio: rho = (n^2 Var(EW) - sum var) / ((sum sd)^2 - sum var)
    w = 60
    v_i = lr.rolling(w, min_periods=40).var()
    sd_i = np.sqrt(v_i)
    ew = lr.mean(axis=1)
    n = lr.notna().sum(axis=1)
    v_ew = ew.rolling(w, min_periods=40).var()
    num = n ** 2 * v_ew - v_i.sum(axis=1)
    den = sd_i.sum(axis=1) ** 2 - v_i.sum(axis=1)
    f["avgcorr"] = (num / den).clip(-1, 1)
    for k, (a, b, _) in RATIOS.items():
        if a in px and b in px:
            f[f"ratio_{k}"] = px[a] / px[b]
    return f


def comove(df, px, sp, weeks=104):
    """What BTC moves with in the same weeks (correlation of weekly returns, last 2 years)."""
    btc = df["close"].resample("W-FRI").last()
    P = px.resample("W-FRI").last()
    r_b = np.log(btc).diff().tail(weeks)
    R = np.log(P).diff().reindex(r_b.index)
    corr = R.apply(lambda s: s.corr(r_b) if s.notna().sum() >= weeks * 0.8 else np.nan).dropna()
    names = {**ETF, **dict(zip(sp["Symbol"], sp["Security"]))}
    if INTL_CSV.exists():
        it = pd.read_csv(INTL_CSV)
        names.update({s: f"{n} ({i})" for s, n, i in zip(it["Symbol"], it["Name"], it["Index"])})
    sector = dict(zip(sp["Symbol"], sp["GICS Sector"]))
    tab = pd.DataFrame({"sym": corr.index, "r": corr.values})
    tab["name"] = tab["sym"].map(names)
    tab["kind"] = np.where(tab["sym"].isin(list(ETF)), "fund", "company")
    sec = tab[tab["kind"] == "company"].assign(sector=lambda t: t["sym"].map(sector)).groupby("sector")["r"].mean().sort_values(ascending=False)
    bonds = [k for k in ("SHY", "IEI", "IEF", "TLT", "TIP", "AGG", "LQD", "HYG", "JNK", "EMB", "BNDX", "MUB", "MBB", "BKLN") if k in corr.index]
    return {"top": tab.sort_values("r", ascending=False).head(15), "bottom": tab.sort_values("r").head(8),
            "sectors": sec, "bonds": tab[tab["sym"].isin(bonds)].sort_values("r", ascending=False), "n": int(len(tab))}


KEY = ["SPY", "QQQ", "IWM", "TLT", "IEF", "HYG", "LQD", "GLD", "UUP", "EEM", "XLE", "XLF", "XLK", "SMH", "ARKK", "USO", "DBC", "FXY",
       "TIP", "EMB", "KRE", "XLU", "XLP", "EWJ", "FXI"]


def corr_analogs(df, px, win=90, top=8, gap=90):
    """'Analogues of analogues': periods when the set of BTC correlations with 25 key markets looked like today."""
    keys = [k for k in KEY if k in px.columns]
    P = px[keys].reindex(df.index).ffill(limit=4)
    rb = np.log(df["close"]).diff()
    R = np.log(P).diff()
    mask = R.abs().sum(axis=1) > 0                            # trading days
    rb, R = rb[mask], R[mask]
    cor = pd.DataFrame({k: rb.rolling(win, min_periods=int(win * 0.8)).corr(R[k]) for k in keys}).dropna(how="any")
    if len(cor) < 500:
        return None
    cur = cor.iloc[-1].to_numpy()
    c = df["close"]
    f30, f90 = (c.shift(-30) / c - 1) * 100, (c.shift(-90) / c - 1) * 100
    X = cor.to_numpy()
    dist = np.sqrt(((X - cur) ** 2).mean(axis=1))
    order = np.argsort(dist)
    picked = []
    last_known = c.index[-91]
    for i in order:
        d = cor.index[i]
        if d > last_known:
            continue
        if all(abs((d - p).days) > gap for p in picked):
            picked.append(d)
        if len(picked) == top:
            break
    rows = [{"date": d.strftime("%Y-%m-%d"), "f30": float(f30.loc[d]), "f90": float(f90.loc[d])} for d in picked]
    now = {k: float(v) for k, v in cor.iloc[-1].items()}
    return {"rows": rows, "now": now, "med30": float(np.median([r["f30"] for r in rows])), "med90": float(np.median([r["f90"] for r in rows])),
            "up30": float(np.mean([r["f30"] > 0 for r in rows])), "n_keys": len(keys)}
