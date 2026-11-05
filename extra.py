"""Extra signals: dollar liquidity, big players' positions (CFTC), the crypto ecosystem, tech attention (Hacker News),
space weather. Each source is independent: if one fails, the rest keep going. Output: data/extra.csv (daily)."""
import io
import json
import time

import numpy as np
import pandas as pd
import requests

import config as C

OUT = C.DATA_DIR / "extra.csv"
HN_CACHE = C.DATA_DIR / "hn_cache.json"
UA = {"User-Agent": "btc-predictor/1.0 (research; wywigacz@users.noreply.github.com)"}

# name -> (label, kind of transform: attn = log level of attention/size, lvl = level/rate, chg = percentage changes of a balance)
KIND = {
    "x_rrp": ("Liquidity: bank cash parked at the Fed (reverse repo)", "chg"),
    "x_tga": ("Liquidity: US government account balance at the Fed (TGA)", "chg"),
    "x_soma": ("Liquidity: the Fed's bond portfolio (SOMA)", "chg"),
    "x_netliq": ("Net dollar liquidity (Fed - government account - reverse repo)", "chg"),
    "x_sofr": ("SOFR overnight rate", "lvl"), "x_effr": ("EFFR overnight rate", "lvl"), "x_sofr_effr": ("Money market stress (SOFR minus EFFR)", "lvl"),
    "x_tvl": ("Crypto: total value locked in DeFi", "attn"), "x_dex": ("Crypto: decentralized exchange volume", "attn"),
    "x_usdt": ("Crypto: USDT (Tether) supply", "chg"), "x_usdc": ("Crypto: USDC supply", "chg"),
    "x_ln_cap": ("Lightning Network: channel capacity", "chg"), "x_ln_nodes": ("Lightning Network: number of nodes", "attn"),
    "x_txcnt": ("BTC network: number of transactions (CoinMetrics)", "attn"), "x_iss": ("BTC network: value of newly mined coins", "attn"),
    "x_exsupply": ("BTC network: bitcoins held on exchanges", "chg"), "x_exflow": ("BTC network: BTC flowing into exchanges (USD)", "attn"),
    "x_exin_btc": ("BTC network: BTC flowing into exchanges (BTC)", "attn"), "x_exout_btc": ("BTC network: BTC flowing out of exchanges (BTC)", "attn"),
    "x_gdelt_tone": ("Tone of bitcoin news (GDELT)", "lvl"),
    "x_kp": ("Space: geomagnetic storms (Kp index)", "lvl"), "x_ap": ("Space: geomagnetic activity (Ap)", "lvl"),
    "x_f107": ("Space: solar radio flux (F10.7)", "lvl"),
}
CFTC = {  # name -> (dataset, CFTC contract code (fixed despite name changes), long column, short column, label)
    "x_cot_es": ("gpe5-46if", "13874A", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds: S&P 500"),
    "x_cot_nq": ("gpe5-46if", "209742", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds: Nasdaq"),
    "x_cot_ty": ("gpe5-46if", "043602", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds: 10-year bonds"),
    "x_cot_jy": ("gpe5-46if", "097741", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds: yen"),
    "x_cot_ec": ("gpe5-46if", "099741", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds: euro"),
    "x_cot_vx": ("gpe5-46if", "1170E1", "lev_money_positions_long", "lev_money_positions_short", "Leveraged funds: VIX"),
    "x_cot_gc": ("72hh-3qpy", "088691", "m_money_positions_long_all", "m_money_positions_short_all", "Funds: gold"),
    "x_cot_cl": ("72hh-3qpy", "067651", "m_money_positions_long_all", "m_money_positions_short_all", "Funds: oil"),
}
KIND.update({k: (v[4] + " (net position / market)", "lvl") for k, v in CFTC.items()})
HN_TERMS = {"x_hn_bitcoin": "bitcoin", "x_hn_crypto": "crypto", "x_hn_ai": "AI", "x_hn_recession": "recession", "x_hn_layoffs": "layoffs", "x_hn_inflation": "inflation"}
KIND.update({k: (f'Hacker News: mentions of "{v}"', "attn") for k, v in HN_TERMS.items()})


def _get(url, timeout=60, **kw):
    for i in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"no response: {url[:90]}")


def _ser(d):
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def nyfed():
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    out = {}
    j = _get(f"https://markets.newyorkfed.org/api/rp/reverserepo/propositions/search.json?startDate=2013-09-23&endDate={end}").json()
    rrp = {}
    for op in j["repo"]["operations"]:
        rrp[op["operationDate"]] = rrp.get(op["operationDate"], 0) + float(op.get("totalAmtAccepted") or 0)
    out["x_rrp"] = _ser(rrp)
    for name, path, start in (("x_sofr", "secured/sofr", "2018-04-02"), ("x_effr", "unsecured/effr", "2011-08-01")):
        j = _get(f"https://markets.newyorkfed.org/api/rates/{path}/search.json?startDate={start}&endDate={end}").json()
        out[name] = _ser({r["effectiveDate"]: r["percentRate"] for r in j["refRates"]})
    j = _get("https://markets.newyorkfed.org/api/soma/summary.json").json()
    out["x_soma"] = _ser({r["asOfDate"]: float(r["total"]) for r in j["soma"]["summary"] if r.get("total")})
    return out


def tga():
    rows, page = [], 1
    while page < 40:
        j = _get("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance"
                 f"?filter=record_date:gte:2011-08-01&fields=record_date,account_type,close_today_bal,open_today_bal&page[size]=10000&page[number]={page}").json()
        rows += j["data"]
        if page >= j["meta"]["total-pages"]:
            break
        page += 1
    d = {}
    for r in rows:
        t = r["account_type"].lower()
        if "closing balance" in t and ("treasury general account" in t or "federal reserve account" in t):
            v = r["close_today_bal"] if r["close_today_bal"] not in (None, "null", "") else r["open_today_bal"]
            try:
                d[r["record_date"]] = float(v)
            except (TypeError, ValueError):
                pass
        elif t == "federal reserve account" and r["record_date"] not in d:
            try:
                d[r["record_date"]] = float(r["close_today_bal"])
            except (TypeError, ValueError):
                pass
    return {"x_tga": _ser(d)}


def llama():
    out = {}
    j = _get("https://api.llama.fi/v2/historicalChainTvl").json()
    out["x_tvl"] = _ser({pd.Timestamp(int(r["date"]), unit="s"): r["tvl"] for r in j if r["tvl"] > 0})
    j = _get("https://api.llama.fi/overview/dexs?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true").json()
    out["x_dex"] = _ser({pd.Timestamp(int(t), unit="s"): v for t, v in j["totalDataChart"] if v and v > 0})
    for name, sid in (("x_usdt", 1), ("x_usdc", 2)):
        j = _get(f"https://stablecoins.llama.fi/stablecoincharts/all?stablecoin={sid}").json()
        out[name] = _ser({pd.Timestamp(int(r["date"]), unit="s"): r["totalCirculatingUSD"]["peggedUSD"] for r in j if r.get("totalCirculatingUSD")})
    return out


def lightning():
    j = _get("https://mempool.space/api/v1/lightning/statistics/all").json()
    return {"x_ln_cap": _ser({pd.Timestamp(int(r["added"]), unit="s"): r["total_capacity"] for r in j}),
            "x_ln_nodes": _ser({pd.Timestamp(int(r["added"]), unit="s"): sum(r.get(k) or 0 for k in ("tor_nodes", "clearnet_nodes", "unannounced_nodes", "clearnet_tor_nodes"))
                                for r in j})}


def coinmetrics():
    out = {}
    for name, m in (("x_txcnt", "TxCnt"), ("x_iss", "IssTotUSD"), ("x_exsupply", "SplyExNtv"), ("x_exflow", "FlowInExUSD"),
                    ("x_exin_btc", "FlowInExNtv"), ("x_exout_btc", "FlowOutExNtv")):
        url = f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics={m}&frequency=1d&start_time=2011-01-01&page_size=10000"
        d = {}
        while url:
            j = _get(url).json()
            for row in j.get("data", []):
                if row.get(m) is not None:
                    d[row["time"][:10]] = float(row[m])
            url = j.get("next_page_url")
        if d:
            out[name] = _ser(d)
    return out


def cftc():
    out = {}
    for name, (ds, pat, lo, sh, _) in CFTC.items():
        try:
            u = (f"https://publicreporting.cftc.gov/resource/{ds}.json?$limit=5000&$order=report_date_as_yyyy_mm_dd%20DESC"
                 f"&$where=cftc_contract_market_code=%27{pat}%27")
            rows = _get(u).json()
            d = {}
            for r in rows:
                try:
                    d[r["report_date_as_yyyy_mm_dd"][:10]] = (float(r[lo]) - float(r[sh])) / float(r["open_interest_all"]) * 100
                except (KeyError, ValueError, ZeroDivisionError):
                    pass
            if d:
                s = _ser(d)
                s.index = s.index + pd.Timedelta(days=3)       # CFTC publishes the Tuesday snapshot on Friday
                out[name] = s
        except Exception:
            continue
    return out


def space():
    t = _get("https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt", timeout=120).text
    rows = [l.split() for l in t.splitlines() if l and not l.startswith("#")]
    df = pd.DataFrame(rows)
    dt = pd.to_datetime(df[0] + "-" + df[1] + "-" + df[2], errors="coerce")
    kp = df[[7, 8, 9, 10, 11, 12, 13, 14]].astype(float).mean(axis=1)
    ap, f107 = df[23].astype(float), df[25].astype(float)
    m = dt >= "2011-01-01"
    clean = lambda s: pd.Series(s[m].values, index=dt[m].values).where(lambda x: x >= 0)
    return {"x_kp": clean(kp), "x_ap": clean(ap), "x_f107": clean(f107)}


def hackernews():
    """Monthly number of Hacker News stories with a given word (cached; only missing months are fetched)."""
    cache = json.loads(HN_CACHE.read_text(encoding="utf-8")) if HN_CACHE.exists() else {}
    months = pd.date_range("2011-08-01", pd.Timestamp.now(), freq="MS")
    out = {}
    for name, term in HN_TERMS.items():
        c = cache.setdefault(name, {})
        for i, m0 in enumerate(months):
            key = m0.strftime("%Y-%m")
            recent = i >= len(months) - 2
            if key in c and not recent:
                continue
            m1 = m0 + pd.offsets.MonthBegin(1)
            u = (f"https://hn.algolia.com/api/v1/search_by_date?query={requests.utils.quote(term)}&tags=story&hitsPerPage=0"
                 f"&numericFilters=created_at_i>{int(m0.timestamp())},created_at_i<{int(m1.timestamp())}")
            try:
                c[key] = int(_get(u, timeout=30).json()["nbHits"])
            except Exception:
                continue
            time.sleep(0.15)
        # a month's count is known only after it ends, so we assign it from day 1 of the NEXT month (no peeking into the future)
        now = pd.Timestamp.now().normalize()
        done = {k: v for k, v in c.items() if pd.Timestamp(k + "-01") + pd.offsets.MonthBegin(1) <= now}
        s = pd.Series({pd.Timestamp(k + "-01") + pd.offsets.MonthBegin(1): v / pd.Timestamp(k + "-01").days_in_month for k, v in done.items()}).sort_index()
        out[name] = s.reindex(pd.date_range(s.index.min(), now)).ffill()
    HN_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return out


GDELT_CACHE = C.DATA_DIR / "gdelt_tone.csv"


def gdelt_tone():
    """Daily average tone of English-language articles about bitcoin (GDELT DOC 2.0, since 2017, no key). We only fetch the last 3 months."""
    old = pd.read_csv(GDELT_CACHE, index_col=0, parse_dates=True)["tone"] if GDELT_CACHE.exists() else pd.Series(dtype=float)
    start = (old.index.max() - pd.Timedelta(days=90)) if len(old) else pd.Timestamp("2017-01-01")
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    time.sleep(6)                                                # GDELT: at most one request every 5 s
    r = _get("https://api.gdeltproject.org/api/v2/doc/doc", timeout=120, params={
        "query": "bitcoin sourcelang:english", "mode": "timelinetone", "format": "json",
        "startdatetime": start.strftime("%Y%m%d000000"), "enddatetime": end.strftime("%Y%m%d000000")})
    pts = r.json()["timeline"][0]["data"]
    s = pd.Series({pd.Timestamp(p["date"][:8]): float(p["value"]) for p in pts})
    s = s[s != 0] if len(s) else s                               # 0 = no articles on that day (e.g. the current day)
    s = s.combine_first(old).sort_index() if len(old) else s.sort_index()
    s.rename("tone").to_csv(GDELT_CACHE)
    return {"x_gdelt_tone": s}


JOBS = [("NY Fed (liquidity, rates)", nyfed), ("US Treasury (TGA)", tga), ("DefiLlama (DeFi, DEX, USDT, USDC)", llama), ("mempool.space (Lightning)", lightning),
        ("CoinMetrics (exchanges, transactions)", coinmetrics), ("CFTC (positions in other markets)", cftc), ("GFZ (space weather)", space),
        ("Hacker News (tech attention)", hackernews), ("GDELT (tone of bitcoin news)", gdelt_tone)]


def update(verbose=True):
    data, status = {}, []
    for name, fn in JOBS:
        t0 = time.time()
        try:
            r = fn()
            data.update(r)
            status.append((name, "OK", f"{len(r)} series, {time.time() - t0:.0f}s"))
        except Exception as e:
            status.append((name, "ERROR", str(e)[:120]))
    if verbose:
        for s in status:
            print(f"  [{s[1]:5}] {s[0]}: {s[2]}", flush=True)
    old = pd.read_csv(OUT, index_col=0, parse_dates=True) if OUT.exists() else pd.DataFrame()
    new = pd.DataFrame(data)
    if "x_sofr" in new and "x_effr" in new:
        new["x_sofr_effr"] = new["x_sofr"] - new["x_effr"]
    merged = new.combine_first(old) if len(old) else new
    merged = merged.sort_index().loc["2011-01-01":]
    # net liquidity = Fed portfolio - government account - reverse repo (USD millions); series with different frequencies aligned to the last known value
    if {"x_soma", "x_tga", "x_rrp"} <= set(merged.columns):
        idx = pd.date_range("2013-09-23", merged.index.max())
        soma = merged["x_soma"].reindex(idx).ffill(limit=10) / 1e6
        tga_ = merged["x_tga"].reindex(idx).ffill(limit=10)
        rrp = merged["x_rrp"].reindex(idx).fillna(0) / 1e6
        merged["x_netliq"] = (soma - tga_ - rrp).reindex(merged.index)
    C.DATA_DIR.mkdir(exist_ok=True)
    merged.to_csv(OUT, float_format="%.8g")
    return merged, status
