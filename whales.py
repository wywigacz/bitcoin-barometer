"""Big-player moves: flows into Bitcoin ETFs, purchases and sales by Strategy (MicroStrategy),
and positions of the best and worst traders on the Hyperliquid exchange (all positions there are public).

Sources (no API keys):
  - TFTC Bitcoin ETF Tracker, CC BY 4.0 licence (daily data from SoSoValue): https://www.tftc.io/bitcoin-etf-flows
  - Bitbo, Strategy transaction table (matches the 8-K filings with the SEC): https://bitbo.io/treasuries/microstrategy/
  - Hyperliquid, public /info API and trader leaderboard.

Output: data/whales.csv (daily) and data/hl_snapshots.csv (position snapshots every 6 hours; the history cannot be downloaded
retroactively, so we have been recording it since 2026-10-29). These series do NOT go into the big scanner; confirm.py tests them separately."""
import re
import time

import numpy as np
import pandas as pd
import requests

import config as C

OUT = C.DATA_DIR / "whales.csv"
SNAP = C.DATA_DIR / "hl_snapshots.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"}
HL = "https://api.hyperliquid.xyz/info"
TOP = 150          # how many of the best and of the worst traders we follow in each snapshot


def _get(url, **kw):
    for i in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"no response: {url[:90]}")


def _hl(body):
    for i in range(3):
        try:
            r = requests.post(HL, json=body, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1 + 2 * i)
    raise RuntimeError(f"Hyperliquid: {body.get('type')}")


def etf():
    j = _get("https://www.tftc.io/bitcoin-etf-flows/data.json").json()
    rows = next(v for v in j.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "netFlowUsd" in v[0])
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["date"])
    out = pd.DataFrame(index=d.index)
    out["w_etf_flow"] = d["netFlowUsd"].astype(float)
    out["w_etf_aum"] = pd.to_numeric(d["totalNetAssetsUsd"], errors="coerce")
    per = pd.DataFrame([x if isinstance(x, dict) else {} for x in d["perEtfUsd"]], index=d.index)
    for tic, name in (("IBIT", "w_etf_ibit"), ("FBTC", "w_etf_fbtc"), ("GBTC", "w_etf_gbtc")):
        if tic in per:
            out[name] = pd.to_numeric(per[tic], errors="coerce")
    return out


def strategy():
    t = _get("https://bitbo.io/treasuries/microstrategy/").text
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S):
        c = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        c = [x for x in c if x]
        if len(c) >= 4 and re.match(r"\d+/\d+/\d\d(\d\d)?$", c[0]):
            n = c[1].replace(",", "")
            neg = n.startswith("(")
            try:
                btc = float(n.strip("()")) * (-1 if neg else 1)
                total = float(c[3].replace(",", ""))
            except ValueError:
                continue
            fmt = "%m/%d/%Y" if len(c[0].split("/")[2]) == 4 else "%m/%d/%y"
            rows.append((pd.to_datetime(c[0], format=fmt), btc, total))
    if len(rows) < 50:
        raise RuntimeError(f"Strategy table: only {len(rows)} rows")
    d = pd.DataFrame(rows, columns=["date", "btc", "total"]).groupby("date").agg({"btc": "sum", "total": "last"}).sort_index()
    # the table skips some small transactions, but the holdings after each report are complete: we compute the change from the difference in holdings
    chg = d["total"].diff().fillna(d["total"].iloc[0])
    return pd.DataFrame({"w_mstr_btc": chg, "w_mstr_total": d["total"]})


def hl_funding(since):
    """BTC funding rate on Hyperliquid (hourly, since May 2023), averaged per day."""
    start = int(pd.Timestamp(since).timestamp() * 1000)
    acc = []
    for _ in range(200):
        page = _hl({"type": "fundingHistory", "coin": "BTC", "startTime": start})
        if not page:
            break
        acc += page
        nxt = int(page[-1]["time"]) + 1
        if len(page) < 500 or nxt <= start:
            break
        start = nxt
        time.sleep(0.2)
    if not acc:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(int(r["time"]), unit="ms"): float(r["fundingRate"]) for r in acc})
    return s.groupby(s.index.normalize()).mean() * 24 * 365 * 100     # in % per year


def hl_snapshot():
    """Snapshot of BTC positions: the best traders (largest all-time account profit) versus the worst."""
    lb = _get("https://stats-data.hyperliquid.xyz/Mainnet/leaderboard").json()["leaderboardRows"]
    rows = []
    for r in lb:
        try:
            perf = dict(r["windowPerformances"])
            rows.append((r["ethAddress"], float(r["accountValue"]), float(perf["allTime"]["pnl"])))
        except (KeyError, ValueError, TypeError):
            continue
    L = pd.DataFrame(rows, columns=["addr", "value", "pnl"])
    smart = L[L["value"] >= 100_000].nlargest(TOP, "pnl")
    dumb = L[L["value"] >= 10_000].nsmallest(TOP, "pnl")
    ctx = _hl({"type": "metaAndAssetCtxs"})
    i = [u["name"] for u in ctx[0]["universe"]].index("BTC")
    px, oi = float(ctx[1][i]["markPx"]), float(ctx[1][i]["openInterest"])
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    snap = {"time": now, "px": px, "oi_usd": oi * px}
    for grp, G in (("smart", smart), ("dumb", dumb)):
        net, nl, ns = 0.0, 0, 0
        for a in G["addr"]:
            try:
                st = _hl({"type": "clearinghouseState", "user": a})
            except RuntimeError:
                continue
            for p in st.get("assetPositions", []):
                q = p.get("position", {})
                if q.get("coin") == "BTC":
                    sz = float(q["szi"])
                    net += sz * px
                    nl += sz > 0
                    ns += sz < 0
            time.sleep(0.06)
        snap.update({f"{grp}_net_usd": net, f"{grp}_long": nl, f"{grp}_short": ns, f"{grp}_n": len(G)})
    old = pd.read_csv(SNAP, parse_dates=["time"]) if SNAP.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([snap])], ignore_index=True).to_csv(SNAP, index=False, float_format="%.6g")
    return snap


def hl_daily():
    if not SNAP.exists():
        return pd.DataFrame()
    s = pd.read_csv(SNAP, parse_dates=["time"])
    s["day"] = s["time"].dt.normalize()
    g = s.groupby("day").mean(numeric_only=True)
    out = pd.DataFrame(index=g.index)
    for grp in ("smart", "dumb"):
        out[f"w_hl_{grp}"] = g[f"{grp}_net_usd"] / g["oi_usd"] * 100            # the group's net position as % of the whole BTC market on Hyperliquid
        out[f"w_hl_{grp}_longpct"] = g[f"{grp}_long"] / (g[f"{grp}_long"] + g[f"{grp}_short"]).replace(0, np.nan) * 100
    out["w_hl_oi"] = g["oi_usd"]
    return out


# Known wallets: every address was checked on the blockchain against public reports (date, amount) before being added (2026-10-29).
ENTITIES = {
    # US government: the address that received 69,369 BTC seized from the hacker “Individual X” on Nov 3, 2020 (DOJ; Arkham)
    "w_ent_us": ["bc1qa5wkgaew2dkv56kfvj49j0av5nml45x9ek9hz6"],
    # Germany, BKA police (Movie2k case): 5 source wallets (49,859 BTC in total, reports say 49,858) and 2 relay wallets;
    # sold Jun 19 - Jul 12, 2024 (Arkham, CoinDesk, heise)
    "w_ent_de": ["bc1qss8z846643h8ct2e8rzkwwslrt3ns874vtv93w", "bc1qgrvchamnmmaancn3vwea6elnvexpylzh30rhjz", "bc1qnwft8re0tkkyekd4nl34dqvgmwx3zsfyxfadfg",
                 "bc1q9agtd4sk5r06vqcsy3ewdr0dslf03dlj3agtva", "bc1q93njc4we4s088a2nz6c9e6vthc5h4ake53rxmd",
                 "bc1qq0l4jgg9rcm3puhhfwaz4c9t8hdee8hfz6738z", "bc1q0unygz3ddt8x0v33s6ztxkrnw0s0tl7zk4yxwd"],
}
ENT_CACHE = C.DATA_DIR / "entities_cache.json"
MP = "https://mempool.space/api"


def _mp(path):
    for i in range(6):
        try:
            r = requests.get(MP + path, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError(f"mempool.space: {path[:60]}")


def _addr_deltas(a, known):
    """New transactions of an address (newest first, up to the first one already known): {txid: [time, balance change in BTC]}."""
    new, last = {}, None
    while True:
        page = _mp(f"/address/{a}/txs/chain" + (f"/{last}" if last else ""))
        stop = False
        for t in page:
            if t["txid"] in known:
                stop = True
                break
            d = sum(v["value"] for v in t["vout"] if v.get("scriptpubkey_address") == a) \
                - sum(v["prevout"]["value"] for v in t["vin"] if (v.get("prevout") or {}).get("scriptpubkey_address") == a)
            new[t["txid"]] = [t["status"]["block_time"], d / 1e8]
        if stop or len(page) < 25:
            return new
        last = page[-1]["txid"]
        time.sleep(1.2)


def entities():
    import json
    cache = json.loads(ENT_CACHE.read_text(encoding="utf-8")) if ENT_CACHE.exists() else {}
    out = {}
    for name, addrs in ENTITIES.items():
        ev = []
        for a in addrs:
            c = cache.setdefault(a, {})
            c.update(_addr_deltas(a, c))
            ev += list(c.values())
            time.sleep(1)
        s = pd.Series([d for _, d in ev], index=pd.to_datetime([t for t, _ in ev], unit="s").normalize())
        out[name] = s.groupby(level=0).sum().sort_index().cumsum().clip(lower=0)
    ENT_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    # El Salvador publishes all its reserve addresses itself (National Bitcoin Office, bitcoin.gob.sv, via the mempool.space API)
    j = _get("https://bitcoin.gob.sv/api/v1/wallet/ONBTC").json()
    ev = [(t["time"], t["value"] / 1e8) for w in j.values() for t in w.get("transactions", [])]
    s = pd.Series([d for _, d in ev], index=pd.to_datetime([t for t, _ in ev], unit="s").normalize())
    out["w_ent_sv"] = s.groupby(level=0).sum().sort_index().cumsum()
    out["w_ent_sv_n"] = pd.Series(len(j), index=[out["w_ent_sv"].index[-1]], dtype=float)
    idx = pd.date_range(min(v.index.min() for v in out.values()), pd.Timestamp.now().normalize())
    return pd.DataFrame({k: (v.reindex(idx).ffill() if k != "w_ent_sv_n" else v.reindex(idx)) for k, v in out.items()})


def stable_peg():
    """Daily USDT and USDC prices (DefiLlama, since late 2020). A premium above 1 USD can be a sign of a flight from risk into “digital cash”."""
    j = _get("https://stablecoins.llama.fi/stablecoinprices").json()
    rows = [(pd.Timestamp(int(r["date"]), unit="s").normalize(), r["prices"].get("tether"), r["prices"].get("usd-coin")) for r in j if r["date"] > 0]
    d = pd.DataFrame(rows, columns=["date", "w_usdt_px", "w_usdc_px"]).dropna(how="all", subset=["w_usdt_px", "w_usdc_px"])
    d = d.groupby("date").last()
    return d[(d > 0.8) & (d < 1.2)]


RR_SNAP = C.DATA_DIR / "options_rr.csv"


def options_rr():
    """Options skew (25-delta risk reversal) for the expiry closest to 30 days: implied volatility of calls minus puts
    at the same “distance” from the price. Positive: the market pays more for bets on a rise. The history is not available
    for free, so we save a snapshot at every update (since 2026-10-29)."""
    from scipy.stats import norm
    res = _get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option").json()["result"]
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    rows = []
    for x in res:
        try:
            _, exp, k, cp = x["instrument_name"].split("-")
            T = (pd.Timestamp(pd.to_datetime(exp, format="%d%b%y")) + pd.Timedelta(hours=8) - now).total_seconds() / (365 * 86400)
            iv, F = float(x["mark_iv"]) / 100, float(x["underlying_price"])
            if T <= 3 / 365 or iv <= 0 or F <= 0:
                continue
            K = float(k)
            d1 = (np.log(F / K) + 0.5 * iv * iv * T) / (iv * np.sqrt(T))
            delta = norm.cdf(d1) if cp == "C" else norm.cdf(d1) - 1
            rows.append((exp, T, cp, K, F, iv, delta))
        except (ValueError, KeyError, TypeError):
            continue
    O = pd.DataFrame(rows, columns=["exp", "T", "cp", "K", "F", "iv", "delta"])
    if O.empty:
        raise RuntimeError("Deribit: no options")
    exp = O.groupby("exp")["T"].first()
    e = (exp - 30 / 365).abs().idxmin()
    E = O[O["exp"] == e]
    c = E[(E["cp"] == "C") & (E["K"] > E["F"])].sort_values("delta")
    p = E[(E["cp"] == "P") & (E["K"] < E["F"])].sort_values("delta")
    if len(c) < 3 or len(p) < 3:
        raise RuntimeError("Deribit: too few options to interpolate")
    ivc = float(np.interp(0.25, c["delta"], c["iv"]))
    ivp = float(np.interp(-0.25, p["delta"], p["iv"]))
    atm = float(E.iloc[(E["K"] - E["F"]).abs().argsort()[:2]]["iv"].mean())
    snap = {"time": now.floor("min"), "rr25": (ivc - ivp) * 100, "iv_c25": ivc * 100, "iv_p25": ivp * 100, "atm": atm * 100,
            "days": float(exp[e] * 365), "expiry": e}
    old = pd.read_csv(RR_SNAP, parse_dates=["time"]) if RR_SNAP.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([snap])], ignore_index=True).to_csv(RR_SNAP, index=False, float_format="%.6g")
    s = pd.read_csv(RR_SNAP, parse_dates=["time"])
    g = s.groupby(s["time"].dt.normalize())[["rr25", "atm"]].mean()
    return g.rename(columns={"rr25": "w_rr25", "atm": "w_atm_iv"})


def update(verbose=True):
    old = pd.read_csv(OUT, index_col=0, parse_dates=True) if OUT.exists() else pd.DataFrame()
    parts, status = [], []
    jobs = [("Bitcoin ETFs (TFTC)", etf), ("Strategy: BTC purchases and sales (Bitbo)", strategy),
            ("Hyperliquid: funding rate", lambda: hl_funding(old["w_hl_fund"].dropna().index.max() - pd.Timedelta(days=3)
                                                                    if "w_hl_fund" in old and old["w_hl_fund"].notna().any() else "2023-05-01").rename("w_hl_fund").to_frame()),
            ("Hyperliquid: positions of the best and worst traders", lambda: (hl_snapshot(), hl_daily())[1]),
            ("Government wallets of the US, Germany and El Salvador (blockchain)", entities),
            ("USDT and USDC stablecoin prices (DefiLlama)", stable_peg),
            ("25-delta options skew (Deribit, snapshot)", options_rr)]
    for name, fn in jobs:
        t0 = time.time()
        try:
            r = fn()
            parts.append(r)
            status.append((name, "OK", f"{r.shape[1]} series, {r.notna().any(axis=1).sum()} days, {time.time() - t0:.0f}s"))
        except Exception as e:
            status.append((name, "ERROR", str(e)[:120]))
    if verbose:
        for s in status:
            print(f"  [{s[1]:5}] {s[0]}: {s[2]}", flush=True)
    new = pd.concat([p for p in parts if len(p)], axis=1) if any(len(p) for p in parts) else pd.DataFrame()
    merged = new.combine_first(old) if len(old) else new
    merged = merged.sort_index()
    merged.index.name = "date"
    C.DATA_DIR.mkdir(exist_ok=True)
    merged.to_csv(OUT, float_format="%.8g")
    return merged, status
