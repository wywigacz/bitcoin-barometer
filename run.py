"""Bitcoin Barometer: update data -> analysis -> forecasts -> dashboard.

  python run.py                    incremental update (default)
  python run.py --full             download the full history from scratch
  python run.py --install-schedule Windows Task Scheduler job (every 6 hours)
  python run.py --remove-schedule  remove the job
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import config as C
import confirm
import dirtrack
import ensemble
import events
import extra as xt
import judge
import lev
import opthist
import model
import markets
import news
import regularity
import report
import rv
import scan
import sim
import sources
import store
import touch
import trader
import volev
import whales
import world
import worldlog

TASK = "Bitcoin-Barometer"


def update(full=False):
    old = store.load_daily()
    since = None if (full or old.empty) else old.index.max() - pd.Timedelta(days=30)
    print("Downloading data" + (" (full history)" if since is None else f" from {since:%Y-%m-%d}") + " ...")
    fetched, status = sources.fetch_all(since)
    for name, st, info in status:
        print(f"  [{st:5}] {name}: {info}")
    if "close" not in fetched:
        sys.exit("No BTC price - stopping (the data was not changed).")
    raw = store.merge_save(old, fetched)
    df = model.build_frame(raw)

    res = model.analyze(df)                      # signals, composite indicator, regime
    fr = res["frame"]
    print("Events and attribution of moves ...")
    E = events.analyze_events(df)
    print("World log (Wikipedia) and Bitcoin headlines (GDELT) ...")
    try:
        last = df.index[-1] - pd.Timedelta(days=1)
        worldlog.update(last, budget_s=3600 if full else 300)
        worldlog.save_daily(pd.date_range(worldlog.START, last))
    except Exception as e:
        print("  [warning] world log:", str(e)[:120])
    try:
        if not os.environ.get("BTC_SKIP_NEWS"):                 # lets us skip this when another process downloads the headlines
            news.update_news(df, budget_s=1800 if full else 120)
    except Exception as e:
        print("  [warning] GDELT headlines:", str(e)[:120])
    print("Stocks and bonds (S&P 500 + funds) ...")
    MKT = None
    try:
        px, sp = markets.update()
        mf = markets.features(px, sp)
        mf.to_csv(markets.FEAT, float_format="%.6g")
        MKT = {"comove": markets.comove(df, px, sp), "analogs": markets.corr_analogs(df, px), "feat": mf}
    except Exception as e:
        print("  [warning] markets:", str(e)[:150])
    print("Dollar liquidity, CFTC positions, DeFi, Lightning, Hacker News, space weather ...")
    try:
        xt.update()
    except Exception as e:
        print("  [warning] extra signals:", str(e)[:120])
    print("Volatility from 5-minute data (Binance) ...")
    try:
        rv.update()
    except Exception as e:
        print("  [warning] 5-minute data:", str(e)[:120])
    print("Deribit options history and Binance leverage ...")
    for nm, fn in (("options history", lambda: opthist.update(max_days=30)), ("leverage", lev.update)):
        try:
            fn()
        except Exception as e:
            print(f"  [warning] {nm}:", str(e)[:120])
    print("Big players: ETF funds, Strategy, Hyperliquid traders ...")
    try:
        whales.update()
    except Exception as e:
        print("  [warning] big players:", str(e)[:120])
    print("World data (markets, attention, weather, the Sun) ...")
    try:
        raw_w, fails = world.update_world(full=full or not world.WORLD_CSV.exists())
        if fails:
            print("  [warning] missing series:", ", ".join(fails))
    except Exception as e:
        print("  [warning] world:", str(e)[:120])
        raw_w = pd.read_csv(world.WORLD_CSV, index_col=0, parse_dates=True) if world.WORLD_CSV.exists() else None
    if raw_w is not None and len(raw_w):
        extra = pd.DataFrame(index=df.index)
        for k in ("ntx", "fees", "txvol", "uaddr", "difficulty", "hash", "minerrev", "adr", "volume", "mvrv", "fng"):
            if k in df:
                extra[f"d_{k}"] = df[k]
        if "cbclose" in df:
            extra["d_cbprem"] = (df["cbclose"] / df["close"] - 1) * 100
        if "bfund" in df:
            extra["d_bfund"] = df["bfund"].rolling(7, min_periods=3).mean() * 100
        cols = raw_w.columns
        if "y_btcf" in cols:
            extra["d_basis"] = (raw_w["y_btcf"].reindex(df.index).ffill(limit=4) / df["close"] - 1) * 100
        if "y_c_eth" in cols:
            extra["d_ethbtc"] = raw_w["y_c_eth"].reindex(df.index) / df["close"]
            alts = [c for c in ("y_c_xrp", "y_c_ltc", "y_c_bch", "y_c_xmr", "y_c_ada", "y_c_trx", "y_c_link", "y_c_doge") if c in cols]
            if alts:                                     # fixed basket listed since 2017, computed only when all are present (no jumps at new listings)
                A = raw_w[alts].reindex(df.index).dropna()
                extra["d_altbtc"] = np.exp(np.log(A).mean(axis=1)) / df["close"].reindex(A.index)
        raw_x = raw_w.loc["2011-08-01":].join(extra, how="outer")
        T = world.transforms(raw_x, df.index)
    else:
        T = None
    print("System regularity and relationship scanner ...")
    Rg = regularity.run(df)
    S = scan.run_scan(df, T) if T is not None else None
    print("Confirmation test (big players, hypotheses written down in advance) ...")
    CF = None
    try:
        CF = confirm.run(df)
        print("  " + ", ".join(f'{r["id"]}: {r["status"]}' for r in CF["rows"]))
    except Exception as e:
        print("  [warning] confirmation test:", str(e)[:120])
    DT = None
    try:                                                         # separate direction track (chance of a rise in 7 days, Brier score)
        DT = dirtrack.run(df, CF)
        if DT:
            print(f'  direction track: {"passed" if DT["passed"] else "did not pass"}, today a {DT["p_now"] * 100:.0f}% chance of a rise in 7 days')
    except Exception as e:
        print("  [warning] direction track:", str(e)[:120])
    conf = None
    if CF and CF["confirmed"]:                                   # only confirmed signals go into a separate method in the ensemble
        conf = pd.concat([confirm._z(CF["features"][c]) for c in CF["confirmed"]], axis=1).mean(axis=1)
    print("Model ensemble (walk-forward) ...")
    SHORT = [1, 3]                                               # short horizons: only a test of whether we can beat a coin toss
    ens = ensemble.run(df, E["events"], fr["features"], fr["scores"], fr["composite"], Rg, T, conf, SHORT + C.HORIZONS)
    ens["short"] = {h: ens["horizons"].pop(h) for h in SHORT}
    ens["forecasts"] = [f for f in ens["forecasts"] if f["horizon"] not in SHORT]
    print("Chances of touching price levels ...")
    TCH, SIM = None, None
    sig_now = ensemble.vol_models(df)[0]
    try:
        TCH = touch.run(df, sig_now)
    except Exception as e:
        print("  [warning] touch chances:", str(e)[:120])
    print("Judge: betting markets (Polymarket, Kalshi) ...")
    JD = None
    try:
        JD = judge.run(df, sig_now, (TCH or {}).get("best", "drift"), df.index[-1])
    except Exception as e:
        print("  [warning] judge:", str(e)[:120])
    print("Calm or stormy (volatility events) ...")
    VE = None
    try:
        VE = volev.run(df, sig_now)
        print("  " + ", ".join(f'{k}: {v["p_now"] * 100:.0f}%{" (confirmed)" if v["confirmed"] else ""}' for k, v in VE.items() if v.get("p_now") is not None))
    except Exception as e:
        print("  [warning] volatility events:", str(e)[:120])
    print("Virtual portfolio (today's decision and a test on history) ...")
    try:
        SIM = sim.run(df, sig_now)
        if SIM.get("live"):
            print(f"  portfolio: {SIM['live']['value']:,.0f} PLN on deposits of {SIM['live']['deposited']:,.0f} PLN, {SIM['live']['w'] * 100:.0f}% in BTC")
    except Exception as e:
        print("  [warning] virtual portfolio:", str(e)[:120])
    TR = None
    try:                                                         # virtual trader: a decision at every cloud update
        TR = trader.run(df, sig_now, VE)
        if TR.get("log"):
            r0 = TR["log"][0]
            print(f'  trader: {r0["local"]} {r0["action"]} {r0["amount_pln"]:,.0f} PLN, portfolio {TR["value"]:,.0f} PLN' + ("" if TR["wrote"] else " (no new decision)"))
    except Exception as e:
        print("  [warning] trader:", str(e)[:120])
    res["forecasts"] = ens["forecasts"]
    for f in res["forecasts"]:
        f["pred_z"] = 0.0
        f["p5"], f["p95"] = f.get("p5"), f.get("p95")
    res["quality"] = {h: {"active": v["alpha"] > 0, "n": v["n"]} for h, v in ens["horizons"].items()}
    log = store.update_log(res, df)
    stats = store.log_stats(log)
    try:                                                         # ledger for the future: every method, append-only
        ens["ledger"] = store.ledger_stats(store.update_ledger(res["asof"], res["price"], ens, df["close"]))
    except Exception as e:
        print("  [warning] method ledger:", str(e)[:120])

    C.OUT_DIR.mkdir(exist_ok=True)
    slim = {k: v for k, v in res.items() if k != "frame"}
    slim["status"] = status
    slim["log_stats"] = stats
    slim["ensemble"] = {k: v for k, v in ens.items() if k not in ("Xraw", "hmm")}
    slim["ensemble"]["hmm"] = {k: v for k, v in ens["hmm"].items() if k != "p_hi_series"}
    slim["events"] = {"table": E["table"].round(4).to_dict("records"), "moves": E["moves"], "cover": E["cover"], "upcoming": E["upcoming"]}
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    slim["scan"] = ({"summary": S["summary"], "discoveries": S["discoveries"].round(4).to_dict("records"),
                     "candidates": S["candidates"].round(4).to_dict("records")} if S else None)
    slim["regularity"] = {k: v for k, v in Rg.items() if k not in ("rolling", "kalman")}
    slim["regularity"]["kalman"] = {k: v for k, v in Rg["kalman"].items() if k != "series"}
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    slim["confirm"] = {k: v for k, v in CF.items() if k != "features"} if CF else None
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    slim["touch"] = {k: v for k, v in TCH.items()} if TCH else None
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    slim["volev"] = VE
    slim["judge"] = JD
    slim["dirtrack"] = DT
    slim["trader"] = {k: v for k, v in (TR or {}).items() if k != "log"} or None
    C.LATEST_JSON.write_text(json.dumps(slim, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    report.write_dashboard(res, status, log, stats, ens, E, df, S, Rg, MKT, CF, TCH, SIM, VE, JD, DT, TR)
    report.write_journal(TR)
    print_summary(res)
    return res


def print_summary(res):
    print(f"\nBTC {res['price']:,.0f} USD  ({res['asof']})  daily volatility {res['daily_vol']*100:.2f}%")
    print(f"Composite indicator: {res['composite']:+.2f}  -> {res['regime']}")
    print(f"{'Horizon':>9} {'Date':>11} {'P10':>9} {'Median':>9} {'P90':>9} {'P(rise)':>10}  model")
    for f in res["forecasts"]:
        q = res["quality"][f["horizon"]]
        tag = "model ensemble" if q["active"] else "volatility only"
        print(f"{f['horizon']:>7} d {f['target']:>11} {f['p10']:>9,.0f} {f['p50']:>9,.0f} {f['p90']:>9,.0f} {f['prob_up']*100:>9.0f}%  {tag}")
    print(f"\nDashboard: {C.DASHBOARD_HTML}")


def schedule(install=True):
    if install:
        cmd = ["schtasks", "/Create", "/F", "/TN", TASK, "/SC", "HOURLY", "/MO", "6",
               "/TR", f'"{sys.executable}" "{C.ROOT / "run.py"}"']
    else:
        cmd = ["schtasks", "/Delete", "/F", "/TN", TASK]
    print(subprocess.run(cmd, capture_output=True, text=True).stdout or "done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--install-schedule", action="store_true")
    ap.add_argument("--remove-schedule", action="store_true")
    a = ap.parse_args()
    if a.install_schedule:
        schedule(True)
    elif a.remove_schedule:
        schedule(False)
    else:
        update(a.full)
