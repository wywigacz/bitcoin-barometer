"""A virtual trader (an educational experiment, no real money and no real trades), since 2026-10-31.

It takes over the live portfolio from sim.py (the same cash and BTC as on Oct 30, 2026) and from then on decides at every cloud
update, that is about every 6 hours, at the current Bitstamp price (PLN exchange rate from the NBP). Rules (written down 2026-10-31, public):
  - deposits: 10,000 PLN at the first decision of every month,
  - goal: as many zlotys as possible at the end, i.e. the highest growth of the portfolio value over the long run,
  - the BTC share is set by the same three strategies as in sim.py, weighted with the Hedge algorithm (the strategy that would
    have multiplied the money the most over the last year gets the biggest say); the trend signal uses the current price, not yesterday's close,
  - trade only when the target differs from the current share by more than 5 percentage points (a 0.2% fee on each trade would eat the gain from small moves),
  - every decision is recorded with its reasons and the condition under which the trader will change its mind; the log is append-only,
  - every trade is scored after 1 and 7 days: how many PLN better or worse it did than leaving the portfolio unchanged.
Decisions are written only by the cloud run (GITHUB_ACTIONS) or a run with BTC_TRADER=1; a normal local run only reads the log.
Action codes KUP (buy) / SPRZEDAJ (sell) / TRZYMAJ (hold) are internal values stored in data/trader_log.csv; do not rename them."""
import os

import numpy as np
import pandas as pd
import requests

import config as C
import sim as SI

LOG = C.DATA_DIR / "trader_log.csv"
START = pd.Timestamp("2026-10-31")
MIN_GAP_H = 3                                        # no two decisions closer than 3 hours apart
TZ = "Europe/Warsaw"


def _live_price():
    try:
        r = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd/", timeout=20).json()
        return float(r["last"]), float(r["open"]), "Bitstamp"
    except Exception:
        return None, None, None


def _usdpln(fallback):
    try:
        r = requests.get("https://api.nbp.pl/api/exchangerates/rates/a/usd/?format=json", timeout=20).json()
        return float(r["rates"][0]["mid"]), "NBP"
    except Exception:
        return float(fallback), "last known"


def _zl(v):
    return f"{v:,.0f} PLN"


def _pc(v):
    return f"{v * 100:.0f}%"


def _start_state():
    """Portfolio state taken over from the sim.py log (the last day before the trader's start)."""
    s = pd.read_csv(SI.LIVE_LOG)
    s = s[pd.to_datetime(s["date"]) < START]
    last = s.iloc[-1]
    return {"cash": float(last["cash"]), "btc": float(last["btc"]), "deposited": float(last["deposited"]),
            "last_month": str(last["last_month"]), "dca_btc": float(last["dca_btc"])}


def _explain(ctx, t, p, w_now, target, action, amt, deposit):
    """Reasons in three parts: what I see, what I do and why, when I will change my mind."""
    best = max(SI.STRATS, key=lambda k: p[k])
    chg = ctx["chg24"]
    up = ctx["mom28"] > 0
    seen = (f"Bitcoin costs {ctx['usd']:,.0f} USD, which is {_zl(ctx['px'])}"
            + (f"; {chg * 100:+.1f}% over the last 24 hours" if chg is not None else "")
            + f". Over 28 days the price {'rose' if up else 'fell'} by {abs(np.expm1(ctx['mom28'])) * 100:.1f}%"
            + f", so the trend is {'up' if up else 'down'}. Expected swings: {ctx['vol_y']:.0f}% a year against a typical {ctx['vol_n']:.0f}%"
            + (" (calmer than usual)." if ctx["vol_y"] < ctx["vol_n"] else " (stormier than usual).")
            + (f" Chance of a stormy week: {_pc(ctx['storm'])}." if ctx.get("storm") is not None else ""))
    why = (f"The three strategies want: steady {_pc(t['regularne'])}, risk {_pc(t['ryzyko'])}, trend {_pc(t['trend'])} in BTC. "
           f"“{SI.NAMES[best].capitalize()}” would have earned the most over the last year, so it has the biggest say ({_pc(p[best])}). "
           f"The weighted average gives a target of {_pc(target)} in BTC; right now I hold {_pc(w_now)}.")
    dep = f"The monthly deposit of {_zl(deposit)} came in. " if deposit else ""
    gap = round(abs(target - w_now) * 100)
    act = {"KUP": f"I buy BTC for {_zl(amt)}.",
           "SPRZEDAJ": f"I sell BTC for {_zl(amt)}.",
           "TRZYMAJ": f"I do nothing: the gap is only {gap} percentage point{'' if gap == 1 else 's'}, and the fee would eat the gain from such a small adjustment."}[action]
    flip = ctx["px28"]
    if up:
        cond = (f"I will change my mind if the price falls below {_zl(flip)} (the level of 28 days ago): the trend would turn, "
                f"the trend strategy would drop from 100% to 25% in BTC, and I would sell some.")
    else:
        cond = (f"I will change my mind if the price rises above {_zl(flip)} (the level of 28 days ago): the trend would turn, "
                f"the trend strategy would raise the BTC share to 100%, and I would buy more.")
    return {"seen": seen, "do": dep + act + " " + why, "change": cond}


def _score(log, px_now, now):
    """Trade scorecard: the result after 1 and 7 days compared with leaving the portfolio unchanged (gain from the decision minus the fee)."""
    out = []
    tr = log[log["action"] != "TRZYMAJ"]
    for _, r in tr.iterrows():
        t0 = pd.Timestamp(r["ts"])
        dq = float(r["btc_after"]) - float(r["btc_before"])            # how much BTC was added (+) or removed (-)
        res = {"ts": r["ts"], "action": r["action"], "amount_pln": float(r["amount_pln"])}
        for h, lab in ((1, "d1"), (7, "d7")):
            later = log[pd.to_datetime(log["ts"]) >= t0 + pd.Timedelta(days=h)]
            p1 = float(later.iloc[0]["price_pln"]) if len(later) else (px_now if now >= t0 + pd.Timedelta(days=h) else None)
            res[lab] = None if p1 is None else dq * (p1 - float(r["price_pln"])) - float(r["amount_pln"]) * SI.FEE
        out.append(res)
    return out


def run(df, sig, VE=None):
    X, T, P, _ = SI.inputs(df, SI.causal_sigma(df))
    last = X.index[-1]
    x, t, p = X.loc[last], T.loc[last].to_dict(), P.loc[last].to_dict()
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    may_write = bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("BTC_TRADER"))
    due = len(old) == 0 or now - pd.Timestamp(old.iloc[-1]["ts"]) >= pd.Timedelta(hours=MIN_GAP_H)
    usd, open24, src = _live_price()
    if usd is None:
        usd, open24, src = float(df["close"].iloc[-1]), float(df["close"].iloc[-2]), "daily close"
    fx, fx_src = _usdpln(x["pln"])
    px = usd * fx
    c28 = float(df["close"].iloc[-28])                               # close from 28 days ago (as in the trend strategy)
    mom28 = float(np.log(usd / c28))
    t_live = dict(t, trend=1.0 if mom28 > 0 else 0.25)
    target = float(sum(p[k] * t_live[k] for k in SI.STRATS))
    storm = None
    try:
        storm = float(VE["burza7"]["p_now"]) if VE and VE.get("burza7") else None
    except Exception:
        pass
    ctx = {"usd": usd, "px": px, "chg24": usd / open24 - 1 if open24 else None, "mom28": mom28, "px28": c28 * fx,
           "vol_y": float(x["sig"] * np.sqrt(365) * 100), "vol_n": float(x["sig_norm"] * np.sqrt(365) * 100), "storm": storm}
    new = None
    if may_write and due:
        st = (dict(old.iloc[-1][["cash", "btc", "deposited", "last_month", "dca_btc"]]) if len(old) else _start_state())
        cash, btc, dep, dca_btc = float(st["cash"]), float(st["btc"]), float(st["deposited"]), float(st["dca_btc"])
        ym, deposit = now.strftime("%Y-%m"), 0.0
        if ym != str(st["last_month"]):
            deposit = SI.DEPOSIT
            cash += deposit
            dep += deposit
            dca_btc += deposit * (1 - SI.FEE) / px
        value = cash + btc * px
        w_now = btc * px / value if value > 0 else 0.0
        btc0, action, amt = btc, "TRZYMAJ", 0.0
        if abs(target - w_now) > SI.BAND:
            amt = abs(target - w_now) * value
            if target > w_now:
                amt = min(amt, cash)
                btc += amt * (1 - SI.FEE) / px
                cash -= amt
                action = "KUP"
            else:
                btc -= amt / px
                cash += amt * (1 - SI.FEE)
                action = "SPRZEDAJ"
        ex = _explain(ctx, t_live, p, w_now, target, action, amt, deposit)
        new = {"ts": now.strftime("%Y-%m-%d %H:%M"), "local": now.tz_localize("UTC").tz_convert(TZ).strftime("%Y-%m-%d %H:%M"),
               "price_usd": round(usd, 2), "usdpln": round(fx, 4), "price_pln": round(px, 2), "src": f"{src}, {fx_src} rate",
               "deposit": deposit, "action": action, "amount_pln": round(amt, 2), "target": round(target, 4), "w_before": round(w_now, 4),
               "btc_before": btc0, "btc_after": btc, "btc": btc, "cash": round(cash, 2), "value": round(cash + btc * px, 2),
               "deposited": dep, "dca_btc": dca_btc, "dca_value": round(dca_btc * px, 2), "last_month": ym,
               **{f"t_{k}": round(t_live[k], 4) for k in SI.STRATS}, **{f"p_{k}": round(p[k], 4) for k in SI.STRATS},
               "seen": ex["seen"], "do": ex["do"], "change": ex["change"]}
        old = pd.concat([old, pd.DataFrame([new])], ignore_index=True)
        C.DATA_DIR.mkdir(exist_ok=True)
        old.to_csv(LOG, index=False)
    if not len(old):
        return {"ctx": ctx, "target": target, "log": [], "score": []}
    L = old.iloc[-1]
    val_now = float(L["cash"]) + float(L["btc"]) * px
    tr = old[old["action"] != "TRZYMAJ"]
    return {"ctx": ctx, "target": target, "n": int(len(old)), "n_trades": int(len(tr)), "since": str(old.iloc[0]["local"]),
            "value": val_now, "deposited": float(L["deposited"]), "cash": float(L["cash"]), "btc": float(L["btc"]),
            "w": float(L["btc"]) * px / val_now if val_now > 0 else 0.0, "dca_value": float(L["dca_btc"]) * px,
            "fees": float(tr["amount_pln"].sum() * SI.FEE), "wrote": new is not None,
            "log": old.iloc[::-1].to_dict("records"), "score": _score(old, px, now)}
