"""A virtual, actively managed portfolio (an educational experiment, no real money and no real trades).

Rules (set in advance on 2026-10-29, public and never changed after the fact):
  - deposits: 10,000 PLN at the start and 10,000 PLN on the first day of every month,
  - every day (after the daily close) the program chooses what share of the portfolio to hold in BTC and what share in cash (PLN),
  - three strategies propose a BTC share (internal key -> display name):
      regularne -> "steady" - always 100% (like plain regular buying),
      ryzyko    -> "risk"   - volatility control: min(100%, normal volatility / forecast volatility) (Moreira and Muir 2017),
      trend     -> "trend"  - 100% when the price rose over the last 28 days, otherwise 25% (confirmed hypothesis H7),
  - learning: strategy weights follow the Hedge algorithm (exponential weights): weight ~ exp(ETA x log gain of the strategy over the last 365 days),
    at least 5% per strategy; it learns only from the past,
  - trade only when the target differs from the current share by more than 5 percentage points; fee 0.2% of the amount traded,
  - buy/sell at the daily closing price; the decision uses only data known at the end of that day.
Benchmark: plain regular buying (every deposit goes straight into BTC, no selling, same fee).
Taxes ignored here (in Poland crypto gains are taxed at 19%, PIT-38 form); see taxed() for the after-tax version.

Change 2026-10-30 (item 5 of the plan): volatility for the "risk" strategy is an exponential average of squared returns (no parameters
fitted on the whole history), so the backtest and the exam do not peek into the future. Plus a robustness block:
results for every start date since 2014, a comparison with a "blind" portfolio with the same average BTC exposure
(fixed weight, no signals at all), a bootstrap confidence interval, fragility of the trend window and a simple average of the strategies (1/N)."""
import json

import numpy as np
import pandas as pd

import config as C

DEPOSIT = 10000.0
FEE = 0.002
BAND = 0.05
ETA = 2.0
WINDOW = 365
FLOOR = 0.05
LIVE_START = pd.Timestamp("2026-10-28")
BACKTEST_START = pd.Timestamp("2018-01-01")
EXAM_START = pd.Timestamp("2024-10-29")              # exam on the last 2 years
LIVE_LOG = C.DATA_DIR / "sim_live.csv"
STRATS = ("regularne", "ryzyko", "trend")           # internal keys (stored in CSV columns t_*/p_*), do not rename
NAMES = {"regularne": "steady", "ryzyko": "risk", "trend": "trend"}   # display names used in the journal text


def inputs(df, sig):
    import world
    w = pd.read_csv(world.WORLD_CSV, index_col=0, parse_dates=True)["y_pln"]
    pln = w.reindex(df.index.union(w.index)).sort_index().ffill(limit=7).reindex(df.index)
    X = pd.DataFrame(index=df.index)
    X["px"] = df["close"] * pln                                  # BTC price in PLN
    X["usd"] = df["close"]
    X["pln"] = pln
    X["sig"] = sig
    X["sig_norm"] = pd.Series(sig, df.index).loc["2013":].expanding(365).median().reindex(df.index)
    X["mom28"] = np.log(df["close"] / df["close"].shift(28))
    X = X.dropna(subset=["px", "sig"])
    r = X["px"].pct_change().fillna(0)
    T = pd.DataFrame(index=X.index)
    T["regularne"] = 1.0
    T["ryzyko"] = (X["sig_norm"] / X["sig"]).clip(upper=1.0).fillna(1.0)
    T["trend"] = np.where(X["mom28"] > 0, 1.0, 0.25)
    # strategy results: daily return = yesterday's BTC share x today's BTC return (what each would have earned on its own)
    G = np.log1p(T.shift(1).fillna(1.0).mul(r, axis=0))
    S = G.rolling(WINDOW, min_periods=60).sum().fillna(0)
    E = np.exp(ETA * (S - S.max(axis=1).to_numpy()[:, None]))
    P = E.div(E.sum(axis=1), axis=0)
    P = (P.clip(lower=FLOOR)).div(P.clip(lower=FLOOR).sum(axis=1), axis=0)
    X["target"] = (P * T).sum(axis=1)
    return X, T, P, S


def _why(d, x, t, p, w_now, target, action, amt_pln):
    pc = lambda v: f"{v * 100:.0f}%"
    s_y, n_y = x["sig"] * np.sqrt(365) * 100, x["sig_norm"] * np.sqrt(365) * 100
    risk = (f"expected swings of {s_y:.0f}% a year against a normal {n_y:.0f}%, so “risk” wants {pc(t['ryzyko'])} in BTC"
            if np.isfinite(n_y) else f"“risk” wants {pc(t['ryzyko'])} in BTC")
    trend = f"the price moved {x['mom28'] * 100:+.1f}% in 28 days, so “trend” wants {pc(t['trend'])}"
    best = max(STRATS, key=lambda k: p[k])
    learn = (f"weights learned over the last year: steady {pc(p['regularne'])}, risk {pc(p['ryzyko'])}, trend {pc(p['trend'])} "
             f"(“{NAMES[best]}” is doing best)")
    head = f"Target {pc(target)} in BTC, was {pc(w_now)}. "
    kw = f"{amt_pln:,.0f}"
    act = {"KUP": f"I buy BTC for {kw} PLN.", "SPRZEDAJ": f"I sell BTC for {kw} PLN.",
           "TRZYMAJ": f"The gap is under {BAND * 100:.0f} percentage points, so I do nothing (and save the fee)."}[action]
    return head + act + " Why: " + risk + "; " + trend + "; " + learn + "."


def simulate(X, T, P, start, end=None, log=True, state=None):
    """Engine: day by day, from start to end. state lets us continue a saved portfolio."""
    D = X.loc[start:end] if end is not None else X.loc[start:]
    cash, btc, dep = (state or {}).get("cash", 0.0), (state or {}).get("btc", 0.0), (state or {}).get("deposited", 0.0)
    last_month = (state or {}).get("last_month")
    first = state is None
    dca_btc, dca_dep = (state or {}).get("dca_btc", 0.0), dep
    rows = []
    for d, x in D.iterrows():
        px = float(x["px"])
        deposit = 0.0
        ym = d.strftime("%Y-%m")
        if first or ym != last_month:
            deposit = DEPOSIT
            cash += deposit
            dep += deposit
            dca_btc += deposit * (1 - FEE) / px
            dca_dep += deposit
            last_month, first = ym, False
        value = cash + btc * px
        w_now = btc * px / value if value > 0 else 0.0
        target = float(x["target"])
        action, amt = "TRZYMAJ", 0.0
        if abs(target - w_now) > BAND:
            amt = abs(target - w_now) * value
            if target > w_now:
                amt = min(amt, cash)
                btc += amt * (1 - FEE) / px
                cash -= amt
                action = "KUP"
            else:
                q = amt / px
                btc -= q
                cash += amt * (1 - FEE)
                action = "SPRZEDAJ"
        value = cash + btc * px
        row = {"date": d.strftime("%Y-%m-%d"), "price_pln": round(px, 2), "deposit": deposit, "action": action, "amount_pln": round(amt, 2),
               "target": round(target, 4), "w_before": round(w_now, 4), "btc": btc, "cash": round(cash, 2), "value": round(value, 2),
               "deposited": dep, "dca_value": round(dca_btc * px, 2), "dca_btc": dca_btc,
               **{f"t_{k}": round(float(T.at[d, k]), 4) for k in STRATS}, **{f"p_{k}": round(float(P.at[d, k]), 4) for k in STRATS}}
        if log:
            row["why"] = _why(d, x, T.loc[d], P.loc[d], w_now, target, action, amt)
        row["last_month"] = last_month
        rows.append(row)
    return pd.DataFrame(rows)


NBP_CSV = C.DATA_DIR / "nbp_rate.csv"
TAX = 0.19


def nbp_rate():
    """NBP (Polish central bank) reference rate (daily, % a year) from the BIS (no key); saved to data/nbp_rate.csv."""
    import io
    import requests
    try:
        r = requests.get("https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.PL", params={"format": "csv", "startPeriod": "2013-01-01"}, timeout=60)
        d = pd.read_csv(io.StringIO(r.text))[["TIME_PERIOD", "OBS_VALUE"]].dropna()
        s = pd.Series(d["OBS_VALUE"].astype(float).to_numpy(), index=pd.to_datetime(d["TIME_PERIOD"]), name="rate")
        s.to_csv(NBP_CSV)
    except Exception:
        pass
    return pd.read_csv(NBP_CSV, index_col=0, parse_dates=True)["rate"] if NBP_CSV.exists() else None


def taxed(X, start, target, rate):
    """The active portfolio and regular buying with Polish tax and interest (simplified version, item 13 of the plan):
    - 19% of the yearly income from virtual currencies: sale proceeds minus purchase costs (cash basis), excess costs
      carry over to later years (Art. 30b(1a) of the Polish PIT Act); the tax is paid on April 30 of the following year,
    - cash in a deposit account: the NBP reference rate minus 1 percentage point (not below 0), interest reduced by the 19% "Belka" tax,
    - at the end both portfolios are sold and taxed (regular buying pays all of its tax only then)."""
    D = X.loc[start:]
    rt = (rate.reindex(D.index.union(rate.index)).sort_index().ffill().reindex(D.index).fillna(0) if rate is not None
          else pd.Series(0.0, D.index))
    cash = btc = dbtc = 0.0
    dep = interest = taxes = carry = 0.0
    rev, cost = {}, {}
    last_month, first, paid_for = None, True, set()

    def settle(y):
        nonlocal carry, taxes
        inc = rev.get(y, 0.0) - cost.get(y, 0.0) - carry
        if inc > 0:
            carry = 0.0
            return TAX * inc
        carry = -inc
        return 0.0

    for d, x in D.iterrows():
        px = float(x["px"])
        cash += cash * max(float(rt.loc[d]) - 1.0, 0.0) / 100 / 365 * (1 - TAX)
        interest_today = 0.0
        ym = d.strftime("%Y-%m")
        if first or ym != last_month:
            cash += DEPOSIT
            dep += DEPOSIT
            dbtc += DEPOSIT * (1 - FEE) / px
            last_month, first = ym, False
        y_prev = d.year - 1
        if d >= pd.Timestamp(f"{d.year}-04-30") and y_prev not in paid_for and y_prev >= pd.Timestamp(start).year:
            t = settle(y_prev)
            paid_for.add(y_prev)
            if t > 0:
                if cash < t:                                   # not enough cash: sell BTC to pay the tax
                    q = min(btc, (t - cash) / (px * (1 - FEE)))
                    btc -= q
                    cash += q * px * (1 - FEE)
                    rev[d.year] = rev.get(d.year, 0.0) + q * px * (1 - FEE)
                cash -= t
                taxes += t
        value = cash + btc * px
        w_now = btc * px / value if value > 0 else 0.0
        tg = float(target.loc[d])
        if abs(tg - w_now) > BAND:
            amt = abs(tg - w_now) * value
            if tg > w_now:
                amt = min(amt, cash)
                btc += amt * (1 - FEE) / px
                cash -= amt
                cost[d.year] = cost.get(d.year, 0.0) + amt
            else:
                btc -= amt / px
                cash += amt * (1 - FEE)
                rev[d.year] = rev.get(d.year, 0.0) + amt * (1 - FEE)
    px = float(D["px"].iloc[-1])
    pre = cash + btc * px
    rev[D.index[-1].year] = rev.get(D.index[-1].year, 0.0) + btc * px * (1 - FEE)
    cash += btc * px * (1 - FEE)
    for y in sorted(set(rev) | set(cost)):                    # settle the years not yet paid (liquidation at the end)
        if y not in paid_for:
            t = settle(y)
            cash -= t
            taxes += t
    dca_pre = dbtc * px
    dca_rev = dbtc * px * (1 - FEE)
    dca_tax = TAX * max(dca_rev - dep, 0.0)                  # costs = all deposits, proceeds only at the end
    return {"deposited": dep, "active_pre": pre, "active_after": cash, "active_tax": taxes,
            "dca_pre": dca_pre, "dca_after": dca_rev - dca_tax, "dca_tax": dca_tax}


def _twr(v, dep):
    """Time-weighted return (no effect from deposits): daily return = (value today - deposit today) / value yesterday."""
    g = np.ones(len(v))
    g[1:] = (v[1:] - dep[1:]) / v[:-1]
    return np.cumprod(g)


def _stats(R, col="value"):
    idx = _twr(R[col].to_numpy(float), R["deposit"].to_numpy(float))
    yrs = max(len(R) / 365, 1e-9)
    dd = float((idx / np.maximum.accumulate(idx) - 1).min())
    return {"final": float(R[col].iloc[-1]), "deposited": float(R["deposited"].iloc[-1]),
            "gain_pct": float(R[col].iloc[-1] / R["deposited"].iloc[-1] - 1),
            "twr_ann": float(idx[-1] ** (1 / yrs) - 1), "max_dd": dd}


def causal_sigma(df):
    """Daily volatility computed only from the past: exponential average of squared returns (fixed alpha from config)."""
    r = np.log(df["close"]).diff().fillna(0) ** 2
    return np.sqrt(r.ewm(alpha=C.EWMA_ALPHA, adjust=False, min_periods=30).mean()).bfill().to_numpy()


def _with_target(X, target):
    X2 = X.copy()
    X2["target"] = target
    return X2


def _boot_ci(d, block=20, reps=2000, seed=11):
    """90% interval for the mean (Politis and Romano stationary bootstrap, average block of 20 days)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    if n < 60:
        return None
    means = np.empty(reps)
    for b in range(reps):
        idx = np.empty(n, int)
        i = rng.integers(n)
        for j in range(n):
            if j and rng.random() < 1 / block:
                i = rng.integers(n)
            idx[j] = i
            i = (i + 1) % n
        means[b] = d[idx].mean()
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def robustness(X, T, P, df):
    """Is the edge skill, or just luck with the start date and lower exposure?"""
    out = {}
    eqw = T[list(STRATS)].mean(axis=1)
    for years in (2, 4):
        rows = []
        for st in pd.date_range("2014-01-01", X.index[-1] - pd.DateOffset(years=years), freq="MS"):
            en = st + pd.DateOffset(years=years) - pd.Timedelta(days=1)
            A = simulate(X, T, P, st, en, log=False)
            if len(A) < 300:
                continue
            w = float((A["btc"] * A["price_pln"] / A["value"]).mean())   # average exposure of the active portfolio
            B = simulate(_with_target(X, w), T, P, st, en, log=False)
            N = simulate(_with_target(X, eqw), T, P, st, en, log=False)
            rows.append({"start": str(st.date()), "vs_dca": float(A["value"].iloc[-1] / A["dca_value"].iloc[-1]),
                         "vs_blind": float(A["value"].iloc[-1] / B["value"].iloc[-1]), "n_vs_dca": float(N["value"].iloc[-1] / N["dca_value"].iloc[-1]),
                         "dd_a": _stats(A)["max_dd"], "dd_dca": _stats(A, "dca_value")["max_dd"], "dd_blind": _stats(B)["max_dd"], "w": w})
        R = pd.DataFrame(rows)
        if R.empty:
            continue
        q = lambda s, p: float(s.quantile(p))
        out[f"{years}y"] = {"n": int(len(R)), "beat_dca": float((R["vs_dca"] > 1).mean()), "med_vs_dca": float(R["vs_dca"].median()),
                            "p10": q(R["vs_dca"], 0.1), "p90": q(R["vs_dca"], 0.9),
                            "beat_blind": float((R["vs_blind"] > 1).mean()), "med_vs_blind": float(R["vs_blind"].median()),
                            "beat_dca_1n": float((R["n_vs_dca"] > 1).mean()),
                            "dd_vs_dca_pp": float((R["dd_a"] - R["dd_dca"]).median() * 100),
                            "dd_vs_blind_pp": float((R["dd_a"] - R["dd_blind"]).median() * 100), "avg_w": float(R["w"].mean())}
    # confidence interval for the edge on the exam (daily returns without the effect of deposits)
    E = simulate(X, T, P, EXAM_START, log=False)
    v, dep = E["value"].to_numpy(float), E["deposit"].to_numpy(float)
    vd = E["dca_value"].to_numpy(float)
    ga = np.log((v[1:] - dep[1:]) / v[:-1])
    gd = np.log((vd[1:] - dep[1:] * (1 - FEE)) / vd[:-1])
    ci = _boot_ci(ga - gd)
    out["exam_ci"] = {"mean_ann": float((ga - gd).mean() * 365), "lo_ann": ci[0] * 365, "hi_ann": ci[1] * 365} if ci else None
    # fragility: the trend rule alone with different windows
    lc = np.log(df["close"])
    frag = []
    for L in (14, 28, 56, 84, 112, 200):
        tgt = pd.Series(np.where((lc - lc.shift(L)).reindex(X.index) > 0, 1.0, 0.25), X.index)
        A = simulate(_with_target(X, tgt), T, P, BACKTEST_START, log=False)
        B = simulate(_with_target(X, tgt), T, P, EXAM_START, log=False)
        frag.append({"L": L, "since2018": float(A["value"].iloc[-1] / A["dca_value"].iloc[-1]), "exam": float(B["value"].iloc[-1] / B["dca_value"].iloc[-1])})
    out["fragility"] = frag
    # 1/N (equal strategy weights) next to Hedge
    for nm, tgt in (("hedge", X["target"]), ("1n", eqw)):
        A = simulate(_with_target(X, tgt), T, P, BACKTEST_START, log=False)
        B = simulate(_with_target(X, tgt), T, P, EXAM_START, log=False)
        out[nm] = {"since2018": float(A["value"].iloc[-1] / A["dca_value"].iloc[-1]), "exam": float(B["value"].iloc[-1] / B["dca_value"].iloc[-1])}
    return out


def run(df, sig):
    sig = causal_sigma(df)                                       # since 2026-10-30: no peeking into the future (see the module description)
    X, T, P, S = inputs(df, sig)
    done = X.index[X.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]   # closed days only
    X, T, P = X.loc[done], T.loc[done], P.loc[done]
    # backtest (recomputed from scratch at every update, with the same rules)
    B = simulate(X, T, P, BACKTEST_START, log=False)
    bt = {"active": _stats(B), "dca": _stats(B, "dca_value"),
          "trades": int((B["action"] != "TRZYMAJ").sum()), "fees_est": float(B["amount_pln"].sum() * FEE),
          "avg_btc": float(B["w_before"].mean()), "start": str(BACKTEST_START.date()),
          "curve": {"d": B["date"].iloc[::14].tolist() + [B["date"].iloc[-1]],
                    "a": B["value"].iloc[::14].round(0).tolist() + [float(B["value"].iloc[-1])],
                    "b": B["dca_value"].iloc[::14].round(0).tolist() + [float(B["dca_value"].iloc[-1])],
                    "dep": B["deposited"].iloc[::14].tolist() + [float(B["deposited"].iloc[-1])]}}
    # exam: start with the first deposit exactly 2 years ago, same rules
    Ex = simulate(X, T, P, EXAM_START, log=False)
    bt["exam"] = {"start": str(EXAM_START.date()), "active": _stats(Ex), "dca": _stats(Ex, "dca_value"),
                  "trades": int((Ex["action"] != "TRZYMAJ").sum())}
    bt["robust"] = robustness(X, T, P, df)
    try:                                                         # item 13: after tax and with interest on cash
        rate = nbp_rate()
        bt["tax"] = {"since2018": taxed(X, BACKTEST_START, X["target"], rate), "exam": taxed(X, EXAM_START, X["target"], rate)}
    except Exception as e:
        print("  [warning] portfolio tax:", str(e)[:120])
    # live portfolio: append-only log (we never recompute past decisions)
    old = pd.read_csv(LIVE_LOG) if LIVE_LOG.exists() else pd.DataFrame()
    if len(old):
        last = old.iloc[-1]
        state = {"cash": float(last["cash"]), "btc": float(last["btc"]), "deposited": float(last["deposited"]),
                 "last_month": str(last["last_month"]), "dca_btc": float(last["dca_btc"])}
        nxt = pd.Timestamp(last["date"]) + pd.Timedelta(days=1)
        new = simulate(X, T, P, nxt, state=state) if nxt <= X.index[-1] else pd.DataFrame()
    else:
        new = simulate(X, T, P, LIVE_START)
    live = pd.concat([old, new], ignore_index=True) if len(new) else old
    if len(live):
        C.DATA_DIR.mkdir(exist_ok=True)
        live.to_csv(LIVE_LOG, index=False)
    out = {"backtest": bt, "live_n": int(len(live)), "rules": {"deposit": DEPOSIT, "fee": FEE, "band": BAND, "eta": ETA, "window": WINDOW}}
    if len(live):
        L = live.iloc[-1]
        out["live"] = {"date": L["date"], "value": float(L["value"]), "deposited": float(L["deposited"]), "cash": float(L["cash"]),
                       "btc": float(L["btc"]), "dca_value": float(L["dca_value"]), "target": float(L["target"]),
                       "w": float(L["btc"] * L["price_pln"] / L["value"]) if L["value"] > 0 else 0.0,
                       "p": {k: float(L[f"p_{k}"]) for k in STRATS}, "t": {k: float(L[f"t_{k}"]) for k in STRATS},
                       "log": live.tail(20).iloc[::-1][["date", "action", "amount_pln", "price_pln", "target", "w_before", "value", "deposit", "why"]].to_dict("records")}
    return out
