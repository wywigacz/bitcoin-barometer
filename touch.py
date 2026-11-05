"""The chance that the price will TOUCH a given level within H days (not just where it ends up).

Method: a simulation of thousands of paths. Each day of a path is a randomly drawn real day from history: its price change at
the close and how high and how low the price reached during the day (all expressed in units of the volatility at that time
and rescaled to today's). This way a "touch" counts the way the dashboard promises: at least once during the day,
not only at the close. Variants, compared with the same backtest:
  base   - volatility returns to its long-term median with a 30-day half-life, neutral direction,
  fhs    - filtered historical simulation (Barone-Adesi et al. 1998): volatility in each path reacts to the drawn
           moves following GJR-GARCH (parameters fitted only to data known on the forecast day, refreshed every year),
  drift  - like base, but with a drift equal to the average daily move over the last 4 years (past only),
  fhs_drift - both at once.
We choose the variant only on forecasts settled before 2024-10-29; the last 2 years are the exam.

Rule (written down 2026-10-30, before the results): a touch counted on the daily highs and lows replaces counting on
closes if the best variant of that definition beats the fixed historical rate (also counted on highs/lows)
both before the cutoff and on the exam. Otherwise we stay with closes and the dashboard says "at the daily close".
Protection against single "wicks" on one exchange: we clip the daily range to the 0.1 to 99.9 percentile of history."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
HALF = 30.0
START = "2013-01-01"
VARIANTS = ("base", "fhs", "drift", "fhs_drift")
HOLDOUT = pd.Timestamp("2024-10-29")                  # exam: the last 2 years are held out from choosing the variant
LABELS = {"base": "Volatility returns to normal", "fhs": "Volatility reacts to moves (FHS, GARCH)",
          "drift": "With the drift of the last 4 years", "fhs_drift": "FHS with drift"}


def _days(df, sig):
    """Standardized days: change at the close (zc) and the reach up/down from yesterday's close (zh, zl)."""
    c = np.log(df["close"].to_numpy(float))
    hi = np.log(df["high"].to_numpy(float)) if "high" in df else c
    lo = np.log(df["low"].to_numpy(float)) if "low" in df else c
    s = np.r_[np.nan, sig[:-1]]                         # volatility known a day earlier
    prev = np.r_[np.nan, c[:-1]]
    zc, zh, zl = (c - prev) / s, (hi - prev) / s, (lo - prev) / s
    bad = (df.index < START) | ~np.isfinite(zc) | ~np.isfinite(zh) | ~np.isfinite(zl)
    zc[bad], zh[bad], zl[bad] = np.nan, np.nan, np.nan
    zh = np.maximum(zh, np.maximum(zc, 0.0))            # the daily high is not below the open and the close
    zl = np.minimum(zl, np.minimum(zc, 0.0))
    ok = np.isfinite(zc)
    if ok.sum() > 100:
        zh = np.minimum(zh, np.nanpercentile(zh[ok], 99.9))
        zl = np.maximum(zl, np.nanpercentile(zl[ok], 0.1))
    return np.column_stack([zc, zh, zl])


def _garch(r_pct):
    import ensemble as EN
    _, p = EN.garch_gjr(r_pct)
    w, a, g, b = p
    return float(a), float(g), float(b)


def _paths(s0, s_lr, pool, H, n, mode="base", gp=None, mu=0.0):
    """Returns (highest point of the path, lowest point of the path, max close, min close) in log(P/P0)."""
    k = RNG.integers(0, len(pool), size=(n, H))
    m, sd = pool[:, 0].mean(), pool[:, 0].std()
    zc, zh, zl = ((pool[k, j] - m) / sd for j in range(3))   # shift and scale shared by the whole day
    if mode in ("fhs", "fhs_drift") and gp is not None:
        a, g, b = gp
        hl = s_lr ** 2
        h = np.full(n, s0 ** 2)
        sc = np.empty((n, H))
        for t in range(H):
            sc[:, t] = np.sqrt(h)
            r = sc[:, t] * zc[:, t]
            h = hl * (1 - a - b - 0.5 * g) + (a + g * (r < 0)) * r ** 2 + b * h
            h = np.clip(h, hl * 0.05, hl * 50)
    else:
        t = np.arange(1, H + 1)
        sc = np.broadcast_to(s_lr + (s0 - s_lr) * np.exp(-np.log(2) * t / HALF), (n, H))
    d = mu if mode in ("drift", "fhs_drift") else 0.0
    close = np.cumsum(zc * sc + d, axis=1)
    prev = np.column_stack([np.zeros(n), close[:, :-1]])
    up, dn = prev + zh * sc + d, prev + zl * sc + d
    return up.max(axis=1), dn.min(axis=1), close.max(axis=1), close.min(axis=1)


def probs(ext, rel, hl=True):
    hi, lo = (ext[0], ext[1]) if hl else (ext[2], ext[3])
    out = []
    for x in rel:
        l = np.log(1 + x)
        out.append(float((hi >= l).mean()) if x > 0 else float((lo <= l).mean()))
    return out


def levels(price):
    step = 10000 if price > 40000 else 5000 if price > 15000 else 1000
    base = np.floor(price / step) * step
    up = [base + step * k for k in range(1, 5)]
    dn = [base - step * k for k in range(0, 4) if base - step * k > 0]
    return sorted(set(dn + up))


def _mu(lr_diff, t):
    """Average daily move over the last 4 years, known on day t."""
    w = lr_diff[max(0, t - 1460):t + 1]
    w = w[np.isfinite(w)]
    return float(w.mean()) if len(w) > 500 else 0.0


def run(df, sig, horizons=(7, 30, 90), n=20000):
    D = _days(df, sig)
    pool = D[np.isfinite(D).all(axis=1)]
    s_hist = pd.Series(sig, df.index)
    s_lr = float(s_hist.loc[START:].median())
    s0 = float(sig[-1])
    P = float(df["close"].iloc[-1])
    lr = np.log(df["close"].to_numpy(float))
    dlr = np.r_[np.nan, np.diff(lr)]
    r_pct = np.nan_to_num(dlr * 100)
    gp = _garch(r_pct[np.searchsorted(df.index, pd.Timestamp(START)):])
    cal = {"hl": calibrate(df, sig, D, hl=True), "close": calibrate(df, sig, D, hl=False)}
    best = {d: max(c, key=lambda m: c[m]["skill_pre"]) for d, c in cal.items()}    # chosen only on data from before the exam
    b_hl = cal["hl"][best["hl"]]
    use_hl = b_hl["skill_pre"] > 0 and b_hl["skill_post"] > 0
    dfn = "hl" if use_hl else "close"
    mode = best[dfn]
    mu = _mu(dlr, len(df) - 1)
    sims = {H: _paths(s0, s_lr, pool, H, n, mode, gp, mu) for H in horizons}
    table = []
    for L in levels(P):
        row = {"level": L, "rel": L / P - 1}
        for H in horizons:
            row[f"p{H}"] = probs(sims[H], [L / P - 1], use_hl)[0]
        table.append(row)
    rels = [-0.2, -0.1, 0.1, 0.2]
    simple = {H: dict(zip(rels, probs(sims[H], rels, use_hl))) for H in horizons}
    strip = lambda c: {m: {k: v for k, v in x.items() if k != "rows"} for m, x in c.items()}
    return {"price": P, "table": table, "simple": simple, "sigma_now": s0, "sigma_lr": s_lr,
            "definition": dfn, "best": mode, "calib": cal[dfn][mode], "variants": strip(cal[dfn]),
            "other_definition": {"definition": "close" if use_hl else "hl", "best": best["close" if use_hl else "hl"],
                                 "variants": strip(cal["close" if use_hl else "hl"])}}


def _brier_parts(pred, hit, bins=10):
    """Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    base = hit.mean()
    rel = res = 0.0
    for b in range(bins):
        m = k == b
        if m.any():
            w = m.mean()
            rel += w * (pred[m].mean() - hit[m].mean()) ** 2
            res += w * (hit[m].mean() - base) ** 2
    return float(rel), float(res), float(base * (1 - base))


def calibrate(df, sig, D, hl=True, H=30, rels=(-0.2, -0.1, 0.1, 0.2), n=3000, step=7):
    """Backtest of all variants: the forecast chance of a touch vs what actually happened. Only data known on the forecast day.
    hl=True: touch on the daily highs/lows; hl=False: on closes only."""
    lp_all = np.log(df["close"].to_numpy(float))
    hi_all = np.log(df["high"].to_numpy(float)) if hl and "high" in df else lp_all
    lo_all = np.log(df["low"].to_numpy(float)) if hl and "low" in df else lp_all
    dlr = np.r_[np.nan, np.diff(lp_all)]
    r_pct = np.nan_to_num(dlr * 100)
    idx = df.index
    s_hist = pd.Series(sig, idx)
    first = int(np.searchsorted(idx, pd.Timestamp("2016-01-01")))
    i0 = int(np.searchsorted(idx, pd.Timestamp(START)))
    gp, next_fit = None, 0
    acc = {m: {"pred": [], "hit": [], "which": [], "when": []} for m in VARIANTS}
    okD = np.isfinite(D).all(axis=1)
    for t in range(first, len(df) - H, step):
        pool = D[: t + 1][okD[: t + 1]]
        if len(pool) < 500:
            continue
        if gp is None or t >= next_fit:                 # GARCH fitted only to the past, refreshed every year
            gp = _garch(r_pct[i0:t + 1])
            next_fit = t + 365
        s_lr = float(s_hist.iloc[:t + 1].loc[START:].median())
        mu = _mu(dlr, t)
        fh = np.nanmax(hi_all[t + 1:t + H + 1]) - lp_all[t]
        fl = np.nanmin(lo_all[t + 1:t + H + 1]) - lp_all[t]
        hits = [float(fh >= np.log(1 + x)) if x > 0 else float(fl <= np.log(1 + x)) for x in rels]
        for m in VARIANTS:
            ext = _paths(float(sig[t]), s_lr, pool, H, n, m, gp, mu)
            for x, pr, h in zip(rels, probs(ext, list(rels), hl), hits):
                a = acc[m]
                a["pred"].append(pr); a["hit"].append(h); a["which"].append(x); a["when"].append(t)
    out = {}
    cut = int(np.searchsorted(idx, HOLDOUT))
    for m, a in acc.items():
        pred, hit, which, when = (np.array(a[k]) for k in ("pred", "hit", "which", "when"))
        # benchmark: the historical rate of touches of a given level, ONLY from trials finished before the forecast day
        base = np.full(len(pred), np.nan)
        for i in range(len(pred)):
            mm = (which == which[i]) & (when <= when[i] - H)
            if mm.sum() >= 30:
                base[i] = hit[mm].mean()
        ok = np.isfinite(base)
        pred, hit, which, base, when = pred[ok], hit[ok], which[ok], base[ok], when[ok]
        pre, post = when + H < cut, when >= cut                # exam: choose on "pre", check on "post"
        sk = lambda m_: float(1 - np.mean((pred[m_] - hit[m_]) ** 2) / np.mean((base[m_] - hit[m_]) ** 2)) if m_.sum() > 20 else np.nan
        bins = [0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = (pred >= lo) & (pred < hi)
            if mm.sum() >= 20:
                rows.append({"lo": lo, "hi": hi, "n": int(mm.sum()), "pred": float(pred[mm].mean()), "real": float(hit[mm].mean())})
        brier = float(np.mean((pred - hit) ** 2))
        brier0 = float(np.mean((base - hit) ** 2))
        rel, res, unc = _brier_parts(pred, hit)
        rel0, res0, _ = _brier_parts(base, hit)
        side = {float(x): (float(pred[which == x].mean()), float(hit[which == x].mean())) for x in rels}
        out[m] = {"rows": rows, "n": int(len(pred)), "brier": brier, "brier_base": brier0, "skill": 1 - brier / brier0, "H": H,
                  "side": side, "reliability": rel, "resolution": res, "uncertainty": unc, "reliability_base": rel0, "resolution_base": res0,
                  "skill_pre": sk(pre), "skill_post": sk(post), "n_post": int(post.sum())}
    return out
