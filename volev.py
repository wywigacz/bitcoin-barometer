"""Volatility events, which is what this system can really forecast:
  burza7  - will the next week be stormy (7-day volatility above its median over the last year),
  dzien5  - will there be a day with a price move of at least 5% (up or down) within 7 days,
  dzien10 - will there be a day with a move of at least 10% within 30 days.
Model: logistic regression on today's volatility forecast, refitted every week only on events
whose outcome was already known. Benchmarks: the fixed rate from the past, and "like recently" (the rate of the event
given what happened in the most recent period). Exam: the 2 years held out from 2024-10-29.
Rule (written down 2026-10-30, before the results): "edge confirmed" only when the model beats "like recently" both before and after the cutoff.
Keys (burza7, dzien5, dzien10, przed, po, calosc) are internal and stay as they are."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HOLDOUT = pd.Timestamp("2024-10-29")
START = pd.Timestamp("2015-09-01")
STEP = 7
EVENTS = {
    "burza7": {"H": 7, "label": "Stormy week", "q": "Will next week be stormy?"},
    "dzien5": {"H": 7, "label": "Day with a ≥5% move in 7 days", "q": "Will there be a day with a move of 5% or more within a week?"},
    "dzien10": {"H": 30, "label": "Day with a ≥10% move in 30 days", "q": "Will there be a day with a move of 10% or more within a month?"},
}


def _series(df, sig):
    r = np.log(df["close"]).diff()
    r2 = r ** 2
    rv7 = r2.rolling(7).mean()                                   # volatility over the last 7 days (known on day t)
    med = rv7.rolling(365, min_periods=200).median()
    absr = r.abs()
    S = pd.DataFrame(index=df.index)
    S["sig"] = sig
    S["x_rel"] = np.log(S["sig"] ** 2 / med)
    S["x_abs"] = np.log(S["sig"])
    # future events (what we score) and their past counterparts (for "like recently")
    S["y_burza7"] = (rv7.shift(-7) > med).astype(float).where(rv7.shift(-7).notna() & med.notna())
    S["p_burza7"] = (rv7 > med).astype(float).where(med.notna())
    fwd = lambda H: absr[::-1].rolling(H).max()[::-1].shift(-1)            # max |r| over days t+1..t+H
    S["y_dzien5"] = (fwd(7) >= np.log(1.05)).astype(float).where(fwd(7).notna())
    S["p_dzien5"] = (absr.rolling(7).max() >= np.log(1.05)).astype(float)
    S["y_dzien10"] = (fwd(30) >= np.log(1.10)).astype(float).where(fwd(30).notna())
    S["p_dzien10"] = (absr.rolling(30).max() >= np.log(1.10)).astype(float)
    return S


def run(df, sig):
    S = _series(df, sig)
    idx = S.index
    out = {}
    for ev, meta in EVENTS.items():
        H = meta["H"]
        x = S["x_rel" if ev == "burza7" else "x_abs"].to_numpy(float)
        y = S[f"y_{ev}"].to_numpy(float)
        prev = S[f"p_{ev}"].to_numpy(float)
        rows = []
        first = int(np.searchsorted(idx, START))
        for t in range(first, len(idx) - H, STEP):
            known = np.arange(0, t - H + 1)
            known = known[np.isfinite(x[known]) & np.isfinite(y[known]) & np.isfinite(prev[known])]
            if len(known) < 500 or not (np.isfinite(x[t]) and np.isfinite(y[t]) and np.isfinite(prev[t])):
                continue
            m = LogisticRegression(C=1.0).fit(x[known][:, None], y[known])
            p_mod = float(m.predict_proba([[x[t]]])[0, 1])
            p_clim = float(y[known].mean())
            same = known[prev[known] == prev[t]]
            p_pers = float(y[same].mean()) if len(same) >= 30 else p_clim
            rows.append((idx[t], y[t], p_mod, p_clim, p_pers))
        R = pd.DataFrame(rows, columns=["d", "y", "mod", "clim", "pers"])
        b = lambda col, msk: float(((R[col][msk] - R["y"][msk]) ** 2).mean())
        pre = (R["d"] + pd.Timedelta(days=H) < HOLDOUT).to_numpy()
        post = (R["d"] >= HOLDOUT).to_numpy()
        res = {}
        for nm, msk in (("przed", pre), ("po", post), ("calosc", np.ones(len(R), bool))):
            bm, bc, bp = b("mod", msk), b("clim", msk), b("pers", msk)
            res[nm] = {"n": int(msk.sum()), "brier": bm, "skill_clim": 1 - bm / bc, "skill_pers": 1 - bm / bp}
        # reliability: in buckets of the forecast chance (whole history)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        rel = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = (R["mod"] >= lo) & (R["mod"] < hi)
            if mk.sum() >= 15:
                rel.append({"lo": lo, "hi": hi, "n": int(mk.sum()), "pred": float(R["mod"][mk].mean()), "real": float(R["y"][mk].mean())})
        # today's chance: model fitted on all known outcomes
        t = len(idx) - 1
        known = np.arange(0, t - H + 1)
        known = known[np.isfinite(x[known]) & np.isfinite(y[known])]
        m = LogisticRegression(C=1.0).fit(x[known][:, None], y[known])
        p_now = float(m.predict_proba([[x[t]]])[0, 1]) if np.isfinite(x[t]) else None
        confirmed = res["przed"]["skill_pers"] > 0 and res["po"]["skill_pers"] > 0
        out[ev] = {**meta, "p_now": p_now, "base_rate": float(y[known].mean()), "eval": res, "reliability": rel, "confirmed": bool(confirmed)}
    return out
