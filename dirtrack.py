"""Separate direction track (item 12 of the plan): the chance that the price will be higher in 7 days, computed directly from confirmed signals.

Why separately: the main method test compares the whole distribution (7 quantiles), and a weak but real direction signal shifts
the distribution so little that it cannot pass that test (a signal the size of H7 gives about 0.8% improvement against a 2% threshold).
Here we judge only the direction.
Model: logistic regression on the average of signals checked in advance (today the 28-day trend and their combination), refitted every week
only on outcomes already known; a forecast every day since 2015. Scoring: Brier score against the fixed historical rate of rises
(computed only from the past), DM test with the HLN correction, direction hit rate.
Rule (written down 2026-10-31, before the results): the chance from this track replaces 50% on the dashboard only if it beats the fixed rate
with p < 0.10 both before 2024-10-29 and on the exam. Note: we confirmed the signals on the whole history, so the "before" period is not
clean; the only clean check is the exam and the future."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import confirm as CO
import ensemble as EN

H = 7
START = pd.Timestamp("2015-09-01")
HOLDOUT = pd.Timestamp("2024-10-29")


def run(df, CF):
    if not CF or not CF.get("confirmed"):
        return None
    F = CF["features"]
    x = pd.concat([CO._z(F[c]) for c in CF["confirmed"]], axis=1).mean(axis=1).to_numpy(float)
    lr = np.log(df["close"])
    y = ((lr.shift(-H) - lr) > 0).astype(float).where((lr.shift(-H)).notna()).to_numpy(float)
    idx = df.index
    rows, model, next_fit = [], None, 0
    for t in range(int(np.searchsorted(idx, START)), len(idx)):
        if not np.isfinite(x[t]):
            continue
        known = np.arange(0, max(t - H + 1, 0))
        known = known[np.isfinite(x[known]) & np.isfinite(y[known])]
        if len(known) < 500:
            continue
        if model is None or t >= next_fit:
            model = LogisticRegression(C=1.0).fit(x[known][:, None], y[known])
            next_fit = t + 7
        p = float(model.predict_proba([[x[t]]])[0, 1])
        clim = float(y[known].mean())
        rows.append((idx[t], p, clim, y[t]))
    R = pd.DataFrame(rows, columns=["d", "p", "clim", "y"])
    live = R.iloc[-1] if len(R) else None
    R = R.dropna(subset=["y"])
    lag = H + 1
    res = {}
    for nm, m in (("przed", (R["d"] + pd.Timedelta(days=H) < HOLDOUT).to_numpy()), ("po", (R["d"] >= HOLDOUT).to_numpy())):
        if m.sum() < 60:
            continue
        lp, lc = (R["p"][m] - R["y"][m]) ** 2, (R["clim"][m] - R["y"][m]) ** 2
        _, pv = EN._dm(lc.to_numpy(), lp.to_numpy(), lag)
        hit = ((R["p"][m] > 0.5) == (R["y"][m] > 0.5)).astype(float)
        res[nm] = {"n": int(m.sum()), "n_eff": float(m.sum() / H), "skill": float(1 - lp.mean() / lc.mean()), "p": pv,
                   "hit": float(hit.mean()), "mde": EN._mde(lc.to_numpy(), lp.to_numpy(), lag), "up_rate": float(R["y"][m].mean())}
    passed = all(k in res and res[k]["skill"] > 0 and res[k]["p"] == res[k]["p"] and res[k]["p"] < 0.10 for k in ("przed", "po"))
    return {"eval": res, "passed": bool(passed), "p_now": None if live is None else float(live["p"]),
            "clim_now": None if live is None else float(live["clim"]), "signals": list(CF["confirmed"])}
