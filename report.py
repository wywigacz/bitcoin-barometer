"""Dashboard HTML: one self-contained file (CSS + JS + data) that works offline."""
import html
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

# ----------------------------------------------------------------------------------------------
# Helper data
# ----------------------------------------------------------------------------------------------
GROUPS = [
    ("Valuation: today's price against history", "We compare today's price with what holders paid on average and with the price over recent years. These are the indicators that have held up best.", ["mvrv", "ma200w", "mayer"]),
    ("Bitcoin miners", "When mining stops paying, some miners give up. Historically this has often happened near price bottoms.", ["puell", "hashribbon"]),
    ("Mood and borrowed money", "Investor fear and greed, bets made with borrowed money, and the amount of cash ready to buy.", ["fng", "funding", "stable"]),
    ("The economy and money around the world", "Since 2020 Bitcoin has often behaved like a risky tech stock, but that is changing.", ["m2", "ndx", "dxy", "vix", "rates"]),
    ("The 4-year cycle", "About every 4 years there is a “halving” (the reward for mining is cut in half). Historically prices rose for about 1.5 years afterwards.", ["halving", "pi"]),
]
LABELS = {
    "mvrv": "Holders' profit (MVRV)", "ma200w": "Price vs 4-year average", "mayer": "Price vs 200-day average",
    "puell": "Mining profitability", "hashribbon": "Miners giving up", "fng": "Fear and greed",
    "funding": "Bets on a rise (borrowed money)", "stable": "Cash in crypto (stablecoins)", "m2": "Amount of money in the US",
    "dxy": "Dollar strength", "ndx": "Tech stocks", "vix": "Stock market fear (VIX)", "rates": "Interest rates",
    "pi": "Top warning (Pi Cycle)", "halving": "Place in the 4-year cycle",
}
SRC_NOTE = {
    "mvrv": "above 1 = holders are in profit on average", "ma200w": "1.0 = level with the 4-year average", "mayer": "1.0 = level with the 200-day average",
    "puell": "lower = mining earns less than usual", "hashribbon": "whether mining power has been falling lately", "fng": "0 = panic, 100 = euphoria",
    "funding": "positive = more bets on a rise", "stable": "rising = more cash ready to spend", "m2": "more money = usually better for the price",
    "dxy": "a stronger dollar usually hurts", "ndx": "change over the last 3 months", "vix": "higher = more fear", "rates": "change over 6 months, percentage points",
    "pi": "below 1 = no warning", "halving": "days since the last halving",
}
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def n(v, d=0, suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:,.{d}f}" + suffix


def raw_value(k, fe):
    g = fe.get
    try:
        return {
            "mvrv": n(g("mvrv"), 2), "ma200w": n(g("ma200w"), 2, "×"), "mayer": n(g("mayer"), 2, "×"),
            "puell": n(g("puell"), 2), "hashribbon": "giving up" if g("hashcap") == 1 else ("no" if g("hashcap") == 0 else "–"),
            "fng": n(g("fng"), 0), "funding": n((g("funding") or float("nan")) * 100, 4, "%"),
            "stable": n((g("stable30") or float("nan")) * 100, 1, "%"), "m2": n((g("m2yoy") or float("nan")) * 100, 1, "% y/y") if g("m2yoy") is not None else "no data",
            "dxy": n((g("dxy30") or float("nan")) * 100, 1, "%"), "ndx": n((g("ndx90") or float("nan")) * 100, 1, "%"),
            "vix": n(g("vix"), 1), "rates": n(g("rates180"), 2, " pp"), "pi": n(g("pi"), 2),
            "halving": n(g("halv_days"), 0, " days"),
        }[k]
    except Exception:
        return "–"


def ser_list(s, nd=2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


# ----------------------------------------------------------------------------------------------
# HTML sections
# ----------------------------------------------------------------------------------------------
def gauge_svg(v):
    """Semicircular gauge from -1 to +1."""
    cx, cy, r = 150, 150, 118
    ang = math.pi * (1 - (v + 1) / 2)              # +1 -> 0 rad (right), -1 -> pi (left)
    nx, ny = cx + (r - 22) * math.cos(ang), cy - (r - 22) * math.sin(ang)
    ticks = ""
    for i in range(0, 21):
        a = math.pi * (1 - i / 20)
        r1, r2 = (r + 8, r + 16) if i % 5 == 0 else (r + 8, r + 12)
        ticks += (f'<line x1="{cx + r1 * math.cos(a):.1f}" y1="{cy - r1 * math.sin(a):.1f}" '
                  f'x2="{cx + r2 * math.cos(a):.1f}" y2="{cy - r2 * math.sin(a):.1f}" class="gt"/>')
    return f"""<svg viewBox="0 0 300 185" class="gauge" role="img" aria-label="Composite indicator {v:+.2f}">
<defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" style="stop-color:var(--red)"/><stop offset=".5" style="stop-color:var(--amber)"/><stop offset="1" style="stop-color:var(--green)"/></linearGradient></defs>
<path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="var(--track)" stroke-width="16" stroke-linecap="round"/>
<path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="url(#gg)" stroke-width="16" stroke-linecap="round" opacity=".92"/>
{ticks}
<g class="needle" style="--rot:{(v) * 90:.1f}deg"><line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy-(r-22)}" stroke="var(--fg)" stroke-width="3" stroke-linecap="round"/></g>
<circle cx="{cx}" cy="{cy}" r="8" fill="var(--fg)"/><circle cx="{cx}" cy="{cy}" r="3" fill="var(--bg2)"/>
<text x="{cx-r}" y="{cy+22}" text-anchor="middle" class="gl">expensive</text>
<text x="{cx+r}" y="{cy+22}" text-anchor="middle" class="gl">cheap</text></svg>"""


def cycle_html(days):
    total = 1460
    zones = [(0, 365, "Rally", "var(--blue)"), (365, 560, "Peak", "var(--amber)"),
             (560, 770, "Decline", "var(--red)"), (770, 1000, "Bottom", "var(--green)"), (1000, 1460, "Recovery", "var(--violet)")]
    seg = "".join(f'<div class="zone" style="flex:{b-a};--c:{c}" title="{nm}: days {a}–{b} after the halving"><span>{nm}</span></div>' for a, b, nm, c in zones)
    d = days if days is not None and not math.isnan(days) else 0
    cur = next((nm for a, b, nm, _ in zones if a <= d < b), "Recovery")
    pos = min(max(d / total, 0), 1) * 100
    return (f'<div class="cycle"><div class="zones">{seg}<div class="marker" style="left:{pos:.2f}%"><i></i></div></div>'
            f'<div class="cyc-foot"><b>{n(d)} days</b> have passed since the last halving ({C.HALVINGS[-1]}). Cycle stage: <b>{cur}</b>. '
            f'Next halving around {C.NEXT_HALVING}.</div></div>')


def forecast_cards(res):
    cards = ""
    p0 = res["price"]
    for f in res["forecasts"]:
        q = res["quality"][f["horizon"]]
        lo, hi = math.log(f["p5"]), math.log(f["p95"])
        pos = lambda v: (math.log(v) - lo) / (hi - lo) * 100
        chg = (f["p50"] / p0 - 1) * 100
        up = f["prob_up"] * 100
        badge = ('<span class="badge on">backed by models</span>' if q["active"]
                 else '<span class="badge">typical swings</span>')
        v = f.get("view")
        view = (f'<div class="view"><span class="muted sm">What the indicators suggest <i>(tests do not confirm that this works)</i></span>'
                f'<b>{n(v["p50"])}</b><span class="muted sm"> · {(v["p50"]/p0-1)*100:+.1f}% · chance of a rise {v["prob_up"]*100:.0f}%</span></div>') if v else ""
        cards += f"""<article class="card fc">
<header><h3>In {f['horizon']} days</h3>{badge}</header>
<div class="muted sm">by {f['target']}</div>
<div class="fc-med">{n(f['p50'])}<small> USD</small></div>
<div class="fc-chg {'pos' if chg >= 0 else 'neg'}">{chg:+.1f}% vs today</div>
<div class="range"><div class="r90" style="left:{pos(f['p10']):.1f}%;width:{pos(f['p90'])-pos(f['p10']):.1f}%"></div>
<div class="r50" style="left:{pos(f['p25']):.1f}%;width:{pos(f['p75'])-pos(f['p25']):.1f}%"></div>
<div class="tick" style="left:{pos(f['p50']):.1f}%"></div><div class="now" style="left:{pos(p0):.1f}%" title="price today"></div></div>
<div class="range-l"><span>{n(f['p10'])}</span><span class="muted">typical range</span><span>{n(f['p90'])}</span></div>
<div class="pup"><div class="ring" style="--p:{up:.0f}"><b>{up:.0f}%</b></div><span class="muted sm">chance the price<br>will be higher than today</span></div>
{view}</article>"""
    return cards


MODEL_LAB = {"naive": "Guess: the price stays the same", "comp": "“Cheap or expensive” score", "ridge": "Combination of many indicators",
             "knn": "Looking for similar days in the past", "gbq": "Self-learning decision trees", "scan": "Searching for links around the world",
             "conf": "Signals confirmed by a test set up in advance (trend)", "eq": "Simple average of all methods (1/N)"}
LANG = {"en": "English-speaking countries", "de": "Germany and neighbours", "fr": "France", "es": "Spain and Latin America", "ru": "Russia",
        "ja": "Japan", "ko": "South Korea", "zh": "China", "tr": "Turkey", "pl": "Poland", "pt": "Brazil and Portugal"}


def tys(v):
    return f"{v / 1000:.0f}k"


def _stars(t, nobs):
    a = abs(t)
    if nobs < 8:
        return "too little data"
    return "high" if a >= 3 else "medium" if a >= 2 else "low"


def summary_html(res, ens, E):
    p0 = res["price"]
    fc = {f["horizon"]: f for f in res["forecasts"]}
    names = {7: "in a week", 30: "in a month", 90: "in 3 months", 180: "in 6 months"}
    items = "".join(f'<li><b>{names.get(H, f"in {H} days")}:</b> from {tys(fc[H]["p10"])} to {tys(fc[H]["p90"])} USD</li>' for H in fc)
    none = all(h["alpha"] == 0 for h in ens["horizons"].values())
    if none:
        direction = ('<b>Will the price go up or down? We cannot reliably predict that.</b> We checked this against history: '
                     'none of our methods did better than the simple assumption “the price will stay the same”. So the chance of a rise is about 50/50.')
    else:
        act = ", ".join(str(H) for H, h in ens["horizons"].items() if h["alpha"] > 0)
        direction = (f'<b>Will the price go up or down?</b> Over {act} days our methods turned out better than guessing and nudge the forecast slightly, '
                     'but this is still very uncertain. For the other periods we cannot reliably predict it.')
    nxt = next((u for u in E["upcoming"] if u["cat"] != "nfp"), None)
    ntxt = f'<li><b>Next important event:</b> {html.escape(nxt["title"])} ({nxt["date"]}).</li>' if nxt else ""
    return (f'<section class="card summary reveal"><h3>In short</h3><ul>'
            f'<li><b>Bitcoin costs {n(p0)} USD today.</b> {res["regime"]}. <span class="muted">(Market score {res["composite"]:+.2f} on a scale from −1 “expensive” to +1 “cheap”.)</span></li>'
            f'<li><b>Where could the price be?</b> The ranges given contain the real price in about 4 cases out of 5:<ul class="inner">{items}</ul></li>'
            f'<li>{direction}</li>'
            f'<li><b>What we do well:</b> judging how much the price may swing. In tests on history our ranges were right in about 8 cases out of 10.</li>'
            f'{ntxt}</ul><p class="muted sm" style="margin:10px 0 0">Below you will find the detailed reasoning. Difficult words are explained in the glossary at the bottom of the page. This is not investment advice.</p></section>')


def glossary_html():
    g = [
        ("Halving", "About every 4 years the reward for the people “mining” Bitcoin is cut in half. Historically prices rose for about 1.5 years after a halving, although each rise was smaller than the one before."),
        ("Typical range (“4 out of 5”)", "The price range that the price will fall into in about 80% of cases. One time in 10 it will be below, one time in 10 above."),
        ("Volatility", "How much the price jumps up and down. For Bitcoin, typically around 2% a day."),
        ("MVRV (holders' profit)", "Today's price divided by the average price at which holders bought their coins. Below 1 most holders are at a loss, which historically has often been near price bottoms."),
        ("Average purchase price", "The average price at which all existing coins last “moved”. The price rarely stays below this line for long."),
        ("Test on history", "We check a method as if it had been used in real time: on each day it only gets the data that was known then, and it guesses what happens next."),
        ("“The price will stay the same”", "A simple benchmark. Every method has to beat it, otherwise there is no point in using it."),
        ("Chance that it is luck", "The lower it is, the more sure we can be that a method's edge is real. We require it to be below 10%."),
        ("Correlation", "A measure of how closely two things move together: +1 always together, 0 independently, −1 in opposite directions."),
        ("FOMC", "Meeting of the US central bank (the Fed) where it decides on interest rates."),
        ("Options and DVOL", "Options are “insurance” against price changes. DVOL shows how much investors pay for that insurance, that is, how big they expect the swings to be."),
        ("Kimchi premium", "How much more Bitcoin costs in South Korea than in the rest of the world. A positive premium means Korean investors are very keen."),
        ("Put/Call", "The ratio of bets on a price fall to bets on a rise. Below 1, bets on a rise dominate."),
        ("CME and CFTC", "CME is a US futures exchange, and the CFTC is the agency that publishes every week how large funds are positioned."),
        ("Market regime", "Our model recognises whether the market is currently in a “calm” or a “nervous” mode (big swings)."),
        ("Power-law trend", "A long-term line around which Bitcoin's price has moved for 15 years. A price below the line has often been an opportunity, above it a sign of overheating."),
        ("Many tests at once", "When you check thousands of things, some will look connected purely by luck, like winning the lottery when you buy thousands of tickets. That is why we use stricter criteria."),
    ]
    return "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in g)


def models_html(ens):
    out = ""
    for H, h in ens["horizons"].items():
        rows = ""
        for r in h["table"]:
            p = r["dm_p"]
            ptxt = "–" if p != p else f"{p*100:.0f}%"
            lab = MODEL_LAB.get(r["model"], "Team of methods (final result)" if r["model"] == "ens" else r["label"])
            if r["model"] == "naive":
                sk = '<td class="muted">benchmark</td>'
            else:
                sk = (f'<td class="muted">0.0%</td>' if abs(r["skill"]) < 0.0005 else f'<td class="{"pos" if r["skill"] > 0 else "neg"}">{r["skill"]*100:+.1f}%</td>')
            rows += (f'<tr class="{"ens" if r["model"] == "ens" else ""}"><td>{lab}</td>{sk}<td>{ptxt}</td>'
                     f'<td>{r["cover80"]*10:.1f} of 10</td></tr>')
        used = ", ".join(f'{MODEL_LAB[m].lower()} {v*100:.0f}%' for m, v in h["weights"].items() if v > 0.005 and m != "naive")
        state = (f'<span class="badge on">in use: {used}</span>' if h["alpha"] > 0
                 else '<span class="badge">no method beat guessing</span>')
        out += (f'<div class="card q"><header><h3>In {H} days</h3>{state}</header>'
                f'<table class="tbl"><tr><th>Method</th><th title="How much smaller the errors are than when assuming the price stays the same (plus = better)">Better than guessing?</th>'
                f'<th title="Chance that the edge is just luck. The lower, the more certain.">Chance that it is luck</th><th title="In how many cases out of 10 the price stayed within the typical range (target: 8)">Hits within the range</th></tr>{rows}</table>'
                f'<div class="muted sm" style="margin-top:8px">Checked on {h["n"]} situations from history.'
                + (lambda m: f' With this many trials the test would detect an edge of about {m * 100:.1f}% or more (in 4 cases out of 5); a smaller one cannot be told apart from luck.'
                   if m == m else "")(float(np.nanmedian([r.get("mde", np.nan) for r in h["table"] if r["model"] not in ("naive", "ens", "eq")] or [np.nan])))
                + '</div></div>')
    return out


def regions_html(ens, df, res):
    last = lambda c: df[c].dropna().iloc[-1] if c in df and df[c].notna().any() else None
    tiles = []
    kim = df["btckrw"] / df["usdkrw"] / df["close"] - 1 if {"btckrw", "usdkrw"} <= set(df.columns) else None
    if kim is not None and kim.notna().any():
        k = kim.dropna()
        tiles.append(("Is Bitcoin more expensive in Korea than in the rest of the world?", f"{k.iloc[-1]*100:+.1f}%", f"average over the last 90 days: {k.tail(90).mean()*100:+.1f}%",
                      "A positive value is the so-called kimchi premium: Korean investors are buying very eagerly."))
    if last("dvol") is not None:
        vrp = ens["features_now"].get("dvol_vrp")
        tiles.append(("How big do investors expect price swings to be?", f"{last('dvol'):.0f}", "DVOL index from the options exchange, annualised",
                      ("Investors are paying more than usual for insurance against swings." if vrp is not None and vrp > 0.15 else "Insurance against swings costs about what recent moves would suggest." if vrp is not None else "")))
    if last("pcr") is not None:
        tiles.append(("Bets on a fall versus bets on a rise", f"{last('pcr'):.2f}", "ratio of bets on a fall to bets on a rise",
                      "Below 1 = bets on a rise dominate, above 1 = bets on a fall."))
    if last("cot_lev") is not None:
        tiles.append(("Large funds on the CME exchange", f"{last('cot_lev')*100:+.0f}%", "their net position relative to the whole market",
                      "A negative value often means they are trading a price gap, not betting on a fall."))
    tiles.append(("Is the market “nervous” right now?", f"{ens['hmm']['p_hi']*100:.0f}%", "chance that a period of big swings is under way", "The model recognises this purely from past data."))
    pz = ens["powerlaw_z"]
    tiles.append(("Price against the long-term trend", ("below" if pz < -0.3 else "above" if pz > 0.3 else "close to it"), f"deviation: {pz:+.1f}" + " (in typical units)",
                  "Below the trend line the price has often been an opportunity, and far above it a sign of overheating."))
    tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b}</b><span class="muted sm">{c}</span><span class="muted sm" style="opacity:.85">{d}</span></div>' for a, b, c, d in tiles)
    rows = ""
    for k, name in LANG.items():
        c = f"wiki_{k}"
        if c not in df:
            continue
        s = df[c].dropna()
        if len(s) < 400:
            continue
        v = float(np.log(s.tail(30).mean() / s.tail(365).mean()))
        w = min(abs(v), 1.0) * 50
        rows += (f'<div class="sig sm-row"><div class="sig-l"><b>{name}</b></div><div class="dv"><i class="mid"></i>'
                 f'<span class="{"g" if v >= 0 else "r"}" style="left:{50 if v >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div>'
                 f'<div class="sig-v"><b>{(math.exp(v) - 1) * 100:+.0f}%</b></div></div>')
    return tl, rows


def _catlab(c):
    import events_data
    return events_data.CATEGORIES.get(c, c)


_WL_CAT = {"biz": "Economy", "pol": "Politics", "armed": "Conflicts", "intl": "International relations", "disaster": "Disasters", "law": "Law and crime",
           "health": "Health", "sci": "Science and technology"}


def _move_row(m, cache):
    import news
    import worldlog
    d = m["date"]
    chips = "".join(f'<span class="cause {x["kind"]}" title="{html.escape(x["title"])} ({x["date"]})">{html.escape(_catlab(x["cat"]))}</span>' for x in m["causes"])
    nd = news.for_day(cache, d)
    wh = worldlog.day_headlines(d)
    detail = ""
    if nd and nd["cats"]:
        chips += "".join(f'<span class="cause news" title="This many Bitcoin headlines from that day were about this topic">{html.escape(_catlab(c["cat"]))} ×{c["n"]}</span>' for c in nd["cats"])
    if nd and nd["g"]:
        detail += '<div class="hl"><b>Bitcoin news headlines from that day</b><ul>' + "".join(
            f'<li>{html.escape(x["t"])} <span class="muted">({html.escape(x["d"])})</span></li>' for x in nd["g"][:5]) + "</ul></div>"
    if wh:
        detail += '<div class="hl"><b>What was happening in the world that day</b><ul>' + "".join(
            f'<li><span class="muted">{_WL_CAT.get(x["c"], x["c"])}:</span> {html.escape(x["t"])}</li>' for x in wh) + "</ul></div>"
    cnt = (len(nd["g"][:5]) if nd and nd["g"] else 0) + len(wh)
    cell = chips or '<span class="muted sm">no events from our list</span>'
    if detail:
        cell += f'<details class="hl-d"><summary>Show what happened ({cnt})</summary>{detail}</details>'
    return f'<tr><td>{d}</td><td class="{"pos" if m["ret"] >= 0 else "neg"}"><b>{m["ret"]:+.1f}%</b></td><td>{cell}</td></tr>'


def moves_html(E):
    import news
    cache = news.load()
    head = '<table class="tbl"><tr><th>Date</th><th>Price change</th><th>What was happening (events and headlines)</th></tr>'
    first = "".join(_move_row(m, cache) for m in E["moves"][:10])
    rest = "".join(_move_row(m, cache) for m in E["moves"][10:50])
    c = E["cover"]
    note = (f'We analysed {c["n_moves"]} big price jumps in history. For each one we record what was happening in the world: events from our list, '
            f'news headlines about Bitcoin (from the GDELT archive) and the Wikipedia log of world events. We do not claim that this caused the jump. '
            f'The point is to have a dense timeline of the world and look for repeating patterns on it (see “Searching for links”). '
            f'For comparison: even for randomly chosen days, some event from the list turns up in {c["placebo_any"]*100:.0f}% of cases.')
    more = f'<details class="more"><summary>Show more jumps</summary>{head}{rest}</table></details>' if rest else ""
    return f'{head}{first}</table>{more}<p class="muted sm" style="margin-top:10px">{note}</p>'


def study_html(E):
    t = E["table"]
    t = t[(t["window"] == 3) & (t["n"] >= 5)].copy()
    t["abs"] = t["t"].abs()
    t = t.sort_values("abs", ascending=False).head(16)
    rows = ""
    for r in t.itertuples():
        w = min(abs(r.t) / 5, 1) * 50
        sig = abs(r.t) >= 2 and r.n >= 8
        rows += (f'<tr class="{"sigrow" if sig else ""}"><td>{html.escape(_catlab(r.cat))}</td><td>{r.n}</td>'
                 f'<td class="{"pos" if r.mean_ret >= 0 else "neg"}">{r.mean_ret:+.1f}%</td><td>{r.pos*100:.0f}%</td>'
                 f'<td><div class="dv"><i class="mid"></i><span class="{"g" if r.t >= 0 else "r"}" style="left:{50 if r.t >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div></td>'
                 f'<td>{_stars(r.t, r.n)}</td></tr>')
    return (f'<table class="tbl"><tr><th>Type of event</th><th>How many times</th><th>Average price change after 3 days</th><th>How often the price rose</th><th></th><th>Confidence in the result</th></tr>{rows}</table>'
            f'<p class="muted sm" style="margin-top:10px">We measure the price change from the end of the day before the event to the end of the third day after it. Bold rows have high or medium confidence. '
            f'When there are many types of events, a few results come out “confident” by pure luck. '
            f'For events detected automatically from markets (a stock drop, a jump in stock market fear, the dollar and so on), these are mostly moves <b>on the same days</b>, not proof that Bitcoin reacts later.</p>')


FEAT_LAB = {"comp": "Overall market score", "val": "Is Bitcoin cheap (valuation)", "macro": "The economy and money around the world",
            "sent": "Mood and borrowed money", "cycle": "Place in the 4-year cycle", "mom30": "Price change over 30 days", "mom90": "Price change over 90 days",
            "volpct": "How big the swings are right now", "powerlaw": "Price against the long-term trend", "hmm_hi": "Is the market “nervous”",
            "hmm_mu": "Direction suggested by the regime model", "ev_eqdn30": "Stock market drops in the last 30 days", "ev_shock30": "Market shocks (30 days)",
            "ev_hack30": "Exchange hacks (30 days)", "stress": "Stock market fear", "fomc_soon": "Fed decision within 3 days",
            "opex_soon": "Contracts expiring within 3 days", "kimchi": "Kimchi premium (Korea)", "wiki_west": "Interest in the West",
            "wiki_asia": "Interest in Asia", "dvol_vrp": "Cost of insurance against swings", "cot": "Positions of large funds",
            "hurst": "Tendency of moves to continue", "pent": "Patterns in the order of moves", "vr5": "Price pullbacks within 5 days",
            "acabs": "Streaks of big moves", "hill": "Frequency of extreme moves", "kalman": "Price trend"}


def _level(x):
    a = abs(x)
    if a < 0.4:
        return "normal"
    word = "high" if x > 0 else "low"
    return ("fairly " if a < 1.2 else "very ") + word


def why_html(res, ens, E, df):
    fc = {f["horizon"]: f for f in res["forecasts"]}
    p0 = res["price"]
    sig = ens["sigma"]
    cards = []

    rows = "".join(f'<tr><td>{H} days</td><td>±{h["sigma_H"]*100:.0f}%</td><td>{(fc[H]["p10"]/p0-1)*100:+.0f}% … {(fc[H]["p90"]/p0-1)*100:+.0f}%</td></tr>'
                   for H, h in ens["horizons"].items())
    dv = df["dvol"].dropna().iloc[-1] if "dvol" in df and df["dvol"].notna().any() else None
    dvt = ""
    if dv is not None:
        impl = dv / math.sqrt(365)
        cmpt = ("Investors expect more movement than recent days would suggest." if impl > sig * 100 * 1.08 else
                "Investors expect less movement than recent days would suggest." if impl < sig * 100 * 0.92 else "That is roughly the same as recent days suggest.")
        dvt = (f'<li>On the options exchange (a market for “insurance” against price changes) investors are pricing daily swings at about <b>{impl:.1f}%</b>, while our model calculates <b>{sig*100:.1f}%</b>. {cmpt}</li>')
    cards.append(("Where does the width of the range come from?",
                  f'<ul><li>Lately Bitcoin\'s price has been changing by about <b>{sig*100:.1f}%</b> a day. We calculate this in three independent ways and combine them, '
                  f'choosing the weight of each way according to how well it worked in the past.</li>'
                  f'<li>The longer the period, the greater the uncertainty (but it grows more slowly than time). The shape of the distribution, that is, how often big jumps happen, we take from Bitcoin\'s real history, '
                  f'with more emphasis on the last two years.</li>{dvt}'
                  f'<li>According to our model, the chance that the market is now in a “nervous” mode of big swings is <b>{ens["hmm"]["p_hi"]*100:.0f}%</b>.</li>'
                  f'<li>The ranges are calibrated on history so that the price stays within them in about 8 cases out of 10.</li></ul>'
                  f'<table class="tbl"><tr><th>How far ahead</th><th>Typical swing</th><th>Price range vs today (8 out of 10)</th></tr>{rows}</table>'))

    lines = ""
    for H, h in ens["horizons"].items():
        cand = [r for r in h["table"] if r["model"] not in ("naive", "ens")]
        b = max(cand, key=lambda r: r["skill"])
        p = b["dm_p"]
        verdict = "The edge is probably real." if (p == p and p <= 0.05) else ("There may be something to it, but it is uncertain." if (p == p and p <= 0.20) else "There is no evidence that it works.")
        mname = MODEL_LAB.get(b["model"], b["label"])
        better = "better" if b["skill"] > 0 else "worse"
        lines += (f'<li><b>In {H} days:</b> the best method ({mname[0].lower() + mname[1:]}) was {abs(b["skill"])*100:.1f}% {better} than guessing; '
                  f'chance that it is luck: {p*100:.0f}%. {verdict}</li>')
    allzero = all(h["alpha"] == 0 for h in ens["horizons"].values())
    cards.append(("Why don't we say “the price will rise” or “fall”?",
                  f'<ul>{lines}</ul><p class="muted sm">We tested every method on history as if it had been used in real time: on each day it only knew what was known then, '
                  f'and it guessed the future. We compare it with the simple assumption “the price stays the same”. A method is only allowed into the forecast when the chance that its edge is luck is below 10%. '
                  f'{"No method met this condition, so the middle of the forecast is today&#39;s price." if allzero else "Where the condition was met, the middle of the forecast is shifted."} '
                  f'This is a common result for Bitcoin: researchers agree that the direction of the price over months is very hard to predict, and what can be judged reliably is mainly the size of the swings.</p>'))

    H = 90 if 90 in ens["horizons"] else list(ens["horizons"])[0]
    ex = ens["horizons"][H]["explain"]
    drv = ""
    for name, c, x in ex.get("ridge", [])[:6]:
        eff = c * ens["horizons"][H]["sigma_H"] * 100
        drv += (f'<tr><td>{FEAT_LAB.get(name, name)}</td><td>{_level(x)}</td><td class="{"pos" if eff >= 0 else "neg"}">{eff:+.1f}% ({"raises" if eff >= 0 else "lowers"})</td></tr>')
    view = fc[H]["view"]
    sg = sorted([p for p in res["signals"] if p["score"] is not None and p["weight"] > 0], key=lambda p: -(p["score"] * p["weight"]))
    low = lambda t: t[0].lower() + t[1:]
    pos = ", ".join(low(LABELS[p["key"]]) for p in sg[:3] if p["score"] > 0) or "none clearly"
    neg = ", ".join(low(LABELS[p["key"]]) for p in sg[::-1][:3] if p["score"] < 0) or "none clearly"
    cards.append((f"What the indicators suggest ({H}-day scenario, not a forecast)",
                  f'<p>If we trusted the indicators, the middle of the forecast would be <b>{n(view["p50"])} USD</b> ({(view["p50"]/p0-1)*100:+.1f}% vs today), and the chance of a rise {view["prob_up"]*100:.0f}%. The biggest contributors:</p>'
                  f'<table class="tbl"><tr><th>Factor</th><th>Where it is today</th><th>Effect on the forecast</th></tr>{drv}</table>'
                  f'<p class="muted sm" style="margin-top:8px">Pointing most strongly to a rise: {pos}. Most strongly to a fall: {neg}. '
                  f'Because this method did not pass the test on history, treat it as an interesting scenario, not a forecast.</p>'))

    kn = ex.get("knn", [])
    ka = ex.get("knn_all")
    if kn:
        rows = "".join(f'<tr><td>{d}</td><td class="{"pos" if r >= 0 else "neg"}">{r:+.0f}%</td></tr>' for d, r, _ in kn)
        kskill = [r for r in ens["horizons"][H]["table"] if r["model"] == "knn"][0]["skill"] * 100
        cards.append((f"Similar days from the past (what happened {H} days later)",
                      f'<p>We look for days in history when the situation was similar to today (market score, price changes, swings, place in the cycle). '
                      f'Of the {ka["n"]} most similar days, <b>{ka["up"]*100:.0f}%</b> ended with a rise, and the typical change was <b>{ka["median_ret"]:+.0f}%</b>. '
                      f'<b>Note:</b> these days come from only {ka.get("episodes", "?")} separate periods (consecutive days of one rally are not independent examples), '
                      f'and in the test on history this method came out {abs(kskill):.0f}% {"better" if kskill > 0 else "worse"} than guessing. Treat it as an illustration, not proof.</p>'
                      f'<table class="tbl"><tr><th>Most similar days (at least 30 days apart)</th><th>Price change after {H} days</th></tr>{rows}</table>'))

    ev = E["events"]
    last = df.index[-1]
    rec = ev[(ev["date"] > last - pd.Timedelta(days=30)) & (ev["date"] <= last) & (~ev["cat"].isin(["nfp", "opex", "fomc"]))]
    recl = "".join(f'<li>{r.date:%Y-%m-%d}: {html.escape(r.title)}</li>' for r in rec.itertuples()) or "<li>In the last 30 days there were no big shocks or events from our list.</li>"
    up = "".join(f'<li>{u["date"]}: {html.escape(u["title"])}' + (f' <span class="muted">(after 3 days the price usually changed by {u["hist"]["mean_ret"]:+.1f}%, based on {u["hist"]["n"]} similar events)</span>' if u["hist"] else "") + "</li>"
                 for u in E["upcoming"][:4])
    cards.append(("What is happening in the world",
                  f'<p style="margin-bottom:4px"><b>Last 30 days</b></p><ul>{recl}</ul><p style="margin:10px 0 4px"><b>Coming up</b></p><ul>{up}</ul>'
                  f'<p class="muted sm">We also feed such events to our methods as an extra hint. They will only start to affect the forecast once the test on history shows that this helps.</p>'))

    cov = " · ".join(f'{H} days: {[r for r in h["table"] if r["model"] == "ens"][0]["cover80"]*10:.1f}' for H, h in ens["horizons"].items())
    cards.append(("How far can you trust this?",
                  f'<ul><li><b>Width of the range: you can trust it.</b> In tests on history the price stayed within the given range in this many cases out of 10 ({cov}). The target is 8.</li>'
                  + ('<li><b>Direction of the move: you cannot trust it.</b> There is no evidence that we can predict whether the price will go up or down, so the chance of a rise is about 50%.</li>'
                     if allzero else
                     '<li><b>Direction of the move: partly.</b> For some periods our methods turned out better than guessing, but the sample is small, so do not treat it as a certainty.</li>') +
                  '<li><b>Sudden, rare events</b> (an exchange collapse, a pandemic) can be stronger than any forecast. We have data from only four full cycles, and the market is changing (ETFs, companies buying Bitcoin).</li>'
                  '<li>The system updates itself and checks its own past forecasts (section below).</li></ul>'))
    return "".join(f'<details class="card why"{" open" if i < 1 else ""}><summary><h3 class="ct">{t}</h3></summary><div class="why-b">{b}</div></details>'
                   for i, (t, b) in enumerate(cards))


def conclusion_text(ens):
    act = [H for H, h in ens["horizons"].items() if h["alpha"] > 0]
    off = [H for H, h in ens["horizons"].items() if h["alpha"] == 0]
    if not act:
        s = ("Conclusion for today: none of the methods is significantly better than the assumption “the price stays the same”, for any of the periods. "
             "That is why the main forecast is based on how much the price usually swings, and we show the scenario from the indicators separately, as a curiosity.")
    else:
        s = (f"Conclusion for today: over {', '.join(map(str, act))} days our methods turned out significantly better than guessing and shift the middle of the forecast "
             f"(with caution: over long periods the examples from history overlap heavily). "
             + (f"Over {', '.join(map(str, off))} days the forecast is based on typical price swings." if off else ""))
    return s + (" A chance of a rise of about 50% is a neutral assumption: historically Bitcoin rose in about 60% of 30-day windows, but taking that into account did not improve the test results, "
                "and past gains need not repeat. The system updates itself, so the weights of the methods change as new data comes in.")


def _pctbar(p):
    return f'<div class="pbar"><i style="left:{p*100:.0f}%"></i></div><div class="range-l sm muted"><span>rare</span><span>higher than {p*100:.0f}% of history</span><span>common</span></div>'


def regularity_html(Rg):
    nw, pc = Rg["now"], Rg["pct"]
    h, pe, vr, aa, hl = nw["hurst"], nw["pent"], nw["vr5"], nw["acabs"], nw["hill"]
    mk, kal, evt = Rg["markov"], Rg["kalman"], Rg["evt"]
    cyc = ", ".join(f'{p:.0f} days' for p, s in Rg["cycles"])
    tiles = [
        ("Do price moves tend to continue?", "Hurst exponent", f"{h:.2f}", pc["hurst"],
         ("A value of 0.5 is pure chance. Here: moves have a slight tendency to continue." if h > 0.53 else "A value of 0.5 is pure chance. Here: moves more often reverse." if h < 0.47 else "A value of 0.5 is pure chance, and that is exactly what we see.")),
        ("Are there any patterns in the order of moves?", "permutation entropy", f"{pe:.3f}", pc["pent"],
         "A value of 1 means no patterns at all (pure chance). That is the case right now." if pe > 0.98 else "Below 1: repeating patterns are appearing in the order of moves."),
        ("Does a rise tend to be followed by a fall?", "variance ratio", f"{vr:+.2f}", pc["vr5"],
         "Negative value: moves over 5 days slightly reverse. Close to zero: pure chance." if vr < 0 else "Positive value: moves over 5 days slightly build on each other."),
        ("Do big moves come in streaks?", "volatility clustering", f"{aa:+.2f}", pc["acabs"],
         "Yes, clearly: a big move is usually followed by more. This is the part of the market that can be forecast." if aa > 0.15 else "Weakly: big moves happen fairly independently."),
        ("How often do extreme moves happen?", "tail index", f"{hl:.1f}", pc["hill"],
         "A value of around 3 means extreme moves are much more common than in a “textbook” distribution." if hl < 4 else "Extreme moves are happening less often than usual over the last year."),
    ]
    tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b2}</b><span class="muted sm" style="margin-top:-2px">technical name: {tech}</span>{_pctbar(c)}<span class="muted sm">{d}</span></div>'
                 for a, tech, b2, c, d in tiles)
    nx = mk["next"]
    names = ["a fall", "no major change", "a rise"]
    weeks = ["down", "flat", "up"]
    ev = (f'<div class="card tile"><span class="muted sm">How bad can a single day be?</span><b>−{evt["var990"]*100:.1f}%</b>'
          f'<span class="muted sm">A daily drop this big happens roughly once in 100 days (a standard bell-curve distribution would say only −{evt["gauss99"]*100:.1f}%). '
          f'Once in 1,000 days: −{evt["var999"]*100:.1f}%.</span></div>') if evt else ""
    extra = (f'<div class="card tile"><span class="muted sm">What usually happens after the last week?</span><b>{max(nx)*100:.0f}% chance of {names[int(np.argmax(nx))]}</b>'
             f'<span class="muted sm">After a “{weeks[mk["now"]]}” week, the next week ended with a fall in {nx[0]*100:.0f}%, no major change in {nx[1]*100:.0f}%, and a rise in {nx[2]*100:.0f}% of cases. '
             f'{"The link is weak, but possible." if mk["p_dep"] < 0.1 else "There is practically no link here."}</span></div>'
             f'<div class="card tile"><span class="muted sm">Does the price have a clear trend right now?</span><b>{kal["slope_w"]*100:+.2f}% per week</b>'
             f'<span class="muted sm">The error of this estimate is ±{kal["sd"]*100:.1f}%, so {"the trend is not different from zero." if abs(kal["z"]) < 2 else "the trend is clear."}</span></div>'
             f'<div class="card tile"><span class="muted sm">Does volatility have a repeating rhythm?</span><b>{Rg["cycles"][0][0]:.0f} days</b>'
             f'<span class="muted sm">Strongest rhythms: {cyc}. The 7-day rhythm is simply the day-of-week effect. Longer rhythms are uncertain, because we only have about 15 years of data.</span></div>')
    dir_random = 0.47 <= h <= 0.57 and pe > 0.97 and abs(vr) < 0.1
    vol_struct = aa > 0.12
    verdict = (("The direction of price moves is close to pure chance. " if dir_random else
                "There are some patterns in the direction of moves, but they need to be confirmed by a test on history. ")
               + ("The size of the swings, on the other hand, has a clear structure (big moves come in streaks, there is a weekly rhythm, extreme jumps happen), which is why the system mainly forecasts the price range and the risk. "
                  if vol_struct else "Streaks of big moves are weak right now. ")
               + "All measures are calculated from the last year, so they change over time.")
    return tl + ev + extra, verdict


def analogs_html(A):
    if not A or not A.get("rows"):
        return ""
    rows = "".join(f'<tr><td>{r["date"]}</td><td class="{"pos" if r["f30"] >= 0 else "neg"}">{r["f30"]:+.0f}%</td><td class="{"pos" if r["f90"] >= 0 else "neg"}">{r["f90"]:+.0f}%</td></tr>' for r in A["rows"])
    return (f'<div class="panel" style="margin-bottom:18px"><h3 class="ct">When did the world look similar to today?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">We describe the world with a profile of the last 30 days: how many entries in the world events log were about wars, markets, banks, sanctions, elections, disasters, pandemics, crypto and other topics ({A["n_feat"]} features). '
            f'We look for periods with the most similar profile and see what Bitcoin did next. This is a curiosity, not a forecast: it is only a handful of separate episodes.</p>'
            f'<table class="tbl"><tr><th>Period (end of the 30-day window)</th><th>Price after 30 days</th><th>Price after 90 days</th></tr>{rows}</table>'
            f'<p class="muted sm" style="margin:10px 0 0">In {A["up30"]*100:.0f}% of these periods the price was higher after 30 days; the typical change was {A["med30"]:+.0f}% after 30 days and {A["med90"]:+.0f}% after 90 days. '
            f'Bitcoin rose in about 60% of all 30-day windows in its history, so this alone proves nothing yet.</p></div>')


def scan_html(S):
    if S is None:
        return '<div class="card empty"><b>No data from the world</b></div>', ""
    s = S["summary"]
    nt = f"{s['n_tests']:,}"
    head = (f'<div class="card"><p style="margin:0 0 6px">We checked <b>{nt} possible links</b> between Bitcoin and {s["n_series"]} indicators from around the world: '
            f'stock markets and currencies of more than 20 countries, oil, gold and food, interest in politics and crises (Wikipedia page views), the weather where Bitcoin is mined, '
            f'sunspots, earthquakes, phases of the Moon and the season of the year. We look at whether what happens today predicts a change in Bitcoin\'s price in a week or a month.</p>'
            f'<p class="muted sm" style="margin:0 0 8px">When you check this many things, some will look connected purely by luck (like winning the lottery when you buy thousands of tickets). '
            f'That is why we use three filters: a correction for the number of tests, a check that the link holds in both halves of history, and a test on the most recent data.</p>'
            f'<div class="funnel"><div><b>{s["raw_p05"]}</b><span>looked connected</span></div><div><b>~{s["expected_fp"]:.0f}</b><span>how many luck alone would produce</span></div>'
            f'<div><b>{s["n_fdr"]}</b><span>left after filter 1</span></div><div><b>{s["n_stable"]}</b><span>after filter 2</span></div>'
            f'<div><b>{s["n_disc"]}</b><span>after filter 3 (real discoveries)</span></div></div></div>')

    def rows(df):
        out = ""
        for r in df.itertuples():
            out += (f'<tr><td>{html.escape(r.label)}</td><td>{r.h} days</td><td>{r.lag} days</td><td class="{"pos" if r.r >= 0 else "neg"}">{r.r:+.2f}</td>'
                    f'<td>{r.p*100:.1f}%</td><td class="{"pos" if r.r_test >= 0 else "neg"}">{r.r_test:+.2f}</td></tr>')
        return out
    th = ('<tr><th>Indicator</th><th>Price change over</th><th>Indicator from</th><th title="from −1 to +1">Strength of the link</th>'
          '<th title="Chance that it is luck, without taking the number of tests into account (after correction it would be much higher)">Chance of luck (uncorrected)</th><th title="Strength of the link in the most recent data">In newer data</th></tr>')
    if len(S["discoveries"]):
        d = f'<table class="tbl">{th}{rows(S["discoveries"])}</table>'
    else:
        d = ('<p class="muted" style="margin:0">No link passed all three filters. That is a normal result: with such a broad search '
             'most “discoveries” are luck. The system keeps searching with every update and will let you know if something holds up.</p>')
    c = f'<table class="tbl">{th}{rows(S["candidates"])}</table>'
    co = ""
    for df_ in (S["comove_pos"], S["comove_neg"]):
        for r in df_.itertuples():
            w = min(abs(r.r), 0.6) / 0.6 * 50
            co += (f'<div class="sig sm-row"><div class="sig-l"><b>{html.escape(r.label)}</b></div><div class="dv"><i class="mid"></i>'
                   f'<span class="{"g" if r.r >= 0 else "r"}" style="left:{50 if r.r >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div>'
                   f'<div class="sig-v"><b>{r.r:+.2f}</b></div></div>')
    an = analogs_html(S.get("analogs"))
    return head, an + (f'<div class="two2"><div class="card"><h3 class="ct">Discoveries that passed all the filters</h3>{d}'
                  f'<h3 class="ct" style="margin-top:18px">Most interesting suspects (did not pass the filters, but we are watching them)</h3>{c}'
                  f'<p class="muted sm" style="margin-top:8px">The strength of the link is a number from −1 to +1 (0 = no link). These suspects hold up over time and also work in newer data, '
                  f'but once the number of tests is taken into account, luck cannot be ruled out.</p></div>'
                  f'<div class="card"><h3 class="ct">What Bitcoin moves together with in the same week</h3>'
                  f'<p class="muted sm" style="margin:0 0 8px">This is a description, not a forecast: plus = they rise together, minus = they move in opposite directions.</p>{co}</div></div>')


def upcoming_html(E):
    rows = ""
    for u in E["upcoming"][:10]:
        h = u["hist"]
        hist = (f'usually {h["mean_ret"]:+.1f}% after 3 days, the price rose in {h["pos"]*100:.0f}% of cases (based on {h["n"]} similar ones)' if h else "no history")
        rows += f'<tr><td>{u["date"]}</td><td>{html.escape(u["title"])}</td><td class="muted">{hist}</td></tr>'
    return f'<table class="tbl"><tr><th>Date</th><th>Event</th><th>How Bitcoin usually reacted</th></tr>{rows}</table>'


def touch_html(TC):
    """Chance of touching price levels within 7, 30 and 90 days, plus a backtest of that chance."""
    if not TC:
        return ""
    pc = lambda p: "&lt;1%" if p < 0.01 else ("&gt;99%" if p > 0.99 else f"{p * 100:.0f}%")
    rows = ""
    for r in TC["table"]:
        if abs(r["rel"]) < 0.02:
            continue
        lab = f'{r["level"] / 1000:.0f}k USD'
        rows += (f'<tr><td><b>{lab}</b> <span class="muted sm">({r["rel"] * 100:+.0f}%)</span></td>'
                 + "".join(f'<td>{pc(r[f"p{H}"])}</td>' for H in (7, 30, 90)) + "</tr>")
    c = TC["calib"]
    cr = "".join(f'<tr><td>{x["lo"] * 100:.0f}–{x["hi"] * 100:.0f}%</td><td>{x["pred"] * 100:.0f}%</td><td>{x["real"] * 100:.0f}%</td><td class="muted">{x["n"]}</td></tr>'
                 for x in c["rows"])
    import touch as TO
    pl = lambda v, d=1: f"{v:+.{d}f}"
    sk = c["skill"] * 100
    side = c.get("side", {})
    best = TC.get("best", "base")
    sidetxt = ""
    if side:                                            # the largest systematic error of the chosen variant
        x = max(side, key=lambda k: abs(side[k][0] - side[k][1]))
        p_, r_ = side[x]
        if abs(p_ - r_) > 0.03:
            co = f'a {"rise" if float(x) > 0 else "fall"} of {abs(float(x)) * 100:.0f}% within a month'
            sidetxt = (f' Biggest weakness: for {co} the model predicted on average {p_ * 100:.0f}%, but it happened in {r_ * 100:.0f}% of months, '
                       f'so it tends to {"overstate" if p_ > r_ else "understate"} this chance.')
    var_rows = "".join(f'<tr><td>{"<b>" if m == best else ""}{TO.LABELS[m]}{"</b> (in use)" if m == best else ""}</td>'
                       f'<td class="{"pos" if v["skill"] > 0 else "neg"}">{pl(v["skill"] * 100)}%</td></tr>'
                       for m, v in TC.get("variants", {}).items())
    drift_note = (' The chosen variant assumes that the trend of the last 4 years (the average daily price move) will continue, which is why the chances of rises are higher here than of falls. '
                  'In the backtest this assumption improved accuracy, but Bitcoin rose strongly for most of its history, and the future need not repeat that.'
                  if "drift" in best else ' The direction is neutral: a rise and a fall of the same percentage are equally likely.')
    decomp = (f' Breaking the result down: the model is better at telling risky situations from calm ones (resolution {c["resolution"]:.4f} vs {c["resolution_base"]:.4f} '
              f'for the fixed frequency), and its chances also match reality better (calibration error {c["reliability"]:.4f} vs {c["reliability_base"]:.4f}).'
              ) if "resolution" in c else ""
    hl = TC.get("definition", "close") == "hl"
    od = TC.get("other_definition") or {}
    ob = (od.get("variants") or {}).get(od.get("best"), {})
    defn = ('We count a touch the way people understand it: it is enough for the price to reach the level at least once during the day (the day\'s high or low), '
            'not only at the close. The simulation draws whole real days from history, including how far they reached up and down. '
            if hl else 'We count a touch at the daily close (the price at midnight UTC); during the day the price may have reached the level more often. ')
    switch = ""
    if ob:
        switch = (f'<p class="muted sm" style="margin:8px 0 0">The definition is chosen by a rule written down in advance: counting by daily highs and lows replaces '
                  f'counting by closes if it beats the fixed frequency both before the Oct 29, 2024 cut-off and on the exam. '
                  + (f'It passed: an edge of {pl(c["skill_pre"] * 100)}% before and {pl(c["skill_post"] * 100)}% on the exam '
                     f'(for comparison, counting by closes: {pl(ob.get("skill_pre", 0) * 100)}% and {pl(ob.get("skill_post", 0) * 100)}%).'
                     if hl else 'It did not pass, so we stay with closes.') + '</p>')
    return (f'<div class="panel" style="margin-top:18px"><h3 class="ct">What is the chance the price will touch a given level?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">This is not about where the price ends up, but whether it touches the level at least once along the way. {defn}'
            f'We calculate this from 20,000 simulated price paths, using today\'s volatility, which returns to normal over time.{drift_note}</p>'
            + f'<div class="tw"><table class="tbl"><tr><th>Level</th><th>Within 7 days</th><th>Within 30 days</th><th>Within 90 days</th></tr>{rows}</table></div>'
            f'<h3 class="ct" style="margin-top:18px">Did these chances hold up?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">Every week since 2016 we calculated the chance of touching levels ±10% and ±20% within 30 days, knowing only the data up to that day, '
            f'and checked what happened ({c["n"]} trials).</p>'
            f'<div class="tw"><table class="tbl"><tr><th>When the model said</th><th>On average it said</th><th>What actually happened</th><th>Trials</th></tr>{cr}</table></div>'
            f'<p class="note"><b>Conclusion:</b> compared with the fixed historical frequency (e.g. “+10% within a month happens in roughly half of all months”), '
            f'calculated only from the past, the model is {pl(sk)}% more accurate (Brier score). It is a small but honest edge: '
            f'it comes from the fact that today\'s volatility says something about the next few weeks.{decomp}{sidetxt}</p>{switch}'
            + (f'<p class="muted sm" style="margin:10px 0 6px">We checked several ways of simulating with the same test; the table shows the edge of each over the fixed frequency. '
               f'Picking the best of several on the same test is slightly optimistic, which is why we show them all.</p>'
               f'<div class="tw"><table class="tbl"><tr><th>Simulation method</th><th>Edge</th></tr>{var_rows}</table></div>' if var_rows else "")
            + '</div>')


def exam_html(ens, TCH=None, SIM=None):
    """Exam on the 2 held-out years: choices made only on data before the cut-off, one check on the last 2 years."""
    if not ens:
        return ""
    pl = lambda v: "–" if v is None or v != v else f"{v * 100:+.1f}%"
    zl = lambda v: f"{v:,.0f} PLN"
    lab = lambda m: MODEL_LAB.get(m, "Team of methods (with a gate)" if m == "ens" else m)
    rows, start = "", None
    for H, v in sorted(ens["horizons"].items(), key=lambda a: int(a[0])):
        ho = v.get("holdout") or {}
        if not ho.get("chosen"):
            continue
        start = ho.get("from")
        r = ho["rows"][ho["chosen"]]
        p = r["p_post"]
        vd = ho.get("verdict", "passed" if ho["passed"] else "failed")
        verdict = {"passed": '<b class="pos">passed, in the forecast from today</b>',
                   "inconclusive": '<span class="muted">inconclusive (too few independent trials), not used</span>',
                   "failed": '<span class="muted">failed</span>'}[vd]
        mde = r.get("mde_post")
        neff = r.get("n_eff_post")
        rows += (f'<tr><td><b>{H} days</b></td><td>{html.escape(lab(ho["chosen"]))}</td><td>{pl(r["skill_pre"])}</td>'
                 f'<td class="{"pos" if r["skill_post"] > 0 else "neg"}">{pl(r["skill_post"])}</td>'
                 f'<td>{"–" if p != p else f"{p:.2f}"}</td>'
                 f'<td>{r["n_post"]}{f" (about {neff:.0f} independent)" if neff is not None and neff == neff else ""}</td>'
                 f'<td>{"–" if mde is None or mde != mde else pl(mde)}</td><td>{verdict}</td></tr>')
    out = (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Exam on the last 2 years</h3>'
           f'<p class="muted sm" style="margin:0 0 10px">We hold out the last 2 years (from {start or "2024-10-29"}). We choose the method, the variant and the strategy using only data from before that date, '
           f'and then check them once on the held-out period, which they “did not see” while being chosen. Only what beats the “no change” assumption '
           f'both before the cut-off and on the exam, with a small chance of luck (p &lt; 0.10), enters the forecast from today. We wrote this rule down before seeing the results. '
           f'We do not tweak the model until the exam comes out well: that would amount to memorising those two years.</p>')
    if rows:
        out += (f'<h3 class="ct" style="margin-top:6px">Forecast of the price range</h3>'
                f'<div class="tw"><table class="tbl"><tr><th>Horizon</th><th>Chosen method</th><th>Edge before</th><th>Edge on the exam</th><th>p</th><th>Trials</th><th>Smallest edge the test could detect</th><th>Result</th></tr>{rows}</table></div>'
                f'<p class="muted sm" style="margin:6px 0 0">Over long horizons the forecasts made within 2 years overlap heavily (for 180 days that is in practice only a few independent periods). '
                f'That is why, with fewer than 20 independent trials, we write “inconclusive” rather than “failed”, and we state how big an edge this exam could detect at all (in 4 cases out of 5). '
                f'The principle does not change: a method that cannot be checked does not enter the forecast. Since Oct 30, 2026 the tests also include a correction for a small number of trials (Harvey, Leybourne and Newbold 1997), '
                f'and the 1, 3 and 7-day horizons are checked every day instead of every week.</p>')
    if TCH and TCH.get("variants"):
        import touch as TO
        tr = "".join(f'<tr><td>{"<b>" if m == TCH["best"] else ""}{TO.LABELS[m]}{" (chosen)</b>" if m == TCH["best"] else ""}</td>'
                     f'<td>{pl(v.get("skill_pre"))}</td><td class="{"pos" if (v.get("skill_post") or 0) > 0 else "neg"}">{pl(v.get("skill_post"))}</td></tr>'
                     for m, v in TCH["variants"].items())
        out += (f'<h3 class="ct" style="margin-top:18px">Chances of touching levels</h3>'
                f'<div class="tw"><table class="tbl"><tr><th>Variant</th><th>Edge before</th><th>Edge on the exam</th></tr>{tr}</table></div>')
    ex = ((SIM or {}).get("backtest") or {}).get("exam")
    if ex:
        a, d = ex["active"], ex["dca"]
        out += (f'<h3 class="ct" style="margin-top:18px">Virtual portfolio</h3>'
                f'<div class="tw"><table class="tbl"><tr><th>From {ex["start"]}, deposits {zl(a["deposited"])}</th><th>Active portfolio</th><th>Regular buying</th></tr>'
                f'<tr><td>Value today</td><td><b>{zl(a["final"])}</b></td><td>{zl(d["final"])}</td></tr>'
                f'<tr><td>Annual return (excluding the effect of deposits)</td><td><b>{pl(a["twr_ann"])}</b></td><td>{pl(d["twr_ann"])}</td></tr>'
                f'<tr><td>Biggest drop along the way</td><td><b>{pl(a["max_dd"])}</b></td><td>{pl(d["max_dd"])}</td></tr></table></div>'
                f'<p class="muted sm" style="margin:6px 0 0">Caveat: we confirmed the trend rule with test H7 on the whole history, including these 2 years, so for the portfolio this is not a fully clean exam. '
                f'The clean check is the live portfolio, run since Oct 28, 2026.</p>')
    return out + "</div>"


def short_html(ens):
    """Short horizons (1 and 3 days): hundreds of times more independent trials, so a weak edge has a chance to show up in the test."""
    sh = (ens or {}).get("short")
    if not sh:
        return ""
    rows = ""
    for H, v in sorted(sh.items()):
        tb = {r["model"]: r for r in v["table"]}
        best = max((r for r in v["table"] if r["model"] not in ("naive", "ens")), key=lambda r: r["skill"])
        e = tb.get("ens", {})
        pv = best.get("dm_p")
        sig = (pv is not None and pv == pv and pv < 0.05)
        rows += (f'<tr><td><b>{H} {"day" if H == 1 else "days"}</b></td><td>{v["n"]}</td><td>{html.escape(best["label"])}</td>'
                 f'<td class="{"pos" if best["skill"] > 0 else "neg"}">{best["skill"] * 100:+.2f}%</td>'
                 f'<td class="{"pos" if sig else "muted"}">{"yes" if sig else "no"} (p = {pv:.2f})</td>'
                 f'<td>{e.get("cover80", 0) * 100:.0f}%</td></tr>')
    return (f'<div class="panel" style="margin-top:18px"><h3 class="ct">And over the short run: 1 and 3 days?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">The shorter the horizon, the more independent trials there are in history, so the easier it is to detect even a small edge. '
            f'Here we check the best method against the “no change” assumption. These horizons do not go into the main forecast until the edge is significant.</p>'
            f'<div class="tw"><table class="tbl"><tr><th>Horizon</th><th>Trials</th><th>Best method</th><th>Edge</th><th>Significant?</th><th>Within the 80% band</th></tr>{rows}</table></div></div>')


def dirtrack_html(DT):
    """Separate direction track: chance of a rise within 7 days from confirmed signals, judged only by the Brier score."""
    if not DT or not DT.get("eval"):
        return ""
    pl = lambda v: f"{v * 100:+.2f}%"
    pc = lambda v: f"{v * 100:.0f}%"
    rows = "".join(f'<tr><td>{lab}</td><td>{e["n"]}</td><td>{pc(e["up_rate"])}</td><td>{pc(e["hit"])}</td>'
                   f'<td class="{"pos" if e["skill"] > 0 and e["p"] < 0.10 else "muted"}">{pl(e["skill"])}</td><td>{e["p"]:.2f}</td></tr>'
                   for k, lab in (("przed", "before Oct 29, 2024"), ("po", "exam: from Oct 29, 2024")) if (e := DT["eval"].get(k)))
    now = (f' Today\'s reading of this track: {pc(DT["p_now"])} chance of a rise within 7 days (fixed frequency: {pc(DT["clim_now"])}).' if DT.get("p_now") is not None else "")
    verdict = ("The track passed the test, so its chance replaces the 50% on the panel." if DT["passed"] else
               "The track did not pass the test: its edge over the fixed frequency of rises is too small to tell apart from luck. That is why the panel still shows 50/50.")
    return (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Separate direction track: will the price be higher in a week?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">The main test judges the whole price range, and a weak direction signal moves it so little that it cannot pass the test. '
            f'Here we check only one question: a rise or a fall in 7 days. The chance is calculated by a simple model from confirmed signals (trend), trained every week on past data only. '
            f'The rival is the fixed frequency of rises from history. Rule written down on Oct 31, 2026: the track replaces the 50% only if it beats the rival with p &lt; 0.10 both before the cut-off and on the exam.</p>'
            f'<div class="tw"><table class="tbl"><tr><th>Period</th><th>Days</th><th>Rising weeks</th><th>Direction right</th><th>Edge (Brier)</th><th>p</th></tr>{rows}</table></div>'
            f'<p class="note"><b>Conclusion:</b> {verdict}{now}</p></div>')


def judge_html(JD):
    """Judge from betting markets: our touch chances versus prices on Polymarket (backtest) and Kalshi (live)."""
    if not JD:
        return ""
    pc = lambda v: "–" if v is None or v != v else f"{v * 100:.0f}%"
    f3 = lambda v: "–" if v is None or v != v else f"{v:.3f}"
    out = ('<div class="panel" style="margin-top:18px"><h3 class="ct">Who is better at guessing touches: us or the betting market?</h3>'
           '<p class="muted sm" style="margin:0 0 10px">On prediction markets (Polymarket, Kalshi) people bet real money on whether bitcoin will touch a given price by the end of the month. '
           'The price of such a bet is the chance in the eyes of the market. We compare it with ours, calculated from data of the same day, and with what actually happened '
           '(Brier score: the lower, the better; guessing 50/50 gives 0.25).</p>')
    P = (JD.get("poly") or {}).get("eval") or {}
    if P:
        rows = "".join(f'<tr><td>{lab}</td><td>{e["n"]} ({e["months"]} months)</td><td>{f3(e["brier_model"])}</td><td>{f3(e["brier_market"])}</td><td>{f3(e["brier_blend"])}</td></tr>'
                       for k, lab in (("przed", "before Oct 29, 2024"), ("po", "exam: from Oct 29, 2024"), ("calosc", "all")) if (e := P.get(k)))
        c = P.get("calosc") or {}
        better = c and c["brier_market"] < c["brier_model"]
        out += (f'<div class="tw"><table class="tbl"><tr><th>Polymarket, settled markets</th><th>Markets</th><th>Our model</th><th>Market</th><th>Half and half</th></tr>{rows}</table></div>'
                + (f'<p class="note"><b>Conclusion:</b> the betting market guesses touches better than our model, especially over the last 2 years. '
                   f'If you want to know whether the price will touch a specific level by the end of the month, the Polymarket price is a better guide than our table; '
                   f'treat ours as a reference point. Markets from the same month share the same price path, so there are only as many independent comparisons as there are months.</p>'
                   if better else '<p class="note"><b>Conclusion:</b> our model does no worse than the betting market; with so few months this is not proof yet.</p>'))
    K = JD.get("kalshi") or {}
    if K.get("now"):
        kr = "".join(f'<tr><td>{"above" if r["up"] else "below"} {r["strike"] / 1000:.1f}k USD</td><td>by {r["close"]}</td><td>{pc(r["p_market"])}</td><td>{pc(r["p_model"])}</td></tr>'
                     for r in K["now"])
        out += (f'<h3 class="ct" style="margin-top:18px">Right now: Kalshi versus our model</h3>'
                f'<p class="muted sm" style="margin:0 0 10px">Reading from {K.get("now_asof", "")}: prices at the moment of recording, our chance from that day\'s close. We record them every 6 hours and settle them after the markets close '
                f'(recorded: {K["rows"]}, settled: {K["resolved"]}).</p>'
                f'<div class="tw"><table class="tbl"><tr><th>Will the price at least once be</th><th>Deadline</th><th>Kalshi market</th><th>Our model</th></tr>{kr}</table></div>')
    return out + '</div>'


def volev_html(VE):
    """Calm or stormy: chances of volatility events and how they were settled."""
    if not VE:
        return ""
    pc = lambda v: f"{v * 100:.0f}%"
    pl = lambda v: f"{v * 100:+.1f}%"
    tiles, rows = "", ""
    for k, v in VE.items():
        if v.get("p_now") is None:
            continue
        e = v["eval"]
        badge = "edge confirmed" if v["confirmed"] else "no confirmed edge"
        tiles += (f'<div class="card tile"><span class="muted sm">{html.escape(v["q"])}</span><b>{pc(v["p_now"])}</b>'
                  f'<span class="muted sm">usually: {pc(v["base_rate"])} · {badge}</span></div>')
        rows += (f'<tr><td>{html.escape(v["label"])}</td><td>{pl(e["przed"]["skill_pers"])}</td><td>{pl(e["po"]["skill_pers"])}</td>'
                 f'<td>{pl(e["calosc"]["skill_clim"])}</td><td class="muted">{e["calosc"]["n"]}</td></tr>')
    rel = (VE.get("burza7") or {}).get("reliability") or []
    reltxt = ""
    hi = [r for r in rel if r["lo"] >= 0.6]
    if hi:
        r = hi[0]
        reltxt = (f' One weakness: at high chances the model tends to be too cautious. When it said on average {pc(r["pred"])} for a stormy week, '
                  f'the week was stormy in {pc(r["real"])} of cases.') if r["real"] - r["pred"] > 0.05 else ""
    return (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Calm or stormy?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">We cannot predict the direction of the price, but we can predict whether big swings are coming. '
            f'A “stormy week” is a week in which the swings are bigger than their median over the last year. The chances are calculated by a simple model from today\'s volatility forecast, '
            f'trained every week on past data only.</p><div class="tiles3">{tiles}</div>'
            f'<div class="tw" style="margin-top:12px"><table class="tbl"><tr><th>Event</th><th>Better than “same as last time”: before Oct 29, 2024</th>'
            f'<th>…on the exam (last 2 years)</th><th>Better than the fixed frequency (whole history)</th><th>Trials</th></tr>{rows}</table></div>'
            f'<p class="note"><b>How to read this:</b> “same as last time” is a fair rival: it assumes that if the last period was stormy, the next one will be too. '
            f'“Edge confirmed” means the model beat it (Brier score) both before the cut-off and on the exam; we wrote this rule down before seeing the results.{reltxt}</p></div>')


def bands_html(ens):
    """Do the bands fit the state of the market: scaling test and coverage in calm, average and stormy periods."""
    if not ens:
        return ""
    pl = lambda v: "–" if v is None or v != v else f"{v * 100:+.1f}%"
    pc = lambda v: f"{v * 100:.0f}%"
    # market-state labels come from ensemble.py; accept both the Polish and the English names
    states = (("spokojnie", "calm"), ("przeciętnie", "average"), ("burzliwie", "stormy"))
    rows = ""
    for H, v in sorted(ens["horizons"].items(), key=lambda a: int(a[0])):
        st, au = v.get("scale_test") or {}, v.get("audit") or {}
        if not st or "skill_pre" not in st:
            continue
        cov = " / ".join(next((pc(au[k]["cov80"]) for k in ks if k in au), "–") for ks in states)
        rows += (f'<tr><td><b>{H} days</b></td><td class="{"pos" if st["used"] else "muted"}">{"yes" if st["used"] else "no"}</td>'
                 f'<td>{pl(st["skill_pre"])}</td><td>{pl(st["skill_post"])}</td><td>{cov}</td></tr>')
    return (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Do the bands fit the state of the market?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">After a calm period swings usually come back, and after a stormy one they die down. Simply scaling today\'s volatility '
            f'to a longer period does not take this into account, so the bands used to be too narrow in calm times and too wide after a storm. The new scaling assumes that volatility returns to its '
            f'norm of the last 2 years. We switch it on for a given horizon only if it improved the forecast before Oct 29, 2024 (with a small chance of luck) and still improves it on the exam.</p>'
            f'<div class="tw"><table class="tbl"><tr><th>Horizon</th><th>New scaling</th><th>Improvement before</th><th>Improvement on the exam</th>'
            f'<th>Price within the 80% band: calm / average / stormy</th></tr>{rows}</table></div>'
            f'<p class="muted sm" style="margin:6px 0 0">The last column settles the final bands since about 2015 in three market states (volatility terciles from the last 3 years). '
            f'Target: about 80% in each. Clearly less in calm periods means the bands are still sometimes too narrow then: after a long quiet spell a surprise comes easily.</p>'
            + rv_race_html(ens) + '</div>')


def rv_race_html(ens):
    """Volatility race on 5-minute data (HAR-RV, HARQ), judged on the final 1, 3 and 7-day bands."""
    allh = {**{int(k): v for k, v in (ens.get("short") or {}).items()}, **{int(k): v for k, v in ens["horizons"].items()}}
    pl = lambda v: "–" if v is None or v != v else f"{v * 100:+.1f}%"
    hn = lambda H: f'{H} {"day" if H == 1 else "days"}'
    rows, frm, used = "", None, []
    for H in (1, 3, 7):
        rc = (allh.get(H) or {}).get("rv_race") or {}
        if not rc.get("cand"):
            continue
        frm = frm or rc.get("from")
        if rc.get("used"):
            used.append(hn(H))
        for k, lab in (("har_rv", "HAR-RV"), ("harq", "HARQ")):
            c = rc["cand"].get(k)
            if c:
                rows += (f'<tr><td><b>{hn(H)}</b></td><td>{lab}</td><td>{pl(c["skill_pre"])} <span class="muted sm">(p {c["p_pre"]:.2f})</span></td>'
                         f'<td>{pl(c["skill_post"])}</td><td class="{"pos" if c["passed"] else "muted"}">{"yes" if c["passed"] else "no"}</td></tr>')
    if not rows:
        return ""
    verdict = (f"Volatility from 5-minute data won on these horizons: {', '.join(used)}, so that is where we calculate the bands from it." if used else
               "None of the candidates met the rule, so we calculate the bands the old way.")
    return (f'<h3 class="ct" style="margin-top:18px">Race: volatility from 5-minute data</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">Two models learn from swings recorded every 5 minutes on Binance: HAR-RV (daily, weekly and monthly swings) '
            f'and HARQ, which trusts yesterday\'s measurement less when it was uncertain. Both are retrained every year, only on what was already known. '
            f'The judge is the accuracy of the final price band (pinball loss), calculated on the same days as the current method, from {frm}. '
            f'Rule written down on Oct 31, 2026: a candidate replaces the current scale if it was better before Oct 29, 2024 (p &lt; 0.10) and is still better on the exam.</p>'
            f'<div class="tw"><table class="tbl"><tr><th>Horizon</th><th>Model</th><th>Improvement before</th><th>Improvement on the exam</th><th>Goes in?</th></tr>{rows}</table></div>'
            f'<p class="note"><b>Conclusion:</b> {verdict}</p>')


def ledger_html(ens):
    """Live log of every method (since Oct 30, 2026): the only fully clean test."""
    LD = (ens or {}).get("ledger")
    if not LD:
        return ""
    head = (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Live log of all methods</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">Since {LD["first"]} we record the forecast of every method for every horizon each day, with the program version number, '
            f'and we never change it. After the deadline we only add what actually happened. This is the only test in which no forecast could have “seen” the future, '
            f'because it was made before the future happened. Forecasts recorded: {LD["rows"]}, settled: {LD["matured"]}.</p>')
    by = [b for b in LD.get("by", []) if b["n"] >= 10]
    if not by:
        return head + '<p class="muted sm" style="margin:0">The first results will appear once at least 10 forecasts of a given method have reached their deadline (for 1 day: in about 2 weeks).</p></div>'
    lab = lambda m: MODEL_LAB.get(m, {"published": "Published forecast"}.get(m, m))
    pl = lambda v: "–" if v is None or v != v else f"{v * 100:+.1f}%"
    rows = "".join(f'<tr><td>{b["horizon"]} days</td><td>{html.escape(lab(b["model"]))}</td><td>{b["n"]}</td><td>{b["cov80"] * 100:.0f}%</td><td>{pl(b["skill"])}</td></tr>'
                   for b in sorted(by, key=lambda b: (b["horizon"], b["model"])))
    return head + (f'<div class="tw"><table class="tbl"><tr><th>Horizon</th><th>Method</th><th>Settled</th><th>Price within the 80% band</th>'
                   f'<th>Edge over “no change”</th></tr>{rows}</table></div></div>')


def history_track_html(ens):
    """Settling on history: thousands of forecasts from the walk-forward test (the model only knows the past on the day of the forecast)."""
    if not ens:
        return ""
    rows = ""
    for H, v in sorted(ens["horizons"].items(), key=lambda a: int(a[0])):
        e = next((r for r in v["table"] if r["model"] == "ens"), None)
        if not e:
            continue
        c80, c90 = e["cover80"] * 100, (e.get("cover90") or 0) * 100
        sk = e["skill"] * 100
        cls = lambda x, t: "pos" if abs(x - t) <= 4 else "neg"
        a = v.get("aci") or {}
        atxt = (f'yes (was {a["cov80_before"] * 100:.0f}% / {a["cov90_before"] * 100:.0f}%)' if a.get("used") else "no") if a else "–"
        rows += (f'<tr><td><b>{H} days</b></td><td>{v["n"]}</td><td class="{cls(c80, 80)}">{c80:.0f}%</td><td class="{cls(c90, 90)}">{c90:.0f}%</td>'
                 f'<td class="{"pos" if sk > 0.5 else "muted"}">{sk:+.1f}%</td><td class="muted">{atxt}</td></tr>')
    return (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Settling on history: do the bands tell the truth?</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">For every week since about 2015 the model made a forecast knowing only the data up to that day, '
            f'and then we compared it with what actually happened. An honest model should have the price within the 80% band in about 80% of cases and within the 90% band in about 90%. '
            f'The “Edge” column shows how much better the whole team of methods is than the plain assumption “no change, typical swings” (0% = just as good). '
            f'The last one: whether we use adaptive bands (ACI), which widen by themselves when the price has recently fallen outside them more often. '
            f'We switch them on only where they make the bands match reality better.</p>'
            f'<div class="tw"><table class="tbl"><tr><th>Horizon</th><th>Forecasts</th><th>Within the 80% band</th><th>Within the 90% band</th><th>Edge over “no change”</th><th>Adaptive bands</th></tr>{rows}</table></div></div>')


def track_html(stats, ens=None):
    hist = history_track_html(ens) + ledger_html(ens)
    if not stats:
        return ('<div class="card empty"><b>The forecast log is still filling up</b>'
                '<p class="muted">We record every forecast, and once its deadline passes we compare it with the real price. '
                'The first results will appear a week after the system was launched.</p></div>' + hist)
    cov = stats["coverage80"] * 100
    extra = ""
    if stats.get("brier") is not None:
        extra += (f'<div class="card tile"><span class="muted sm">Chance of a rise: Brier score</span><b>{stats["brier"]:.3f}'
                  + f'</b><span class="muted sm">0 = perfect, 0.25 = coin toss ({stats["brier_n"]} forecasts)</span></div>')
    if stats.get("coverage90") is not None:
        extra += (f'<div class="card tile"><span class="muted sm">Price within the 90% band</span><b>{stats["coverage90"] * 100:.0f}%</b><span class="muted sm">target: 90%</span></div>')
    return (f'<p class="muted sm" style="margin:0 0 10px">Live, since {stats.get("first", "")}: forecasts recorded in advance and settled after the deadline.</p>'
            f'<div class="tiles">{extra}<div class="card tile"><span class="muted sm">Forecasts checked</span><b>{stats["n"]}</b></div>'
            f'<div class="card tile"><span class="muted sm">Price stayed within the range</span><b>{cov:.0f}%</b><span class="muted sm">target: 80%</span></div>'
            f'<div class="card tile"><span class="muted sm">On average we were off by</span><b>{stats["mape"]:.1f}%</b><span class="muted sm">middle of the forecast vs the real price</span></div>'
            + (f'<div class="card tile"><span class="muted sm">Direction right</span><b>{stats["direction"]*100:.0f}%</b><span class="muted sm">50% = the same as a coin toss (counted for {stats["direction_n"]} forecasts that pointed in a direction)</span></div></div>'
               if stats.get("direction") is not None else
               '<div class="card tile"><span class="muted sm">Direction right</span><b>not applicable</b><span class="muted sm">the forecasts were neutral (middle equal to today\'s price), so they did not point in any direction</span></div></div>')) + hist


def sources_html(status):
    ok = sum(1 for _, s, _ in status if s == "OK")
    pills = "".join(f'<div class="pill {"ok" if s == "OK" else "bad"}" title="{html.escape(i)}"><i></i>{html.escape(nm)}</div>'
                    for nm, s, i in status)
    return ok, len(status), pills


def corr_html(corr):
    names = {"ndx": "Nasdaq", "spx": "S&P 500", "gold": "Gold", "dxy": "Dollar (DXY)"}
    rows = ""
    for k, v in corr.items():
        w = abs(v) * 50
        rows += (f'<div class="sig sm-row"><div class="sig-l"><b>{names[k]}</b></div><div class="dv"><i class="mid"></i>'
                 f'<span class="{"g" if v >= 0 else "r"}" style="left:{50 if v >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div>'
                 f'<div class="sig-v"><b>{v:+.2f}</b></div></div>')
    return rows


# ----------------------------------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------------------------------

def baro_svg(v):
    """Barometer: a 240-degree arc from 'expensive' (left) to 'cheap' (right), with an animated needle."""
    cx = cy = 200
    r, a0, sweep = 150, 150, 240
    pt = lambda th, rad: (cx + rad * math.cos(math.radians(th)), cy + rad * math.sin(math.radians(th)))
    x0, y0 = pt(a0, r)
    x1, y1 = pt(a0 + sweep, r)
    segs = ""
    N = 72
    for i in range(N):
        t0, t1 = a0 + sweep * i / N, a0 + sweep * (i + 1) / N + 0.7
        xa, ya = pt(t0, r)
        xb, yb = pt(t1, r)
        k = (i + 0.5) / N
        segs += (f'<path d="M{xa:.1f} {ya:.1f}A{r} {r} 0 0 1 {xb:.1f} {yb:.1f}" style="stroke:color-mix(in srgb,var(--red) {100 - k * 100:.0f}%,var(--green) {k * 100:.0f}%)"/>')
    ticks = ""
    for vv in (-1, -0.5, 0, 0.5, 1):
        th = a0 + (vv + 1) / 2 * sweep
        xa, ya = pt(th, r + 18)
        xb, yb = pt(th, r + 28 if vv in (-1, 0, 1) else r + 24)
        ticks += f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" stroke="var(--line2)" stroke-width="2" stroke-linecap="round"/>'
    top = pt(270, r + 46)
    vt = f"{v:+.2f}".replace("-", "−")
    return f"""<svg viewBox="0 0 400 360" role="img" aria-label="Market score {vt}">
<path d="M{x0:.1f} {y0:.1f}A{r} {r} 0 1 1 {x1:.1f} {y1:.1f}" fill="none" stroke="var(--track)" stroke-width="26" stroke-linecap="round"/>
<g fill="none" stroke-width="12" stroke-linecap="butt" opacity=".95">{segs}</g>
{ticks}
<text x="{pt(a0, r)[0] - 6:.0f}" y="{pt(a0, r)[1] + 46:.0f}" text-anchor="middle" style="fill:var(--red);font:600 12px var(--f-mono);letter-spacing:.14em">EXPENSIVE</text>
<text x="{pt(a0 + sweep, r)[0] + 6:.0f}" y="{pt(a0 + sweep, r)[1] + 46:.0f}" text-anchor="middle" style="fill:var(--green);font:600 12px var(--f-mono);letter-spacing:.14em">CHEAP</text>
<text x="{top[0]:.0f}" y="{top[1] + 4:.0f}" text-anchor="middle" style="fill:var(--faint);font:600 10.5px var(--f-mono);letter-spacing:.14em">NEUTRAL</text>
<g class="needle" style="--rot:{v * 120:.1f}deg;filter:drop-shadow(0 0 8px color-mix(in srgb,var(--accent) 70%,transparent))"><path d="M195 200L205 200L200 {200 - (r - 30)}Z" fill="var(--accent)"/></g>
<circle cx="200" cy="200" r="12" fill="var(--fg)"/><circle cx="200" cy="200" r="4.5" fill="var(--bg)"/>
<text x="200" y="262" text-anchor="middle" style="fill:var(--fg);font:800 46px var(--f-display);letter-spacing:-.03em">{vt}</text>
</svg>"""


def isobars_svg():
    import random
    rnd = random.Random(11)
    paths = ""
    for k in range(11):
        base = 50 + k * 34
        ph = [rnd.uniform(0, 6.28) for _ in range(3)]
        pts = []
        for j in range(181):
            th = j / 180 * 2 * math.pi
            rr = base * (1 + 0.10 * math.sin(2 * th + ph[0]) + 0.06 * math.sin(3 * th + ph[1]) + 0.035 * math.sin(5 * th + ph[2]))
            pts.append((400 + rr * 1.25 * math.cos(th), 300 + rr * 0.85 * math.sin(th)))
        paths += '<path d="M' + "L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + 'Z"/>'
    return f'<svg class="iso" viewBox="0 0 800 600" aria-hidden="true">{paths}</svg>'


def forecast_tiles(res, ens):
    names = {7: "In a week", 30: "In a month", 90: "In 3 months", 180: "In 6 months"}
    p0 = res["price"]
    out = ""
    for f in res["forecasts"]:
        H = f["horizon"]
        lo, hi = math.log(f["p5"]), math.log(f["p95"])
        pos = lambda v: (math.log(v) - lo) / (hi - lo) * 100
        badge = '<span class="badge on">backed by models</span>' if ens["horizons"][H]["alpha"] > 0 else '<span class="badge">typical swings</span>'
        d = f["target"]
        mon = MON[int(d[5:7]) - 1]
        out += (f'<div class="ft"><span class="when">{names.get(H, f"In {H} days")} · {mon} {int(d[8:10])}</span>'
                f'<div class="rng">{f["p10"] / 1000:.0f} – {f["p90"] / 1000:.0f}<small>k USD</small></div>'
                f'<div class="cone"><div class="a" style="left:{pos(f["p10"]):.1f}%;width:{pos(f["p90"]) - pos(f["p10"]):.1f}%"></div>'
                f'<div class="m" style="left:{pos(f["p50"]):.1f}%"></div><div class="n" style="left:{pos(p0):.1f}%"></div></div>'
                f'<div class="foot"><span>{(f["p10"] / p0 - 1) * 100:+.0f}%</span><span>{(f["p90"] / p0 - 1) * 100:+.0f}%</span></div>'
                f'<div class="why">Chance of a rise about {f["prob_up"] * 100:.0f}%. {badge}</div></div>')
    return out


def verdict_html(res):
    fc = {f["horizon"]: f for f in res["forecasts"]}
    f = fc.get(30) or list(fc.values())[0]
    return f'{res["regime"]}. In a month the price will most likely be between <b>{n(f["p10"] / 1000)}k</b> and <b>{n(f["p90"] / 1000)}k</b> USD.'


def signals_html(res):
    fe = res["features"]
    by = {p["key"]: p for p in res["signals"]}
    out = ""
    for i, (title, sub, keys) in enumerate(GROUPS):
        rows, scs, ws = "", [], []
        for k in keys:
            p = by.get(k)
            if p is None or p["weight"] == 0:
                continue
            s = p["score"]
            if s is None:
                scale, val, txt = '<div class="scale na"></div>', "–", "no data"
            else:
                scs.append(s * p["weight"])
                ws.append(p["weight"])
                scale = f'<div class="scale"><i class="pin" style="left:{(s + 1) / 2 * 100:.0f}%"></i></div>'
                val = raw_value(k, fe)
                txt = "cheap" if s >= 0.35 else "expensive" if s <= -0.35 else "neutral"
            rows += (f'<div class="srow"><div class="s-name"><b>{LABELS[k]}</b><span>{SRC_NOTE[k]}</span></div>{scale}'
                     f'<div class="s-val"><b>{val}</b><span>{txt}</span></div></div>')
        avg = sum(scs) / sum(ws) if ws else None
        chip = "" if avg is None else ('<span class="gchip g">rather cheap</span>' if avg >= 0.15 else '<span class="gchip r">rather expensive</span>' if avg <= -0.15 else '<span class="gchip">neutral</span>')
        out += (f'<details class="grp"{" open" if i == 0 else ""}><summary><div class="gt"><h3>{title}</h3><p>{sub}</p></div>{chip}</summary>'
                f'<div class="rows">{rows}</div></details>')
    return out


def summary_bullets(res, ens, E):
    p0 = res["price"]
    none = all(h["alpha"] == 0 for h in ens["horizons"].values())
    if none:
        direction = ('<b>Will the price go up or down? We cannot reliably predict that.</b> We checked this against history: '
                     'none of our methods did better than the simple assumption “the price will stay the same”. So the chance of a rise is about 50/50.')
    else:
        act = ", ".join(str(H) for H, h in ens["horizons"].items() if h["alpha"] > 0)
        direction = (f'<b>Will the price go up or down?</b> Over {act} days our methods turned out better than guessing and nudge the forecast slightly, '
                     'but this is still very uncertain. For the other periods we cannot reliably predict it.')
    nxt = next((u for u in E["upcoming"] if u["cat"] != "nfp"), None)
    items = [
        f'<li><div><b>Bitcoin costs {n(p0)} USD today.</b> {res["regime"]}. <span class="muted">(Market score {res["composite"]:+.2f} on a scale from −1 “expensive” to +1 “cheap”.)</span></div></li>',
        '<li><div><b>Where could the price be?</b> The ranges below contain the real price in about 4 cases out of 5. For any number of days you can check this with the slider in the “Forecast” section.</div></li>',
        f'<li><div>{direction}</div></li>',
        '<li><div><b>What we do well:</b> judging how much the price may swing. In tests on history our ranges were right in about 8 cases out of 10.</div></li>',
    ]
    if nxt:
        items.append(f'<li><div><b>Next important event:</b> {html.escape(nxt["title"])} ({nxt["date"]}).</div></li>')
    return "".join(items)


def _now_str():
    """Refresh time in Polish time, Europe/Warsaw (if the time zone database is missing: UTC, labelled as such)."""
    try:
        from zoneinfo import ZoneInfo
        t = datetime.now(ZoneInfo("Europe/Warsaw"))
        return f"{MON[t.month - 1]} {t.day}, {t.year}, {t:%H:%M} {t:%Z}"
    except Exception:
        t = datetime.utcnow()
        return f"{MON[t.month - 1]} {t.day}, {t.year}, {t:%H:%M} UTC"


TRADER_CSS = (".tj{border:1px solid var(--line);border-radius:14px;padding:14px 16px;background:var(--card2);display:grid;gap:8px}"
              ".tj+.tj{margin-top:12px}.tj h4{margin:0;display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline;font:600 15px/1.3 var(--f-body)}"
              ".tj .tag{font:600 11px/1 var(--f-mono);letter-spacing:.06em;text-transform:uppercase;padding:5px 8px;border-radius:999px;border:1px solid currentColor}"
              ".tj dl{margin:0;display:grid;gap:6px}.tj dt{font:600 11px/1.2 var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}"
              ".tj dd{margin:0}.tj .foot{font-size:12.5px;color:var(--muted)}")


def _tj_card(r):
    """One trader journal entry: what I see, what I do and why, when I'll change my mind."""
    zl = lambda v: f"{v:,.0f} PLN"
    # KUP / SPRZEDAJ / TRZYMAJ are internal action codes written by trader.py; only the labels shown are English
    lab, cls = {"KUP": ("buy", "pos"), "SPRZEDAJ": ("sell", "neg"), "TRZYMAJ": ("waiting", "muted")}[r["action"]]
    amt = f" {zl(r['amount_pln'])}" if r["action"] != "TRZYMAJ" else ""
    w_after = r["btc"] * r["price_pln"] / r["value"] if r["value"] else 0
    usd = f'{r["price_usd"]:,.0f}'
    return (f'<article class="tj"><h4><span>{html.escape(str(r["local"]))}</span><span class="tag {cls}">{lab}{amt}</span>'
            f'<span class="muted sm">price {zl(r["price_pln"])} ({usd} USD)</span></h4>'
            + f'<dl><div><dt>What I see</dt><dd>{html.escape(str(r["seen"]))}</dd></div>'
            f'<div><dt>What I do and why</dt><dd>{html.escape(str(r["do"]))}</dd></div>'
            f'<div><dt>When I\'ll change my mind</dt><dd>{html.escape(str(r["change"]))}</dd></div></dl>'
            f'<div class="foot">After the decision: portfolio {zl(r["value"])}, {w_after * 100:.0f}% in BTC, cash {zl(r["cash"])}. Price source: {html.escape(str(r["src"]))}.</div></article>')


def _tj_score(TR):
    zl = lambda v: f"{'+' if v >= 0 else '−'}{abs(v):,.0f} PLN"
    sc = TR.get("score") or []
    d1 = [x["d1"] for x in sc if x.get("d1") is not None]
    d7 = [x["d7"] for x in sc if x.get("d7") is not None]
    if not sc:
        return "The trader has not made any trades yet, so there is nothing to settle."
    part = lambda arr, lab: (f"after {lab}: {len(arr)} settled, {sum(v > 0 for v in arr)} in profit, {zl(sum(arr))} in total" if arr else f"after {lab}: too early to tell")
    p1 = part(d1, '1 day')
    return (f"We compare every trade with what would have happened if the trader had done nothing (fees included). {p1[0].upper() + p1[1:]}; {part(d7, '7 days')}. "
            "Single days are mostly luck; only the total over many months means something.")


def trader_html(TR):
    """Virtual trader: decisions about every 6 hours, the latest journal entries and the settlement of trades."""
    if not TR or not TR.get("log"):
        return ('<div class="card empty"><b>The trader will start at the next cloud update</b>'
                '<p class="muted">Decisions are made about every 6 hours; the first one will appear here with its reasoning.</p></div>')
    zl = lambda v: f"{v:,.0f} PLN"
    diff = TR["value"] - TR["dca_value"]
    tiles = [("Value now", zl(TR["value"]), f"deposited {zl(TR['deposited'])}"),
             ("Result", f"{(TR['value'] / TR['deposited'] - 1) * 100:+.1f}%", "since the first deposit on Oct 28, 2026"),
             ("Compared with regular buying", f'{"+" if diff >= 0 else "−"}{zl(abs(diff))}', f"regular buying: {zl(TR['dca_value'])}"),
             ("Now in BTC", f"{TR['w'] * 100:.0f}%", f"cash {zl(TR['cash'])}, fees {zl(TR['fees'])}")]
    tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b}</b><span class="muted sm">{c}</span></div>' for a, b, c in tiles)
    cards = "".join(_tj_card(r) for r in TR["log"][:6])
    return (f'<style>{TRADER_CSS}</style>'
            f'<h3 class="ct" style="margin:0 0 12px">Virtual trader: a decision about every 6 hours</h3>'
            f'<p class="muted sm" style="margin:0 0 12px">Since Oct 31, 2026 the trader checks the market at every update (around 1:00, 7:00, 13:00 and 19:00 Polish time) '
            f'and, at the current price, decides whether to buy, sell or wait, and for how much. Goal: as many zloty (PLN) as possible over the long run. Each entry says what the trader sees, what it does '
            f'and at what price it will change its mind. Entries are only ever added, never edited. So far {TR["n"]} decisions, including {TR["n_trades"]} trades.</p>'
            f'<div class="tiles">{tl}</div>'
            f'<div class="panel" style="margin-top:18px"><h3 class="ct">Journal (newest first)</h3>{cards}'
            f'<p style="margin:14px 0 0"><a href="journal.html"><b>The full trader journal →</b></a></p></div>'
            f'<div class="panel" style="margin-top:18px"><h3 class="ct">Do the decisions pay off?</h3><p style="margin:0">{_tj_score(TR)}</p>'
            f'<p class="note"><b>To be honest:</b> more frequent decisions do not by themselves bring more profit. Nobody can predict the direction of the price over the next hours and days '
            f'(our tests cannot either), and every trade costs a fee. That is why the trader only trades when the picture of the market changes '
            f'(the trend or the level of swings), not at every twitch of the price. This is an experiment with virtual money, not investment advice.</p></div>'
            f'<h3 class="ct" style="margin:28px 0 12px">For comparison: the same rules, one decision a day, and a test on history</h3>')


def write_journal(TR):
    """Separate page docs/journal.html: all of the trader's decisions, easy to read, newest first."""
    if not TR or not TR.get("log"):
        return
    zl = lambda v: f"{v:,.0f} PLN"
    head = (C.ROOT / "template_head.html").read_text(encoding="utf-8")
    a, b = head.find(":root{"), head.find("*{box-sizing")
    tokens = head[a:b] if a >= 0 and b > a else ""
    cards = "".join(_tj_card(r) for r in TR["log"])
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<title>Bitcoin virtual trader journal: every decision explained | Bitcoin Barometer</title>''<meta name="description" content="A virtual Bitcoin trader decides every 6 hours whether to buy, sell or wait, and explains each decision in plain English. Educational, no real money.">''<meta name="author" content="Przemysław Wywigacz"><link rel="canonical" href="https://wywigacz.github.io/bitcoin-barometer/journal.html">''<meta property="og:image" content="https://wywigacz.github.io/bitcoin-barometer/img/og.png">'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700&family=Manrope:wght@400;600&family=IBM+Plex+Mono:wght@600&display=swap">'
            f'<style>{tokens}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:400 15.5px/1.6 var(--f-body);'
            'padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}'
            'main{max-width:760px;margin:0 auto;padding-block:32px 64px}h1{font:700 30px/1.15 var(--f-display);margin:0 0 8px;text-wrap:balance}'
            'a{color:var(--accent)}.muted{color:var(--muted)}.sm{font-size:12.5px}.pos{color:var(--green)}.neg{color:var(--red)}'
            f'{TRADER_CSS}</style></head><body><main>'
            '<p class="sm"><a href="index.html">← Bitcoin Barometer</a></p><h1>Virtual trader journal</h1>'
            f'<p class="muted">A decision about every 6 hours since Oct 31, 2026. Portfolio {zl(TR["value"])} on deposits of {zl(TR["deposited"])}; '
            f'regular buying would have {zl(TR["dca_value"])}. An experiment with virtual money, not investment advice.</p>'
            + f'<p>{_tj_score(TR)}</p>{cards}</main></body></html>')
    (C.DASHBOARD_HTML.parent / "journal.html").write_text(page, encoding="utf-8")


def sim_html(SM):
    """Virtual, actively managed portfolio: status, latest decisions with reasons, test on history."""
    if not SM:
        return '<div class="card empty"><b>The portfolio has not started yet</b><p class="muted">The first decision will appear after the next update.</p></div>'
    import sim as SI
    zl = lambda v: f"{v:,.0f} PLN"
    pc = lambda v: f"{v * 100:.0f}%"
    pl = lambda v: f"{v * 100:+.1f}%"
    out = ""
    L = SM.get("live")
    if L:
        gain = L["value"] / L["deposited"] - 1
        diff = L["value"] - L["dca_value"]
        bars = "".join(f'<div class="sig sm-row"><div class="sig-l"><b>{k.capitalize()}</b><span class="muted sm"> wants {pc(L["t"][k])} in BTC</span></div>'
                       f'<div class="dv"><span class="g" style="left:0;width:{L["p"][k] * 100:.0f}%"></span></div><div class="sig-v"><b>{pc(L["p"][k])}</b></div></div>'
                       for k in SI.STRATS)
        tiles = [("Portfolio value", zl(L["value"]), f"as of {L['date']}"),
                 ("Deposited", zl(L["deposited"]), f"result {pl(gain)}"),
                 ("Compared with regular buying", f'{"+" if diff >= 0 else "−"}{zl(abs(diff))}', f"regular buying: {zl(L['dca_value'])}"),
                 ("Now in BTC", pc(L["w"]), f"rest in cash: {zl(L['cash'])}")]
        tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b}</b><span class="muted sm">{c}</span></div>' for a, b, c in tiles)
        last = L["log"][0]
        logrows = ""
        for r in L["log"]:
            # KUP / SPRZEDAJ / TRZYMAJ are internal action codes written by sim.py
            act = {"KUP": "buy", "SPRZEDAJ": "sell", "TRZYMAJ": "no change"}[r["action"]]
            cls = {"KUP": "pos", "SPRZEDAJ": "neg", "TRZYMAJ": "muted"}[r["action"]]
            dep = f' <span class="muted sm">(+{zl(r["deposit"])} deposit)</span>' if r["deposit"] else ""
            logrows += (f'<tr><td>{r["date"]}{dep}</td><td class="{cls}"><b>{act}</b>{" " + zl(r["amount_pln"]) if r["action"] != "TRZYMAJ" else ""}</td>'
                        f'<td>{pc(r["w_before"])} → {pc(r["target"])}</td><td>{zl(r["value"])}</td></tr>'
                        f'<tr class="why"><td colspan="4" class="muted sm">{html.escape(r["why"])}</td></tr>')
        out += (f'<div class="tiles">{tl}</div>'
                f'<div class="two2" style="margin-top:18px"><div class="panel"><h3 class="ct">Latest decision ({last["date"]})</h3>'
                f'<p style="margin:0">{html.escape(last["why"])}</p></div>'
                f'<div class="panel"><h3 class="ct">What it has learned: strategy weights</h3>'
                f'<p class="muted sm" style="margin:0 0 8px">A strategy that would have earned more over the last year gets a bigger say (the Hedge algorithm). Each gets at least 5%.</p>{bars}</div></div>'
                f'<div class="panel" style="margin-top:18px"><h3 class="ct">Decision log (newest first)</h3>'
                f'<div class="tw"><table class="tbl"><tr><th>Day</th><th>Decision</th><th>BTC share</th><th>Value</th></tr>{logrows}</table></div></div>')
    b = SM["backtest"]
    a, d = b["active"], b["dca"]
    cv = b["curve"]
    n = len(cv["a"])
    mx = max(max(cv["a"]), max(cv["b"])) or 1
    W_, H_ = 640, 220
    pts = lambda arr: " ".join(f"{i / (n - 1) * W_:.1f},{H_ - v / mx * H_:.1f}" for i, v in enumerate(arr))
    yrs = sorted({s[:4] for s in cv["d"]})
    ticks = "".join(f'<text x="{cv["d"].index(next(s for s in cv["d"] if s.startswith(y))) / (n - 1) * W_:.0f}" y="{H_ + 16}" class="ax">{y}</text>' for y in yrs[::2])
    svg = (f'<svg viewBox="0 -6 {W_} {H_ + 24}" role="img" aria-label="Value of the active portfolio and of regular buying since {b["start"]}" style="width:100%;height:auto">'
           f'<polyline points="{pts(cv["dep"])}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 4"/>'
           f'<polyline points="{pts(cv["b"])}" fill="none" stroke="var(--amber)" stroke-width="2"/>'
           f'<polyline points="{pts(cv["a"])}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>{ticks}</svg>')
    row = lambda lab, f: f'<tr><td>{lab}</td><td><b>{f(a)}</b></td><td>{f(d)}</td></tr>'
    tbl = (row("Value at the end", lambda s: zl(s["final"])) + row("Gain relative to deposits", lambda s: pl(s["gain_pct"]))
           + row("Annual return (excluding the effect of deposits)", lambda s: pl(s["twr_ann"])) + row("Biggest drop along the way", lambda s: pl(s["max_dd"])))
    out += (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Test on history: the same rules since {b["start"]}</h3>'
            f'<p class="muted sm" style="margin:0 0 10px">Total deposits {zl(a["deposited"])}. Orange line: plain regular buying. '
            f'Accent-coloured line: the active portfolio. Dashed line: total deposits. The portfolio made {b["trades"]} trades, held {pc(b["avg_btc"])} in BTC on average, '
            f'and fees are included.</p>{svg}'
            f'<div class="tw"><table class="tbl" style="margin-top:10px"><tr><th></th><th>Active portfolio</th><th>Regular buying</th></tr>{tbl}</table></div>'
            f'<p class="note"><b>To be honest:</b> this result is optimistic. We chose the trend strategy knowing from test H7 that the trend worked in exactly these years, '
            f'and a test on history does not know about future changes in the market. The real measure is the live portfolio above: we record its decisions every day and never edit them afterwards. '
            f'Exchange risk is left out; tax is calculated separately below.</p></div>')
    tx = b.get("tax") or {}
    if tx.get("since2018"):
        trow = lambda lab, t: (f'<tr><td>{lab}</td><td>{zl(t["deposited"])}</td><td><b>{zl(t["active_after"])}</b> <span class="muted sm">(tax {zl(t["active_tax"])})</span></td>'
                               f'<td>{zl(t["dca_after"])} <span class="muted sm">(tax {zl(t["dca_tax"])})</span></td>'
                               f'<td>{t["active_after"] / t["dca_after"]:.2f}</td></tr>' if t["dca_after"] else "")
        tr_ = trow(f'since {b["start"]}', tx["since2018"]) + (trow("exam: from Oct 29, 2024", tx["exam"]) if tx.get("exam") else "")
        out += (f'<div class="panel" style="margin-top:18px"><h3 class="ct">After Polish tax and with interest on cash</h3>'
                f'<p class="muted sm" style="margin:0 0 10px">The active portfolio sells, so it pays tax every year, while regular buying pays only when it sells. '
                f'We calculate this as in Poland: 19% of the annual income from crypto (proceeds from sales minus purchase costs; any excess of costs carries over to later years), '
                f'with the tax paid on April 30 of the following year. Cash sits in a deposit paying the National Bank of Poland reference rate minus 1 percentage point, '
                f'and 19% tax on interest (the so-called Belka tax) is deducted. At the end both portfolios are sold and the outstanding tax is paid, so that the comparison is fair.</p>'
                f'<div class="tw"><table class="tbl"><tr><th>Period</th><th>Deposited</th><th>Active portfolio after tax</th><th>Regular buying after tax</th><th>Ratio</th></tr>{tr_}</table></div>'
                f'<p class="muted sm" style="margin:6px 0 0">A simplified version: no transfer costs and no offsetting of losses against other income. This is an educational exercise, not tax advice.</p></div>')
    rb = b.get("robust") or {}
    if rb.get("2y"):
        r2, r4 = rb["2y"], rb.get("4y") or {}
        pp = lambda v: f"{v:+.1f} pp"
        rr = lambda v: f"{v:.2f}"
        wr = ("".join(f'<tr><td>{k[0]}-year windows ({x["n"]})</td><td>{pc(x["beat_dca"])}</td><td>{rr(x["med_vs_dca"])} ({rr(x["p10"])}–{rr(x["p90"])})</td>'
                      f'<td>{pc(x["beat_blind"])}</td><td>{pp(x["dd_vs_dca_pp"])}</td><td>{pp(x["dd_vs_blind_pp"])}</td></tr>'
                      for k, x in (("2y", r2), ("4y", r4)) if x))
        fr = "".join(f'<tr><td>{f["L"]} days{" (in use)" if f["L"] == 28 else ""}</td><td>{rr(f["since2018"])}</td><td>{rr(f["exam"])}</td></tr>' for f in rb.get("fragility", []))
        ci = rb.get("exam_ci")
        citxt = (f'On the exam the active portfolio earned on average {pl(ci["mean_ann"])} a year more than regular buying, but the 90% confidence interval '
                 f'runs from {pl(ci["lo_ann"])} to {pl(ci["hi_ann"])}: zero lies inside it, so this is not proof of an edge. ') if ci else ""
        h1n = (f'A simple average of the three strategies (1/N) does almost as well as the weights learned by the Hedge algorithm: {rr(rb["1n"]["since2018"])} vs {rr(rb["hedge"]["since2018"])} '
               f'since 2018 (ratio to regular buying), so “learning” the weights adds little for now. ') if rb.get("1n") and rb.get("hedge") else ""
        out += (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Skill or luck?</h3>'
                f'<p class="muted sm" style="margin:0 0 10px">One result from one start date says little. That is why we run the same rules from the start of every month since 2014 '
                f'and compare them not only with regular buying, but also with a “blind” portfolio that holds a fixed BTC share equal to the active portfolio\'s average share '
                f'(about {pc(r2["avg_w"])}) and ignores all signals. Only an edge over that one shows that the signals add something, rather than just holding less bitcoin.</p>'
                f'<div class="tw"><table class="tbl"><tr><th></th><th>Better than regular buying</th><th>Ratio to it: median (10–90%)</th>'
                f'<th>Better than the blind portfolio</th><th>Worst drop vs regular buying</th><th>Worst drop vs blind</th></tr>{wr}</table></div>'
                f'<p class="note"><b>Conclusion:</b> the active portfolio beats regular buying in fewer than half of the windows, and its shallower dips come from holding less BTC '
                f'(compared with the blind portfolio the difference in dips is close to zero). On the other hand, it usually beats the blind portfolio with the same exposure, so the signals are not useless. '
                f'{citxt}{h1n}</p>'
                f'<h3 class="ct" style="margin-top:18px">How much does the result depend on the choice of trend window?</h3>'
                f'<p class="muted sm" style="margin:0 0 10px">The trend rule on its own (100% in BTC after a rise over the last N days, otherwise 25%), as a ratio of value to regular buying. '
                f'We chose the 28-day window knowing the history, and the result depends heavily on that choice: a warning against treating the “since 2018” figure as proof.</p>'
                f'<div class="tw"><table class="tbl"><tr><th>Trend window</th><th>Since 2018</th><th>Exam (from Oct 29, 2024)</th></tr>{fr}</table></div></div>')
    return out


def plan_data(df):
    """Monthly BTC prices in zloty (USD/PLN rate from Yahoo) for the deposit simulator."""
    import world
    try:
        w = pd.read_csv(world.WORLD_CSV, index_col=0, parse_dates=True)["y_pln"]
    except Exception:
        return None
    pln = w.reindex(df.index.union(w.index)).sort_index().ffill(limit=7).reindex(df.index)
    px = (df["close"] * pln).dropna()
    m = px.resample("MS").first().dropna()
    if len(m) < 24:
        return None
    return {"m": [d.strftime("%Y-%m") for d in m.index], "p": [round(float(v), 2) for v in m.values]}



def _bars(rows, scale=0.6):
    out = ""
    for name, r in rows:
        w = min(abs(r), scale) / scale * 50
        out += (f'<div class="sig sm-row"><div class="sig-l"><b>{html.escape(str(name))}</b></div><div class="dv"><i class="mid"></i>'
                f'<span class="{"g" if r >= 0 else "r"}" style="left:{50 if r >= 0 else 50 - w:.1f}%;width:{w:.1f}%"></span></div>'
                f'<div class="sig-v"><b>{r:+.2f}</b></div></div>')
    return out


def extra_html():
    """Tiles with extra signals: dollar liquidity, stablecoins, exchanges, Lightning, fund positions, space weather, wild cards."""
    import extra as XT
    import world as W
    if not XT.OUT.exists():
        return ""
    x = pd.read_csv(XT.OUT, index_col=0, parse_dates=True)
    pl = lambda v, d=1: f"{v:+.{d}f}"
    tiles = []

    def chg(col, days, title, note, scale=1.0):
        if col not in x:
            return
        s = x[col].dropna()
        s = s[s > 0]
        if len(s) < days + 30:
            return
        s = s.resample("D").last().ffill(limit=10)
        c = (s / s.shift(days) - 1).dropna() * 100
        tiles.append((title, pl(float(c.iloc[-1])) + "%", float((c < c.iloc[-1]).mean()), note))

    chg("x_netliq", 30, "Net dollar liquidity (month)", "Money the Fed has pumped into the system, minus what sits in the government's account and in reverse repo. A rise usually favours risky assets.")
    chg("x_usdt", 30, "New digital dollars: USDT (month)", "How much Tether has grown. New stablecoins are “cash” waiting to buy crypto.")
    chg("x_exsupply", 30, "Bitcoins sitting on exchanges (month)", "A fall means people are withdrawing coins to their own wallets, so they are more likely holding than selling.")
    chg("x_tvl", 30, "Money in DeFi (month)", "Value locked in decentralised finance. It shows the appetite for risk across the whole crypto world.")
    chg("x_ln_cap", 90, "Lightning Network (3 months)", "How many bitcoins are locked in the fast payment network. A measure of how much BTC is actually used for payments.")
    for col, title in (("x_cot_es", "Leveraged funds on the S&P 500"), ("x_cot_jy", "Leveraged funds on the yen")):
        if col in x and x[col].notna().sum() > 50:
            s = x[col].dropna()
            tiles.append((title, pl(float(s.iloc[-1]), 0) + "%", float((s < s.iloc[-1]).mean()),
                          "Net position of hedge funds as a % of the market (CFTC data). Extreme values have often come before a sharp reversal."
                          + (" Big bets against the yen have in the past ended in a sudden “unwind” and falls in risky assets (August 2024)." if col == "x_cot_jy" else "")))
    if "x_kp" in x:
        s = x["x_kp"].dropna()
        k30 = s.rolling(30).mean()
        tiles.append(("Geomagnetic storms from the Sun (30 days)", f"{k30.iloc[-1]:.1f}", float((k30.dropna() < k30.iloc[-1]).mean()),
                      "Average Kp index. This is a “wild card”: some studies link geomagnetic storms with investor mood. We check it, we do not assume it."))
    today = pd.Timestamp.now().normalize()
    st = W.static_series(pd.date_range(today - pd.Timedelta(days=60), today + pd.Timedelta(days=60)))
    m = st["cal_mercury"]
    now = bool(m.loc[today])
    nxt = m.loc[today:]
    change = nxt.index[nxt != nxt.iloc[0]]
    when = f"{MON[change[0].month - 1]} {change[0].day}" if len(change) else "?"
    tiles.append(("Mercury retrograde", "on" if now else "not on", None,
                  ("Ends on " if now else "Next one from ") + when + ". Astrology has no scientific basis; we track it only to check whether the crowd that believes in it changes its behaviour (so far: no)."))
    tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b}</b>{_pctbar(c) if c is not None else ""}<span class="muted sm">{d}</span></div>' for a, b, c, d in tiles)
    return f'<div class="tiles3">{tl}</div>'


def whales_html(CF, df=None, ens=None):
    """Moves of big players (ETFs, Strategy, Hyperliquid) and the result of the confirmatory test."""
    import confirm as CO
    import whales as WH
    if not WH.OUT.exists():
        return '<div class="card empty"><b>No data on big players</b><p class="muted">It will appear at the next update.</p></div>'
    w = pd.read_csv(WH.OUT, index_col=0, parse_dates=True)
    mln = lambda v: (f"{v/1e9:+.2f}B" if abs(v) >= 1e9 else f"{v/1e6:+.0f}M") + " USD"
    dd = lambda t, y=False: f"{MON[t.month - 1]} {t.day}" + (f", {t.year}" if y else "")
    tiles = []
    if "w_etf_flow" in w:
        f = w["w_etf_flow"].dropna()
        f5, f20 = f.rolling(5).sum(), f.rolling(20).sum()
        tiles.append((f"ETFs: last session ({dd(f.index[-1])})", mln(f.iloc[-1]), float((f < f.iloc[-1]).mean()),
                      "How much money flowed into (plus) or out of (minus) all US bitcoin ETFs in one day."))
        tiles.append(("ETFs: last month (20 sessions)", mln(f20.iloc[-1]), float((f20.dropna() < f20.iloc[-1]).mean()),
                      (f"Of which BlackRock (IBIT): {mln(w['w_etf_ibit'].dropna().iloc[-20:].sum())}. " if "w_etf_ibit" in w else "")
                      + "Research shows that this money tends to follow the price rather than lead it."))
        if "w_etf_aum" in w and w["w_etf_aum"].notna().any():
            a = w["w_etf_aum"].dropna()
            tiles.append(("ETFs: how much they hold in total", f"{a.iloc[-1]/1e9:.0f}B USD", None,
                          f"Value of the bitcoins held by all US ETFs. Last 5 sessions: {mln(f5.iloc[-1])}."))
    if "w_mstr_btc" in w:
        m = w[["w_mstr_btc", "w_mstr_total"]].dropna()
        last = m.iloc[-1]
        b90 = m.loc[m.index >= m.index[-1] - pd.Timedelta(days=90), "w_mstr_btc"].sum()
        verb = "bought" if last["w_mstr_btc"] > 0 else "sold"
        tiles.append(("Strategy (Michael Saylor): holdings", f"{last['w_mstr_total']:,.0f} BTC", None,
                      f"Most recently {verb} {abs(last['w_mstr_btc']):,.0f} BTC (report of {dd(m.index[-1], True)}). "
                      + f"Balance over 3 months: {b90:+,.0f} BTC. In 2026 the company's holdings fell several times after sales; before that it had sold only once, in December 2022, and immediately bought more."))
    if "w_hl_smart" in w and w["w_hl_smart"].notna().any():
        s, d = w["w_hl_smart"].dropna(), w["w_hl_dumb"].dropna()
        side = lambda v: "betting on a rise" if v > 0 else "betting on a fall"
        tiles.append(("Best Hyperliquid traders", side(s.iloc[-1]), None,
                      f"The {WH.TOP} accounts with the biggest all-time profit. Their combined BTC position is {s.iloc[-1]:+.0f}% of this exchange's whole market; "
                      f"{w['w_hl_smart_longpct'].dropna().iloc[-1]:.0f}% of them are betting on a rise."))
        tiles.append(("Worst Hyperliquid traders", side(d.iloc[-1]), None,
                      f"The {WH.TOP} accounts with the biggest loss. Position {d.iloc[-1]:+.0f}% of the market; {w['w_hl_dumb_longpct'].dropna().iloc[-1]:.0f}% are betting on a rise. "
                      f"We have been recording since {dd(s.index[0], True)}, because the exchange does not provide history."))
    if "w_hl_fund" in w:
        h = w["w_hl_fund"].dropna()
        h7 = h.rolling(7).mean().dropna()
        tiles.append(("Fee for betting on a rise (Hyperliquid, 7 days)", f"{h7.iloc[-1]:+.1f}% per year", float((h7 < h7.iloc[-1]).mean()),
                      "When many people want to bet on a rise, they pay the others. A high fee means a crowd on one side."))
    tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b}</b>{_pctbar(c) if c is not None else ""}<span class="muted sm">{d}</span></div>' for a, b, c, d in tiles)
    # known wallets (governments): balances cannot be faked, every address checked against reports
    ent, epi = [], ""
    sp = lambda v: f"{v:,.0f}"
    if "w_ent_us" in w and w["w_ent_us"].notna().any():
        u = w["w_ent_us"].dropna()
        moved = u.index[u.diff().abs() > 1]
        ent.append(("US government: bitcoins from Silk Road", f"{sp(u.iloc[-1])} BTC",
                    f"Coins stolen from the Silk Road black market and seized from a hacker in November 2020. "
                    + (f"Last big move: {dd(moved[-1], True)}." if len(moved) else "Nobody has touched them since 2020.")
                    + " Selling them would mean a lot of supply; since 2025 the US has said it will keep them in a strategic reserve."))
    if "w_ent_sv" in w and w["w_ent_sv"].notna().any():
        v = w["w_ent_sv"].dropna()
        na = int(w["w_ent_sv_n"].dropna().iloc[-1]) if "w_ent_sv_n" in w and w["w_ent_sv_n"].notna().any() else None
        ent.append(("El Salvador: state reserve", f"{sp(v.iloc[-1])} BTC",
                    f"Over the last 30 days {v.iloc[-1] - v.iloc[-31]:+.0f} BTC (the country buys 1 BTC a day). "
                    + (f"We count it from {na} addresses that El Salvador publishes itself." if na else "")))
    if "w_ent_de" in w and w["w_ent_de"].notna().any():
        g = w["w_ent_de"].dropna()
        ent.append(("Germany: bitcoins seized by the police", f"{sp(g.iloc[-1])} BTC",
                    f"In January 2024 the police seized {sp(g.max())} BTC from the operators of the pirate site Movie2k and sold them within 3.5 weeks (Jun 19 – Jul 12, 2024)."))
        if df is not None:
            c = df["close"]
            a0, a1 = c.asof(pd.Timestamp("2024-06-18")), c.asof(pd.Timestamp("2024-07-12"))
            lo = c["2024-06-18":"2024-07-12"]
            a2 = c.asof(pd.Timestamp("2024-08-11"))
            epi = (f'<p class="note"><b>What did the price do while Germany was selling?</b> From the day before the first sale to its end: {(a1 / a0 - 1) * 100:+.0f}%, '
                   f'with a low of {(lo.min() / a0 - 1) * 100:+.0f}% ({dd(lo.idxmin())}). In the month after it ended: {(a2 / a1 - 1) * 100:+.0f}%. '
                   f'At the same time Mt. Gox started returning bitcoins to its creditors, so the two effects cannot be separated. '
                   f'This is a single episode, and we do not draw a rule from a single case.</p>')
    if ent:
        tl += "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b}</b><span class="muted sm">{d}</span></div>' for a, b, d in ent)
    tbl = ""
    if CF:
        lab = {"confirmed": ("confirmed", "pos"), "rejected": ("not confirmed", "muted"), "reverse": ("works the other way round", "neg"),
               "collect": ("collecting data", "muted"), "retired": ("expired, no longer used", "muted")}
        rows = ""
        for r in CF["rows"]:
            st, cls = lab[r["status"]]
            mde = f' This test would detect a correlation of about {r["mde_rho"]:.2f} or more.' if r.get("mde_rho") else ""
            if r["status"] == "collect":
                det = f'{r["n"]} of {r.get("min_n", CO.MIN_DAYS)} days' + (f' (counted from {r["from"]})' if r.get("from") else "") + "." + mde
            else:
                okres = {1: "the next day", 7: "a week"}.get(r["h"], "a month")
                fp = lambda p: f"{p:.4f}" if p < 0.01 else f"{p:.2f}"
                hac = f', with the Newey-West correction {fp(r["p_hac"])}' if r.get("p_hac") is not None else ""
                if r.get("target") == "drop10":
                    det = (f'correlation {r["r"]:+.2f}, p = {fp(r["p"])}{hac}. A 10% drop within 30 days happened in {r["ret_lo"]:.0f}% of cases '
                           f'at the lowest values and in {r["ret_hi"]:.0f}% at the highest.') + mde
                elif r.get("target") == "volratio":
                    det = (f'correlation {r["r"]:+.2f}, p = {fp(r["p"])}{hac}. In a typical week after the lowest values the swings were {r["ret_lo"]:+.0f}% '
                           f'relative to the forecast from the past, after the highest {r["ret_hi"]:+.0f}%') + "." + mde
                else:
                    det = (f'correlation {r["r"]:+.2f}, p = {fp(r["p"])}{hac}. Average BTC return over {okres} after the lowest values: {r["ret_lo"]:+.1f}%, '
                           f'after the highest: {r["ret_hi"]:+.1f}%.') + mde
                hl_ = r.get("health") or {}
                if hl_.get("now"):
                    nw = hl_["now"]
                    det += (f' Health (last 3 years): correlation {nw["rho"]:+.2f}, p {nw["p_hac"]:.2f}; '
                            f'retirement review {hl_["next_check"]}.')
            rows += f'<tr><td><b>{r["id"]}</b></td><td>{html.escape(r["text"])}</td><td class="{cls}"><b>{st}</b></td><td class="sm muted">{det}</td></tr>'
        nc = len(CF["confirmed"])
        ids = [r["id"] for r in CF["rows"] if r["status"] == "confirmed"]
        vids = [r["id"] for r in CF["rows"] if r["status"] == "confirmed" and r.get("target") == "volratio"]
        big = any(r["status"] == "confirmed" and r.get("batch") == 1 for r in CF["rows"])
        verdict = ((f"Passed the test: {', '.join(ids)}. " if ids else "No hypothesis passed the test. ")
                   + ("" if big else "No hypothesis about big players passed, which is in line with research: ETF buyers follow the price rather than lead it. ")
                   + ("Confirmed signals go into the team of methods as a separate method, with the same gate as the rest: they affect the forecast only if they significantly improve it in the “as if live” test. " if nc else "")
                   + (f"{', '.join(vids)} are about the strength of swings, not direction: they point to a stormier week than the past alone would suggest. The correlation is weak, though, "
                      "and they would only go into the volatility model if they improved its forecast on held-out data. The minus signs for these hypotheses are normal: "
                      "a typical week is calmer than the average, which is pushed up by rare outbursts; what matters is the difference between low and high values." if vids else ""))
        decay = ""
        if any(r["id"] == "H7" and r["status"] == "confirmed" for r in CF["rows"]) and df is not None:
            lc = np.log(df["close"])
            m28, y7 = (lc - lc.shift(28)).shift(1), lc.shift(-7) - lc
            seg = []
            for a, b in (("2011", "2015"), ("2016", "2019"), ("2020", "2022"), ("2023", str(df.index[-1].year))):
                x, yy = m28.loc[a:b], y7.loc[a:b]
                k = x.notna() & yy.notna()
                q = pd.qcut(x[k].rank(method="first"), 3, labels=False)
                r_ = (np.exp(yy[k]) - 1) * 100
                seg.append(f"{a}–{b}: {r_[q == 2].mean():+.1f}% after rises vs {r_[q == 0].mean():+.1f}% after falls")
            decay = (f'<p class="note"><b>The trend (H7) passed the test, but it is weakening.</b> Average return in the following week: {"; ".join(seg)}. '
                     f'This is typical: a pattern that more and more people know about wears out over time. ')
            cs = None
            if ens and "7" in {str(k) for k in ens.get("horizons", {})}:
                v7 = ens["horizons"].get(7) or ens["horizons"].get("7")
                cs = next((r["skill"] for r in v7["table"] if r["model"] == "conf"), None)
            if cs is not None and cs <= 0.005:
                decay += (f'As a forecasting method the trend is in the team, but in the “as if live” test since 2015 it does not improve the one-week forecast '
                          f'({cs * 100:+.1f}%), so for now it does not affect the result. The correlation is real, but too weak to shift the whole price distribution.</p>')
            else:
                decay += 'The model uses it as one of the methods, and only when it improves the forecast in the “as if live” test.</p>'
        batches = sorted({r.get("batch", 1) for r in CF["rows"]})
        thrs = ", ".join(f'{min(r["alpha"] for r in CF["rows"] if r.get("batch", 1) == b):.4f}' + f" in batch {b}" for b in batches)
        tbl = (f'<div class="panel" style="margin-top:18px"><h3 class="ct">Confirmatory tests: big players, trend, options, stablecoins and the strength of swings</h3>'
               f'<p class="muted sm" style="margin:0 0 10px">We wrote down the hypotheses on Oct 29, 2026, before seeing the results, and we do not change them after the fact. '
               f'Batch 1 (H1–H6) is about big players, batch 2 (H7–H10) about the effects best described in research on Bitcoin, '
               f'batch 3 (H11–H13) about ideas from a review of research: stablecoins and the options market (the options skew in H13 is reconstructed from the Deribit trade archive since 2018), '
               f'batch 4 (H14) is the trend continued on new data, batch 5 (H15–H17) checks whether high leverage on the Binance exchange signals a crash (a 10% drop within 30 days), rather than direction, '
               f'and batch 6 (H18–H21, written down on Oct 31, 2026) is about the strength of swings: whether next week they will be bigger than past volatility alone predicts. '
               f'Each hypothesis has its direction written down in advance. To accept it as true, the result must be very unlikely to be luck '
               f'(p below {thrs}) and repeat in both halves of history. Data from day D enters the test only from day D+1. '
               f'For comparison we also give p with the Newey-West correction for overlapping returns: it usually comes out lower, which means our method is the more cautious one. '
               f'For each hypothesis we state how strong an effect the test could detect at all: “not confirmed” with a weak test does not mean “there is no effect”, only “it cannot be shown”. '
               f'H14 is H7 checked only on data after the day it was written down: it is the only clean proof. Confirmed signals also have a retirement rule, written down on Oct 30, 2026: '
               f'if at two consecutive annual reviews (from Oct 29, 2027) they are both insignificant (p &gt; 0.10) and weak (correlation &lt; 0.03) over the last 3 years, we stop using them.</p>'
               + f'<div class="tw"><table class="tbl"><tr><th></th><th>Hypothesis</th><th>Result</th><th>Details</th></tr>{rows}</table></div>'
               f'<p class="note"><b>Conclusion:</b> {verdict}</p>{decay}'
               f'<p class="muted sm" style="margin:8px 0 0">Sources: <a href="https://www.tftc.io/bitcoin-etf-flows" target="_blank" rel="noopener">TFTC Bitcoin ETF Tracker</a> (free Creative Commons licence), '
               f'<a href="https://bitbo.io/treasuries/microstrategy/" target="_blank" rel="noopener">Bitbo</a> (Strategy transactions according to 8-K reports), the public Hyperliquid API, wallet balances from the blockchain (mempool.space; El Salvador\'s addresses from bitcoin.gob.sv). '
               f'Mt. Gox has no tile: its bitcoins are spread across many addresses, and the full list cannot be reliably established.</p></div>')
    return f'<div class="tiles3">{tl}</div>{epi}{tbl}'


def markets_html(M, df):
    if not M:
        return '<div class="card empty"><b>No data from stock and bond markets</b><p class="muted">It will appear at the next update.</p></div>'
    f = M["feat"]

    def lastp(col):
        s = f[col].dropna() if col in f else pd.Series(dtype=float)
        return (float(s.iloc[-1]), float((s < s.iloc[-1]).mean())) if len(s) else (None, None)
    tiles = []
    v, p = lastp("breadth200")
    if v is not None:
        tiles.append(("How many S&P 500 companies are in an uptrend?", f"{v:.0f}%", p, "This many companies have a price above their 200-day average. Above 60% is a broad bull market, below 30% a broad sell-off."))
    v, p = lastp("avgcorr")
    if v is not None:
        tiles.append(("Are stocks moving as a “herd”?", f"{v:.2f}", p, "Average correlation between companies over the last 60 days. It tends to be high (above 0.4) in a panic, when everyone sells at once."))
    if "tnx" in df and "irx" in df and df["tnx"].notna().any():
        cv = float(df["tnx"].dropna().iloc[-1] - df["irx"].dropna().iloc[-1])
        sc = (df["tnx"] - df["irx"]).dropna()
        tiles.append(("US yield curve (10 years minus 3 months)", f"{cv:+.2f}" + " pp", float((sc < cv).mean()),
                      "A negative, or “inverted”, curve has historically come before recessions. A positive one is normal."))
    if "ratio_credit" in f:
        s = f["ratio_credit"].dropna()
        ch = float((s.iloc[-1] / s.iloc[-22] - 1) * 100)
        tiles.append(("Appetite for risk in bonds (month)", f"{ch:+.1f}%", float(((s / s.shift(21) - 1) * 100 < ch).mean()),
                      "How junk bonds have moved relative to safe government bonds. A fall means a flight from risk."))
    v, p = lastp("updown")
    if v is not None:
        tiles.append(("How many companies rose over the last week?", f"{v:.0f}%", p, "Share of S&P 500 companies that are higher than a week ago."))
    v, p = lastp("dispersion")
    if v is not None:
        tiles.append(("How different are the companies' results?", f"{v*100:.1f}%", p, "Spread of monthly returns between companies. A high value means the market is strongly picking winners and losers."))
    tl = "".join(f'<div class="card tile"><span class="muted sm">{a}</span><b>{b}</b>{_pctbar(c)}<span class="muted sm">{d}</span></div>' for a, b, c, d in tiles)
    cm = M["comove"]
    top = _bars([(r.name or r.sym, r.r) for r in cm["top"].itertuples()])
    bot = _bars([(r.name or r.sym, r.r) for r in cm["bottom"].itertuples()])
    sec = _bars([(k, v) for k, v in cm["sectors"].items()], scale=0.4)          # sector names come in English from the data
    bnd = _bars([(r.name or r.sym, r.r) for r in cm["bonds"].itertuples()], scale=0.4)
    A = M.get("analogs")
    an = ""
    if A and A.get("rows"):
        rows = "".join(f'<tr><td>{r["date"]}</td><td class="{"pos" if r["f30"] >= 0 else "neg"}">{r["f30"]:+.0f}%</td><td class="{"pos" if r["f90"] >= 0 else "neg"}">{r["f90"]:+.0f}%</td></tr>' for r in A["rows"])
        now = A["now"]
        nows = ", ".join(f'{n_} {now[k]:+.2f}' for k, n_ in (("QQQ", "Nasdaq"), ("TLT", "long bonds"), ("GLD", "gold"), ("UUP", "dollar")) if k in now)
        an = (f'<div class="panel" style="margin-top:18px"><h3 class="ct">When did Bitcoin\'s links with markets look like today?</h3>'
              f'<p class="muted sm" style="margin:0 0 10px">For each day we calculate how closely Bitcoin moved together with {A["n_keys"]} key markets (stocks, bonds, gold, the dollar, commodities, countries) over the last 90 days. '
              f'Today, for example: {nows}. We look for periods with the most similar pattern of these links (an “analogy of analogies”) and see what happened next. This is a curiosity, not a forecast.</p>'
              f'<table class="tbl"><tr><th>Period</th><th>Price after 30 days</th><th>Price after 90 days</th></tr>{rows}</table>'
              f'<p class="muted sm" style="margin:10px 0 0">In {A["up30"]*100:.0f}% of these periods the price was higher after 30 days; typically {A["med30"]:+.0f}% after 30 and {A["med90"]:+.0f}% after 90 days.</p></div>')
    return (f'<div class="tiles3">{tl}</div>'
            f'<div class="two2" style="margin-top:18px"><div class="panel"><h3 class="ct">Most similar to Bitcoin (last 2 years, weekly)</h3>'
            f'<p class="muted sm" style="margin:0 0 8px">Out of {cm["n"]} companies and funds. 1 = always together, 0 = independent.</p>{top}'
            f'<h3 class="ct" style="margin-top:18px">Most “in the opposite direction”</h3>{bot}</div>'
            f'<div class="panel"><h3 class="ct">US stock market sectors</h3><p class="muted sm" style="margin:0 0 8px">Average link between the sector\'s companies and Bitcoin.</p>{sec}'
            f'<h3 class="ct" style="margin-top:18px">Bonds</h3><p class="muted sm" style="margin:0 0 8px">Negative: when bonds get more expensive (a flight to safety), Bitcoin usually gets cheaper.</p>{bnd}</div></div>{an}')


def write_dashboard(res, status, log, stats, ens, E, df, S=None, Rg=None, MKT=None, CF=None, TCH=None, SIM=None, VE=None, JD=None, DT=None, TR=None):
    fr = res["frame"]
    reg_tiles, reg_verdict = regularity_html(Rg)
    scan_head, scan_body = scan_html(S)
    f = fr["features"]
    price = f["price"]
    ep = pd.Timestamp("1970-01-01")
    days = [(t - ep).days for t in price.index]
    comp = fr["composite"]
    ma200w = price.rolling(1400).mean()
    chg = lambda k: (price.iloc[-1] / price.iloc[-1 - k] - 1) * 100
    d1, d7, d30, d365 = chg(1), chg(7), chg(30), chg(365)
    ath = float(price.max())
    ath_date = price.idxmax()
    ok, tot, pills = sources_html(status)
    fe = res["features"]

    data = {
        "dates": days, "price": ser_list(price, 0), "real": ser_list(f["realprice"], 0), "ma200w": ser_list(ma200w, 0),
        "comp": ser_list(comp, 3), "halvings": [(pd.Timestamp(h) - ep).days for h in C.HALVINGS],
        "fc": [{"h": x["horizon"], **{k: x[k] for k in ("p5", "p10", "p25", "p50", "p75", "p90", "p95")}} for x in res["forecasts"]],
    }
    pdata = plan_data(df)
    if pdata:
        data["plan"] = pdata
    data["view"] = [x["view"]["p50"] for x in res["forecasts"]]
    data["up"] = [round(x["prob_up"], 4) for x in res["forecasts"]]
    evd = E["events"]
    sel = evd[((evd["kind"] == "curated") | ((evd["cat"] == "hack") & (evd["imp"] >= 2))) & (evd["date"] <= price.index[-1])]
    data["events"] = [{"d": (r.date - ep).days, "c": html.escape(_catlab(r.cat)), "t": html.escape(r.title), "r": (None if pd.isna(r.r3) else round(float(r.r3), 1))}
                      for r in sel.itertuples()]
    reg2_tiles, reg_rows = regions_html(ens, df, res)

    def delta(label, v):
        return f'<div class="dl"><span class="muted sm">{label}</span><b class="{"pos" if v >= 0 else "neg"}">{v:+.1f}%</b></div>'

    page = (TEMPLATE
            .replace("%%PRICE%%", n(res["price"]))
            .replace("%%ASOF%%", res["asof"])
            .replace("%%NOW%%", _now_str())
            .replace("%%NEVENTS%%", n(len(E["events"][E["events"]["date"] <= price.index[-1]]), 0))
            .replace("%%NSIG%%", str(sum(1 for p in res["signals"] if p["weight"] > 0)))
            .replace("%%DELTAS%%", delta("Today", d1) + delta("7 days", d7) + delta("30 days", d30) + delta("1 year", d365))
            .replace("%%ATH%%", f'{(res["price"]/ath-1)*100:.0f}% from the record of {n(ath)} USD ({ath_date:%Y-%m-%d})')
            .replace("%%GAUGE%%", baro_svg(res["composite"]))
            .replace("%%ISOBARS%%", isobars_svg())
            .replace("%%PRICERAW%%", f"{res['price']:.0f}")
            .replace("%%VERDICT%%", verdict_html(res))
            .replace("%%TILES%%", forecast_tiles(res, ens))
            .replace("%%SUMBULLETS%%", summary_bullets(res, ens, E))
            .replace("%%COMP%%", f'{res["composite"]:+.2f}')
            .replace("%%COMPPOS%%", f'{(res["composite"] + 1) / 2 * 100:.1f}')
            .replace("%%REGIME%%", res["regime"])
            .replace("%%REAL%%", n(fe.get("realprice")))
            .replace("%%MVRV%%", n(fe.get("mvrv"), 2))
            .replace("%%CYCLE%%", cycle_html(fe.get("halv_days")))
            .replace("%%FCARDS%%", forecast_cards(res))
            .replace("%%SIGNALS%%", signals_html(res))
            .replace("%%GLOSSARY%%", glossary_html())
            .replace("%%CONCLUSION%%", conclusion_text(ens))
            .replace("%%WHY%%", why_html(res, ens, E, df))
            .replace("%%MODELS%%", models_html(ens))
            .replace("%%MOVES%%", moves_html(E))
            .replace("%%STUDY%%", study_html(E))
            .replace("%%UPCOMING%%", upcoming_html(E))
            .replace("%%REGTILES%%", reg2_tiles)
            .replace("%%MARKETS%%", markets_html(MKT, df))
            .replace("%%EXTRA%%", extra_html())
            .replace("%%WHALES%%", whales_html(CF, df, ens))
            .replace("%%TOUCH%%", touch_html(TCH) + judge_html(JD) + volev_html(VE))
            .replace("%%BANDS%%", bands_html(ens))
            .replace("%%SIMPORT%%", trader_html(TR) + sim_html(SIM))
            .replace("%%SHORTH%%", short_html(ens) + dirtrack_html(DT) + exam_html(ens, TCH, SIM))
            .replace("%%NSERIES%%", n(S["summary"]["n_series"], 0) if S else "several hundred")
            .replace("%%REGULARITY%%", reg_tiles)
            .replace("%%REGVERDICT%%", reg_verdict)
            .replace("%%SCANHEAD%%", scan_head)
            .replace("%%SCANBODY%%", scan_body)
            .replace("%%REGROWS%%", reg_rows)
            .replace("%%VOLW%%", f"an average of moves over recent weeks {ens['vol']['weights']['ewma']*100:.0f}%, a model in which big swings “spill over” into the following days {ens['vol']['weights']['garch']*100:.0f}%, a model based on the daily price range {ens['vol']['weights']['har']*100:.0f}%"
                     + (f"; on top of that, with a weight of {ens['vol']['implied_w']*100:.0f}%, the swings priced on the Deribit options exchange, which improves the accuracy of the swing forecast by {ens['vol']['implied_gain']*100:.1f}%"
                        if ens['vol'].get('implied_w') else "")
                     + (f". We also tested volatility calculated from 5-minute data (Binance, since 2017): on held-out data from {ens['vol']['rv_test_from']} "
                        + (f"it improved the swing forecast by {ens['vol']['rv_gain_test']*100:.1f}%, so it is in the mix with a weight of {ens['vol']['rv_w']*100:.0f}%"
                           if ens['vol'].get('rv_w') else
                           f"it made the swing forecast worse by {abs(ens['vol'].get('rv_gain_test', 0))*100:.1f}%, so we do not use it")
                        if ens['vol'].get('rv_test_from') else ""))
            .replace("%%TRACK%%", track_html(stats, ens))
            .replace("%%CORR%%", corr_html(res["corr90"]))
            .replace("%%PILLS%%", pills)
            .replace("%%SRCOK%%", f"{ok}/{tot}")
            .replace("%%SRCCLS%%", "ok" if ok == tot else ("warn" if ok >= tot - 2 else "bad"))
            .replace("%%MIN_IC%%", str(C.MIN_IC))
            .replace("%%DATA%%", json.dumps(data, separators=(",", ":")))
            .replace("%%CHARTJS%%", CHARTJS))
    C.OUT_DIR.mkdir(exist_ok=True); C.DASHBOARD_HTML.parent.mkdir(exist_ok=True)
    C.DASHBOARD_HTML.write_text(page, encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Template (CSS + HTML + JS)
# ----------------------------------------------------------------------------------------------
TEMPLATE = (C.ROOT / "template_head.html").read_text(encoding="utf-8") + (C.ROOT / "template_body.html").read_text(encoding="utf-8")
CHARTJS = (C.ROOT / "chart.js").read_text(encoding="utf-8")
