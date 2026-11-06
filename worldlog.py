"""World event log: what happened, day by day (Wikipedia: Portal Current events, since 2011).

Raw entries are kept in yearly files (data/worldlog/YYYY.json.gz), and for analysis we build a compact daily series
(data/worldlog_daily.csv): the number of entries in each category and the number of entries containing given keywords."""
import gzip
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import config as C

DIR = C.DATA_DIR / "worldlog"
DAILY = C.DATA_DIR / "worldlog_daily.csv"
START = pd.Timestamp("2011-08-17")
UA = {"User-Agent": "bitcoin-barometer/1.0 (research; wywigacz@users.noreply.github.com)"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

CATS = {  # Wikipedia heading -> short key
    "armed conflicts and attacks": "armed", "business and economy": "biz", "disasters and accidents": "disaster",
    "international relations": "intl", "law and crime": "law", "politics and elections": "pol",
    "health and environment": "health", "science and technology": "sci", "arts and culture": "arts", "sports": "sport",
}
KEYWORDS = {
    "war": r"\b(war|invasion|invade\w*|missiles?|shell\w*|troops|offensive|airstrikes?|ceasefire|frontline|artillery)\b",
    "terror": r"\b(terror\w*|bomb\w*|shooting|gunmen|hostages?|suicide|massacre|stabb\w*)\b",
    "deaths": r"\b(killed|dead|deaths?|dies|died|casualties|fatalities)\b",
    "economy": r"\b(inflation|recession|gdp|unemployment|economy|economic|central bank|interest rates?|growth|deficit)\b",
    "markets": r"\b(stocks?|stock market|shares|index|markets?|nasdaq|dow|s&p|wall street|bonds?|yields?|currency|dollar|euro|yen)\b",
    "finstress": r"\b(bank|banks|default\w*|bailout|debt|bankrupt\w*|insolven\w*|crisis|collapse\w*|rescue|downgrade\w*)\b",
    "sanctions": r"\b(sanctions?|tariffs?|embargo|trade war|export ban|boycott|restrictions)\b",
    "elections": r"\b(election\w*|vote[sd]?|voting|referendum|ballot|polls?|candidate\w*)\b",
    "power": r"\b(coup|resign\w*|impeach\w*|prime minister|president|cabinet|dismiss\w*|sworn in|parliament)\b",
    "naturaldis": r"\b(earthquake\w*|hurricane\w*|typhoon\w*|flood\w*|wildfire\w*|tsunami|eruption|cyclone|storm|drought|heat ?wave)\b",
    "pandemic": r"\b(virus|pandemic|outbreak|vaccine\w*|epidemic|covid\w*|coronavirus|infections?|quarantine|lockdown)\b",
    "protest": r"\b(protest\w*|riot\w*|unrest|demonstrat\w*|strike[sd]?|clashes)\b",
    "energy": r"\b(oil|gas|opec|pipeline|energy|fuel|electricity|power plant|nuclear|blackout)\b",
    "tech": r"\b(artificial intelligence|ai|cyberattack\w*|hack\w*|data breach|google|apple|microsoft|meta|facebook|nvidia|tesla|amazon|openai|chip|semiconductor)\b",
    "crypto": r"\b(bitcoin|crypto\w*|ethereum|binance|coinbase|stablecoin|blockchain|ftx|tether)\b",
    "china": r"\b(china|chinese|beijing|hong kong|taiwan|xi jinping|pboc)\b",
    "ukraine": r"\b(ukrain\w*|russia\w*|putin|kyiv|moscow|kremlin|zelensky)\b",
    "mideast": r"\b(israel\w*|gaza|iran\w*|hamas|hezbollah|syria\w*|palestin\w*|lebanon|yemen|houthi\w*|saudi)\b",
    "usa": r"\b(united states|u\.s\.|trump|biden|obama|congress|senate|white house|pentagon|washington|fbi|supreme court)\b",
    "europe": r"\b(european union|eu|ecb|brexit|germany|france|britain|uk|italy|spain|nato)\b",
}
KRX = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}


def _page(day):
    return f"Portal:Current_events/{day.year}_{MONTHS[day.month - 1]}_{day.day}"


def _flatten_split(parts):
    """re.split with two groups returns [text, g1, g2, text, ...]; we reduce that to [text, name, text, name, ...]."""
    out = [parts[0]]
    for i in range(1, len(parts) - 2, 3):
        out.append(parts[i] if parts[i] is not None else parts[i + 1])
        out.append(parts[i + 2])
    return out


def parse_day(html_):
    """Returns {category: [leaf entries]} from the HTML of a day page."""
    # older pages: <div class="current-events-content-heading">, newer ones: <p><b>Category</b></p>
    parts = re.split(r'<div[^>]*current-events-content-heading[^>]*>(.*?)</div>|<p><b>(.*?)</b>\s*</p>', html_, flags=re.S)
    parts = [p for p in _flatten_split(parts)]
    out = {}
    for i in range(1, len(parts) - 1, 2):
        name = re.sub(r"<[^>]+>", "", parts[i]).strip().lower()
        key = CATS.get(name, "other")
        items = []
        for m in re.finditer(r"<li>((?:(?!<li>|</li>|<ul>|</ul>).)*)</li>", parts[i + 1], flags=re.S):     # only the leaves of the list
            raw = m.group(1)
            t = re.sub(r"<[^>]+>", "", raw)
            t = re.sub(r"\s+", " ", t).strip()
            t = re.sub(r"\s*\([^()]*\)\s*$", "", t)
            if len(t) >= 30:
                items.append(t[:170])
        if items:
            out.setdefault(key, []).extend(items[:14])
    return out


def fetch_day(day, session):
    for i in range(3):
        try:
            r = session.get("https://en.wikipedia.org/w/api.php", params={"action": "parse", "page": _page(day), "prop": "text", "format": "json", "formatversion": 2}, timeout=40)
            if r.status_code == 200:
                j = r.json()
                if "parse" in j:
                    return parse_day(j["parse"]["text"])
                return {}
            if r.status_code in (429, 500, 502, 503):
                time.sleep(3 * (i + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return None


def _year_path(y):
    return DIR / f"{y}.json.gz"


def load_year(y):
    p = _year_path(y)
    if p.exists():
        try:
            return json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
        except Exception:
            return {}
    return {}


def save_year(y, data):
    DIR.mkdir(parents=True, exist_ok=True)
    _year_path(y).write_bytes(gzip.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9))


def update(last_day, budget_s=1e9, workers=6, verbose=True):
    """Fetches the missing days up to last_day. Days already saved are skipped (you can stop and resume)."""
    t0 = time.time()
    days = pd.date_range(START, last_day)
    data = {y: load_year(y) for y in sorted(set(days.year))}
    todo = [d for d in days if d.strftime("%Y-%m-%d") not in data[d.year]]
    # we refresh the last 3 days (entries are sometimes added with a delay)
    recent = [d for d in days[-3:] if d not in todo]
    todo += recent
    if verbose:
        print(f"  world event log: {len(todo)} days to fetch", flush=True)
    s = requests.Session()
    s.headers.update(UA)
    done = 0
    for k in range(0, len(todo), workers * 4):
        if time.time() - t0 > budget_s:
            break
        batch = todo[k:k + workers * 4]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(lambda d: (d, fetch_day(d, s)), batch))
        for d, r in res:
            if r is not None:
                data[d.year][d.strftime("%Y-%m-%d")] = r
                done += 1
        if (k // (workers * 4)) % 10 == 0:
            for y, v in data.items():
                save_year(y, v)
            if verbose:
                print(f"  ... {done}/{len(todo)} days, {time.time() - t0:.0f}s", flush=True)
        time.sleep(0.4)
    for y, v in data.items():
        save_year(y, v)
    return data


def build_daily(index):
    """Daily series: number of entries per category and with keywords (for the scanner)."""
    rows = {}
    for y in sorted(set(index.year)):
        for day, cats in load_year(y).items():
            rec = {"n_total": sum(len(v) for v in cats.values())}
            for c in set(CATS.values()) | {"other"}:
                rec[f"c_{c}"] = len(cats.get(c, []))
            texts = [t for v in cats.values() for t in v]
            for k, rx in KRX.items():
                rec[f"k_{k}"] = sum(1 for t in texts if rx.search(t))
            rows[pd.Timestamp(day)] = rec
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    return df.reindex(index.intersection(df.index)) if False else df


def save_daily(index):
    df = build_daily(index)
    C.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DAILY)
    return df


def day_headlines(day, per=2):
    """The day's most important entries (to show next to big moves): a few each from business, politics and conflicts."""
    d = pd.Timestamp(day)
    cats = load_year(d.year).get(d.strftime("%Y-%m-%d"), {})
    out = []
    for key in ("biz", "pol", "armed", "intl", "disaster", "law", "health", "sci"):
        for t in cats.get(key, [])[:per]:
            out.append({"c": key, "t": t})
    return out[:7]
