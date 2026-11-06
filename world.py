"""A broad set of world data series (markets, commodities, currencies, political attention, weather, the Sun, earthquakes, the Moon)
for the relationship scanner. All free and without API keys. If a single series fails, it is skipped."""
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import config as C
import extra as XT
import markets as MK

WORLD_CSV = C.DATA_DIR / "world.csv"
FAILS = C.DATA_DIR / "world_fails.json"
UA = {"User-Agent": "bitcoin-barometer/1.0 (research; wywigacz@users.noreply.github.com)"}

# name -> (Yahoo symbol, label)
YAHOO = {
    # world stock markets
    "nikkei": ("^N225", "Nikkei (Japan)"), "hsi": ("^HSI", "Hang Seng (Hong Kong)"), "shanghai": ("000001.SS", "Shanghai (China)"),
    "kospi": ("^KS11", "KOSPI (South Korea)"), "ftse": ("^FTSE", "FTSE 100 (UK)"), "dax": ("^GDAXI", "DAX (Germany)"),
    "cac": ("^FCHI", "CAC 40 (France)"), "bovespa": ("^BVSP", "Bovespa (Brazil)"), "nifty": ("^NSEI", "Nifty (India)"),
    "taiex": ("^TWII", "TAIEX (Taiwan)"), "asx": ("^AXJO", "ASX 200 (Australia)"), "eem": ("EEM", "Emerging markets"),
    "tur": ("TUR", "Turkey (ETF)"), "argt": ("ARGT", "Argentina (ETF)"), "eza": ("EZA", "South Africa (ETF)"), "ewz": ("EWZ", "Brazil (ETF)"),
    "ewj": ("EWJ", "Japan (ETF)"), "fxi": ("FXI", "China large cap (ETF)"), "inda": ("INDA", "India (ETF)"),
    # currencies
    "try": ("TRY=X", "USD/TRY (Turkish lira)"), "ars": ("ARS=X", "USD/ARS (Argentine peso)"), "ngn": ("NGN=X", "USD/NGN (naira)"),
    "pln": ("PLN=X", "USD/PLN (Polish zloty)"), "brl": ("BRL=X", "USD/BRL"), "eur": ("EURUSD=X", "EUR/USD"), "gbp": ("GBPUSD=X", "GBP/USD"), "aud": ("AUDUSD=X", "AUD/USD"),
    "cad": ("CAD=X", "USD/CAD"), "chf": ("CHF=X", "USD/CHF"), "rub": ("RUB=X", "USD/RUB"), "inr": ("INR=X", "USD/INR"),
    "zar": ("ZAR=X", "USD/ZAR"), "mxn": ("MXN=X", "USD/MXN"), "cnh": ("CNY=X", "USD/CNY"), "idr": ("IDR=X", "USD/IDR"),
    # commodities and energy
    "natgas": ("NG=F", "Natural gas"), "copper": ("HG=F", "Copper"), "silver": ("SI=F", "Silver"), "platinum": ("PL=F", "Platinum"),
    "wheat": ("ZW=F", "Wheat"), "corn": ("ZC=F", "Corn"), "sugar": ("SB=F", "Sugar"), "coffee": ("KC=F", "Coffee"),
    "uranium": ("URA", "Uranium (ETF)"), "brent": ("BZ=F", "Brent oil"),
    # sectors, bonds, credit, volatility
    "xle": ("XLE", "US energy"), "xlf": ("XLF", "US financials"), "xlk": ("XLK", "US technology"), "xlu": ("XLU", "US utilities"),
    "xlv": ("XLV", "US health care"), "defense": ("ITA", "US defense"), "clean": ("ICLN", "Clean energy"), "regbank": ("KRE", "US regional banks"),
    "semis": ("SMH", "Semiconductors"), "arkk": ("ARKK", "ARK Innovation"), "gdx": ("GDX", "Gold miners"), "vnq": ("VNQ", "US real estate"),
    "tip": ("TIP", "Inflation-linked bonds"), "lqd": ("LQD", "Investment-grade corporate bonds"), "jnk": ("JNK", "Junk bonds"),
    "tyx": ("^TYX", "30-year yield"), "fvx": ("^FVX", "5-year yield"), "vvix": ("^VVIX", "VVIX (volatility of volatility)"),
    "skew": ("^SKEW", "CBOE SKEW"), "russell": ("^RUT", "Russell 2000"), "nvda": ("NVDA", "Nvidia"),
    # world stock indices
    "i_dji": ("^DJI", "Dow Jones"), "i_tsx": ("^GSPTSE", "TSX (Canada)"), "i_mxx": ("^MXX", "IPC (Mexico)"), "i_merv": ("^MERV", "Merval (Argentina)"),
    "i_stoxx": ("^STOXX50E", "Euro Stoxx 50"), "i_aex": ("^AEX", "AEX (Netherlands)"), "i_ibex": ("^IBEX", "IBEX 35 (Spain)"), "i_mib": ("FTSEMIB.MI", "FTSE MIB (Italy)"),
    "i_smi": ("^SSMI", "SMI (Switzerland)"), "i_omx": ("^OMX", "OMX (Sweden)"), "i_bfx": ("^BFX", "BEL 20 (Belgium)"), "i_psi": ("PSI20.LS", "PSI (Portugal)"),
    "i_atx": ("^ATX", "ATX (Austria)"), "i_wig": ("WIG.WA", "WIG (Poland)"), "i_szse": ("399001.SZ", "Shenzhen (China)"), "i_sti": ("^STI", "STI (Singapore)"),
    "i_klse": ("^KLSE", "KLCI (Malaysia)"), "i_jkse": ("^JKSE", "Jakarta (Indonesia)"), "i_bse": ("^BSESN", "Sensex (India)"), "i_nz": ("^NZ50", "NZX 50 (New Zealand)"),
    "i_ta": ("^TA125.TA", "TA-125 (Israel)"), "i_xu": ("XU100.IS", "BIST 100 (Turkey)"), "i_jse": ("^J203.JO", "JSE (South Africa)"), "i_omxc": ("^OMXC25", "OMX C25 (Denmark)"),
    "i_omxh": ("^OMXH25", "OMX H25 (Finland)"),
    # fear in different markets
    "v_vix9d": ("^VIX9D", "Stock market fear, 9 days (VIX9D)"), "v_vix3m": ("^VIX3M", "Stock market fear, 3 months (VIX3M)"), "v_vxn": ("^VXN", "Nasdaq fear (VXN)"),
    "v_ovx": ("^OVX", "Oil market fear (OVX)"), "v_gvz": ("^GVZ", "Gold market fear (GVZ)"),
    # rates and commodities
    "zq": ("ZQ=F", "Fed rate expectations (futures)"), "zt": ("ZT=F", "2-year Treasuries (futures)"), "palladium": ("PA=F", "Palladium"),
    "aluminium": ("ALI=F", "Aluminium"), "lumber": ("LBR=F", "Lumber"), "cocoa": ("CC=F", "Cocoa"), "cotton": ("CT=F", "Cotton"), "ojuice": ("OJ=F", "Orange juice"),
    "cattle": ("LE=F", "Cattle"), "freight": ("BDRY", "Dry bulk shipping"),
    # emerging market currencies
    "krw": ("KRW=X", "USD/KRW (Korea)"), "twd": ("TWD=X", "USD/TWD (Taiwan)"), "thb": ("THB=X", "USD/THB (Thailand)"), "php": ("PHP=X", "USD/PHP (Philippines)"),
    "egp": ("EGP=X", "USD/EGP (Egypt)"), "pkr": ("PKR=X", "USD/PKR (Pakistan)"), "cop": ("COP=X", "USD/COP (Colombia)"), "clp": ("CLP=X", "USD/CLP (Chile)"),
    "pen": ("PEN=X", "USD/PEN (Peru)"), "ils": ("ILS=X", "USD/ILS (Israel)"), "hkd": ("HKD=X", "USD/HKD (Hong Kong)"), "uah": ("UAH=X", "USD/UAH (Ukraine)"), "vnd": ("VND=X", "USD/VND (Vietnam)"),
    # other cryptocurrencies and CME futures
    "btcf": ("BTC=F", "CME Bitcoin futures"), "c_eth": ("ETH-USD", "Ethereum"), "c_sol": ("SOL-USD", "Solana"), "c_xrp": ("XRP-USD", "XRP"), "c_bnb": ("BNB-USD", "BNB"),
    "c_doge": ("DOGE-USD", "Dogecoin"), "c_ada": ("ADA-USD", "Cardano"), "c_ltc": ("LTC-USD", "Litecoin"), "c_trx": ("TRX-USD", "Tron"), "c_avax": ("AVAX-USD", "Avalanche"),
    "c_link": ("LINK-USD", "Chainlink"), "c_dot": ("DOT-USD", "Polkadot"), "c_xmr": ("XMR-USD", "Monero"), "c_bch": ("BCH-USD", "Bitcoin Cash"),
    # crypto-related companies
    "coin": ("COIN", "Coinbase"), "mstr": ("MSTR", "Strategy (MSTR)"), "mara": ("MARA", "Marathon (miner)"), "riot": ("RIOT", "Riot (miner)"),
    "clsk": ("CLSK", "CleanSpark (miner)"), "hut": ("HUT", "Hut 8 (miner)"), "cifr": ("CIFR", "Cipher (miner)"), "iren": ("IREN", "IREN (miner)"),
    "wulf": ("WULF", "TeraWulf (miner)"), "xyz": ("XYZ", "Block (formerly Square)"),
    # energy: electricity cost for miners and inflation pressure
    "ttf": ("TTF=F", "European gas (TTF)"), "coal": ("MTF=F", "Coal (Rotterdam)"), "gasoline": ("RB=F", "Gasoline (US)"), "heatoil": ("HO=F", "Heating oil (US)"),
}
# name -> (English Wikipedia article, label)
WIKI = {
    "trump": ("Donald_Trump", "Interest: Donald Trump"), "biden": ("Joe_Biden", "Interest: Joe Biden"), "recession": ("Recession", "Interest: recession"),
    "inflation": ("Inflation", "Interest: inflation"), "fed": ("Federal_Reserve", "Interest: the Fed"), "crash": ("Stock_market_crash", "Interest: stock market crash"),
    "bankrun": ("Bank_run", "Interest: bank runs"), "russia": ("Russia", "Interest: Russia"), "ukraine": ("Ukraine", "Interest: Ukraine"),
    "china": ("China", "Interest: China"), "iran": ("Iran", "Interest: Iran"), "israel": ("Israel", "Interest: Israel"), "taiwan": ("Taiwan", "Interest: Taiwan"),
    "musk": ("Elon_Musk", "Interest: Elon Musk"), "crypto": ("Cryptocurrency", "Interest: cryptocurrencies"), "ethereum": ("Ethereum", "Interest: Ethereum"),
    "gold": ("Gold", "Interest: gold"), "covid": ("COVID-19", "Interest: COVID-19"), "tariff": ("Tariff", "Interest: tariffs"),
    "oilprice": ("Price_of_oil", "Interest: oil prices"), "heatwave": ("Heat_wave", "Interest: heat waves"), "hurricane": ("Tropical_cyclone", "Interest: cyclones"),
    "earthquake": ("Earthquake", "Interest: earthquakes"), "war": ("War", "Interest: war"), "sanctions": ("International_sanctions", "Interest: sanctions"),
    "ai": ("Artificial_intelligence", "Interest: artificial intelligence"), "nft": ("Non-fungible_token", "Interest: NFTs"), "scam": ("Ponzi_scheme", "Interest: Ponzi schemes"),
    "unemployment": ("Unemployment", "Interest: unemployment"), "debt": ("National_debt_of_the_United_States", "Interest: US debt"),
    # people and institutions
    "powell": ("Jerome_Powell", "Interest: Jerome Powell (Fed)"), "yellen": ("Janet_Yellen", "Interest: Janet Yellen"), "saylor": ("Michael_Saylor", "Interest: Michael Saylor"),
    "sbf": ("Sam_Bankman-Fried", "Interest: Sam Bankman-Fried"), "cz": ("Changpeng_Zhao", "Interest: Changpeng Zhao (Binance)"), "satoshi": ("Satoshi_Nakamoto", "Interest: Satoshi Nakamoto"),
    "buterin": ("Vitalik_Buterin", "Interest: Vitalik Buterin"), "gensler": ("Gary_Gensler", "Interest: Gary Gensler (SEC)"), "xi": ("Xi_Jinping", "Interest: Xi Jinping"),
    "putin": ("Vladimir_Putin", "Interest: Vladimir Putin"), "bukele": ("Nayib_Bukele", "Interest: Nayib Bukele (El Salvador)"), "milei": ("Javier_Milei", "Interest: Javier Milei"),
    "blackrock": ("BlackRock", "Interest: BlackRock"), "binance": ("Binance", "Interest: Binance"), "coinbase_w": ("Coinbase", "Interest: Coinbase"), "tether_w": ("Tether_(cryptocurrency)", "Interest: Tether"),
    "sec": ("U.S._Securities_and_Exchange_Commission", "Interest: SEC"), "imf": ("International_Monetary_Fund", "Interest: IMF"), "ecb": ("European_Central_Bank", "Interest: ECB"),
    "nvidia": ("Nvidia", "Interest: Nvidia"), "openai": ("OpenAI", "Interest: OpenAI"), "chatgpt": ("ChatGPT", "Interest: ChatGPT"), "tesla": ("Tesla,_Inc.", "Interest: Tesla"),
    # financial concepts and bubbles
    "bubble": ("Economic_bubble", "Interest: speculative bubbles"), "tulip": ("Tulip_mania", "Interest: tulip mania"), "dotcom": ("Dot-com_bubble", "Interest: dot-com bubble"),
    "gfc": ("2008_financial_crisis", "Interest: 2008 crisis"), "depression": ("Great_Depression", "Interest: Great Depression"), "hyperinf": ("Hyperinflation", "Interest: hyperinflation"),
    "stagflation": ("Stagflation", "Interest: stagflation"), "qe": ("Quantitative_easing", "Interest: quantitative easing"), "goldstd": ("Gold_standard", "Interest: gold standard"),
    "cbdc": ("Central_bank_digital_currency", "Interest: central bank digital currencies"), "stablecoin": ("Stablecoin", "Interest: stablecoins"),
    "halving": ("Bitcoin_halving", "Interest: halving"), "blockchain": ("Blockchain", "Interest: blockchain"), "defi": ("Decentralized_finance", "Interest: DeFi"),
    "mining": ("Cryptocurrency_mining", "Interest: crypto mining"), "etf": ("Exchange-traded_fund", "Interest: ETFs"), "shortsq": ("Short_squeeze", "Interest: short squeeze"),
    "gamestop": ("GameStop_short_squeeze", "Interest: GameStop"), "ftx": ("FTX", "Interest: FTX"), "mtgox": ("Mt._Gox", "Interest: Mt. Gox"), "silk": ("Silk_Road_(marketplace)", "Interest: Silk Road"),
    "ransomware": ("Ransomware", "Interest: ransomware"), "hack": ("Security_hacker", "Interest: hackers"),
    # technologies that could threaten or help BTC
    "quantum": ("Quantum_computing", "Interest: quantum computers"), "metaverse": ("Metaverse", "Interest: metaverse"), "web3": ("Web3", "Interest: Web3"),
    # world, climate, disasters
    "climate": ("Climate_change", "Interest: climate change"), "elnino": ("El_Niño–Southern_Oscillation", "Interest: El Niño"), "wildfire": ("Wildfire", "Interest: wildfires"),
    "flood": ("Flood", "Interest: floods"), "drought": ("Drought", "Interest: droughts"), "pandemic": ("Pandemic", "Interest: pandemics"), "nuclear": ("Nuclear_weapon", "Interest: nuclear weapons"),
    "ww3": ("World_War_III", "Interest: World War III"), "brexit": ("Brexit", "Interest: Brexit"), "venezuela": ("Venezuela", "Interest: Venezuela"),
    "argentina": ("Argentina", "Interest: Argentina"), "turkey": ("Turkey", "Interest: Turkey"), "nkorea": ("North_Korea", "Interest: North Korea"),
    "elsalvador": ("El_Salvador", "Interest: El Salvador"), "india": ("India", "Interest: India"), "japan": ("Japan", "Interest: Japan"),
    "solar": ("Solar_flare", "Interest: solar flares"), "eclipse": ("Solar_eclipse", "Interest: solar eclipses"), "ufo": ("Unidentified_flying_object", "Interest: UFOs"),
    "astrology": ("Astrology", "Interest: astrology"), "mercury_r": ("Apparent_retrograde_motion", "Interest: Mercury retrograde"),
    "worldcup": ("FIFA_World_Cup", "Interest: World Cup"), "olympics": ("Olympic_Games", "Interest: Olympics"), "superbowl": ("Super_Bowl", "Interest: Super Bowl"),
    "lottery": ("Lottery", "Interest: lotteries"), "casino": ("Casino", "Interest: casinos"), "poverty": ("Poverty", "Interest: poverty"),
    "layoff": ("Layoff", "Interest: layoffs"), "housing": ("Real_estate_bubble", "Interest: housing bubble"), "bankrupt": ("Bankruptcy", "Interest: bankruptcies"),
}
# name -> (lat., lon., label) regions linked to mining and energy demand
WEATHER = {
    "texas": (32.45, -99.7, "Texas (mining)"), "sichuan": (30.6, 104.0, "Sichuan (China)"), "xinjiang": (43.8, 87.6, "Xinjiang (China)"),
    "kazakh": (51.2, 71.4, "Kazakhstan"), "iceland": (64.1, -21.9, "Iceland"), "siberia": (52.3, 104.3, "Siberia (Irkutsk)"),
    "nyc": (40.7, -74.0, "New York"), "tokyo": (35.7, 139.7, "Tokyo"), "seoul": (37.6, 127.0, "Seoul"),
}
LABELS = {f"y_{k}": v[1] for k, v in YAHOO.items()}
LABELS.update({f"wk_{k}": v[1] for k, v in WIKI.items()})
LABELS.update({f"wx_{k}_t": f"Temperature: {v[2]}" for k, v in WEATHER.items()})
LABELS.update({f"wx_{k}_p": f"Rainfall: {v[2]}" for k, v in WEATHER.items()})
LABELS.update({"sunspots": "Sunspots", "quakes": "Earthquakes M6+ (count)", "moon_sin": "Moon phase (sin)", "moon_cos": "Moon phase (cos)",
               "season_sin": "Yearly seasonality (sin)", "season_cos": "Yearly seasonality (cos)"})
LABELS.update({f"dow_{i}": f"Day of week: {d}" for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])})
LABELS.update({"cal_monthend": "Calendar: month end", "cal_quarterend": "Calendar: quarter end", "cal_lunarny": "Calendar: Chinese New Year (±7 days)",
               "cal_taxday": "Calendar: US tax day", "cal_xmas": "Calendar: Christmas holidays",
               "cal_mercury": "Wild card: Mercury retrograde", "cal_fri13": "Wild card: Friday the 13th", "cal_goldenweek": "Calendar: China's Golden Week",
               "cal_thanksgiving": "Calendar: Thanksgiving (US)", "cal_ramadan": "Calendar: Ramadan", "cal_sport": "Wild card: World Cup, Olympics, Super Bowl",
               "cal_dst": "Calendar: US clock change"})
LABELS.update({k: v[0] for k, v in XT.KIND.items()})
DSER = {  # series from the daily BTC table added to the scanner: (label, kind)
    "d_ntx": ("BTC network: number of transactions", "attn"), "d_fees": ("BTC network: transaction fees (USD)", "attn"), "d_mempool": ("BTC network: transaction queue (mempool)", "attn"),
    "d_txvol": ("BTC network: value transferred", "attn"), "d_uaddr": ("BTC network: unique addresses", "attn"), "d_difficulty": ("BTC network: mining difficulty", "attn"),
    "d_hash": ("BTC network: hash rate", "attn"), "d_minerrev": ("BTC network: miner revenue", "attn"), "d_adr": ("BTC network: active addresses", "attn"),
    "d_volume": ("BTC trading volume (Bitstamp)", "attn"), "d_cbprem": ("Coinbase premium (US demand)", "lvl"), "d_basis": ("CME price vs spot price", "lvl"),
    "d_bfund": ("Leverage funding rate (Bybit, 7 days)", "lvl"), "d_mvrv": ("Holders' profit (MVRV)", "lvl"), "d_fng": ("Fear and greed (crypto)", "lvl"),
    "d_ethbtc": ("Ethereum vs Bitcoin", "ratio"), "d_altbtc": ("Altcoin basket vs Bitcoin", "ratio"),
}
LABELS.update({k: v[0] for k, v in DSER.items()})
STATIC = {"moon_sin", "moon_cos", "season_sin", "season_cos", "cal_monthend", "cal_quarterend", "cal_lunarny", "cal_taxday", "cal_xmas"} | {f"dow_{i}" for i in range(7)}
WL = {"n_total": "all events", "c_armed": "armed conflicts and attacks", "c_biz": "business and economy", "c_disaster": "disasters and accidents",
      "c_intl": "international relations", "c_law": "law and crime", "c_pol": "politics and elections", "c_health": "health and environment",
      "c_sci": "science and technology", "k_war": "war", "k_terror": "terrorism", "k_deaths": "deaths", "k_economy": "economy (inflation, rates)",
      "k_markets": "markets and exchanges", "k_finstress": "financial stress (banks, debt)", "k_sanctions": "sanctions and tariffs", "k_elections": "elections",
      "k_power": "changes of government", "k_naturaldis": "natural disasters", "k_pandemic": "pandemics and diseases", "k_protest": "protests and strikes",
      "k_energy": "energy and commodities", "k_tech": "technology and big tech", "k_crypto": "cryptocurrencies", "k_china": "China", "k_ukraine": "Russia and Ukraine",
      "k_mideast": "Middle East", "k_usa": "USA", "k_europe": "Europe"}
LABELS.update({f"we_{k}": f"World: {v}" for k, v in WL.items()})
LABELS.update({f"m_{k}": f"{v} (fund)" for k, v in MK.ETF.items()})
LABELS.update({f"mk_ratio_{k}": v[2] for k, v in MK.RATIOS.items()})
LABELS.update({f"mk_{k}": v for k, v in MK.LEVELS.items()})
try:
    _it = pd.read_csv(MK.INTL_CSV)
    LABELS.update({f"g_{a}": f"{b} ({c})" for a, b, c in zip(_it["Symbol"], _it["Name"], _it["Index"])})
except Exception:
    pass
try:
    _sp = pd.read_csv(MK.SP500)
    LABELS.update({f"s_{a}": f"{b} (stock)" for a, b in zip(_sp["Symbol"], _sp["Security"])})
except Exception:
    pass
ATTN = {f"wk_{k}" for k in WIKI} | {"quakes", "sunspots"} | {f"we_{k}" for k in WL}
WX = {f"wx_{k}_t" for k in WEATHER} | {f"wx_{k}_p" for k in WEATHER}
STATIC = {"moon_sin", "moon_cos", "season_sin", "season_cos"} | {f"dow_{i}" for i in range(7)}


def _get(url, timeout=30, tries=2):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(1.0 * (i + 1))
    return None


def _day(ts):
    return pd.Timestamp(int(ts), unit="s").normalize()


def _yahoo(item, p1, p2):
    name, (sym, _) = item
    r = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}?period1={p1}&period2={p2}&interval=1d")
    if r is None:
        return name, None
    try:
        res = r.json()["chart"]["result"][0]
        s = pd.Series({_day(t): c for t, c in zip(res["timestamp"], res["indicators"]["quote"][0]["close"]) if c is not None}, dtype=float)
        return f"y_{name}", s.sort_index()
    except Exception:
        return name, None


def _wiki(item, start, end):
    name, (art, _) = item
    r = _get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{art}/daily/{start}/{end}", timeout=40)
    if r is None:
        return name, None
    try:
        return f"wk_{name}", pd.Series({pd.Timestamp(i["timestamp"][:8]): float(i["views"]) for i in r.json()["items"]}).sort_index()
    except Exception:
        return name, None


def _weather(item, start, end):
    name, (la, lo, _) = item
    r = _get(f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}&start_date={start}&end_date={end}"
             "&daily=temperature_2m_mean,precipitation_sum&timezone=UTC", timeout=60)
    if r is None:
        return []
    try:
        d = r.json()["daily"]
        t = pd.to_datetime(d["time"])
        return [(f"wx_{name}_t", pd.Series(d["temperature_2m_mean"], index=t, dtype=float)),
                (f"wx_{name}_p", pd.Series(d["precipitation_sum"], index=t, dtype=float))]
    except Exception:
        return []


def fetch_world(since=None):
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = pd.Timestamp("2011-08-01") if since is None else pd.Timestamp(since) - pd.Timedelta(days=10)
    p1, p2 = int(start.timestamp()), int(time.time()) + 86400
    out, fails = {}, []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, s in ex.map(lambda it: _yahoo(it, p1, p2), YAHOO.items()):
            (out.__setitem__(k, s) if s is not None else fails.append(k))
    ws = max(start, pd.Timestamp("2015-07-01")).strftime("%Y%m%d")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, s in ex.map(lambda it: _wiki(it, ws, today.strftime("%Y%m%d")), WIKI.items()):
            (out.__setitem__(k, s) if s is not None else fails.append(k))
    for item in WEATHER.items():
        for k, s in _weather(item, start.strftime("%Y-%m-%d"), (today - pd.Timedelta(days=6)).strftime("%Y-%m-%d")):
            out[k] = s
        time.sleep(0.3)
    r = _get("https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv", timeout=60)
    if r is not None:
        try:
            df = pd.read_csv(io.StringIO(r.text), sep=";", header=None, usecols=[0, 1, 2, 4], names=["y", "m", "d", "sn"])
            df = df[df["sn"] >= 0]
            out["sunspots"] = pd.Series(df["sn"].values, index=pd.to_datetime(dict(year=df["y"], month=df["m"], day=df["d"])), dtype=float)
        except Exception:
            fails.append("sunspots")
    r = _get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={start:%Y-%m-%d}&endtime={today:%Y-%m-%d}&minmagnitude=6&orderby=time-asc", timeout=60)
    if r is not None:
        try:
            q = pd.read_csv(io.StringIO(r.text), usecols=["time"])
            cnt = pd.to_datetime(q["time"]).dt.tz_localize(None).dt.normalize().value_counts()
            idx = pd.date_range(start, today)
            out["quakes"] = cnt.reindex(idx, fill_value=0).astype(float)
        except Exception:
            fails.append("quakes")
    return out, fails


def update_world(full=False):
    old = pd.read_csv(WORLD_CSV, index_col=0, parse_dates=True) if WORLD_CSV.exists() else pd.DataFrame()
    tried = set(json.loads(FAILS.read_text())) if FAILS.exists() else set()   # series the source does not have: we do not retry the full download over and over
    missing = [c for c in [f"y_{k}" for k in YAHOO] + [f"wk_{k}" for k in WIKI] if c not in old.columns and c.split("_", 1)[1] not in tried]
    since = None if (full or old.empty or missing) else old.index.max() - pd.Timedelta(days=30)
    data, fails = fetch_world(since)
    if since is None:
        FAILS.write_text(json.dumps(sorted(fails)))
    new = pd.DataFrame(data)
    merged = new.combine_first(old) if len(old) else new
    merged = merged.sort_index().loc["2011-08-01":]
    C.DATA_DIR.mkdir(exist_ok=True)
    merged.to_csv(WORLD_CSV, float_format="%.8g")
    return merged, fails


def static_series(index):
    d = pd.DataFrame(index=index)
    days = (index - pd.Timestamp("2000-01-06 18:14")).days.to_numpy(float)
    ph = (days / 29.530588853) % 1
    d["moon_sin"], d["moon_cos"] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)
    doy = index.dayofyear.to_numpy(float)
    d["season_sin"], d["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for i in range(7):
        d[f"dow_{i}"] = (index.dayofweek == i).astype(float)
    # calendar: month/quarter end, Chinese New Year, US tax day, Christmas holidays
    me = index.to_series().dt.is_month_end
    d["cal_monthend"] = (index.to_series().shift(-3).dt.month != index.month).astype(float).to_numpy()
    d["cal_quarterend"] = ((index.month % 3 == 0) & (d["cal_monthend"] > 0)).astype(float)
    lny = pd.to_datetime(["2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25",
                          "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06"])
    d["cal_lunarny"] = np.array([float(any(-7 <= (t - x).days <= 7 for x in lny)) for t in index])
    d["cal_taxday"] = ((index.month == 4) & (index.day >= 8) & (index.day <= 18)).astype(float)
    d["cal_xmas"] = (((index.month == 12) & (index.day >= 20)) | ((index.month == 1) & (index.day <= 3))).astype(float)
    # “wild cards”: things that are probably unrelated, but people believe in them or behave differently then; we test them to rule them out or catch them
    d["cal_mercury"] = mercury_retrograde(index)
    d["cal_fri13"] = ((index.day == 13) & (index.dayofweek == 4)).astype(float)
    d["cal_goldenweek"] = ((index.month == 10) & (index.day <= 7)).astype(float)                  # China's Golden Week
    thx = {y: pd.Timestamp(f"{y}-11-01") + pd.offsets.Week(weekday=3) * 4 - pd.Timedelta(days=7 if pd.Timestamp(f"{y}-11-01").dayofweek == 3 else 0)
           for y in range(index.year.min(), index.year.max() + 1)}
    d["cal_thanksgiving"] = np.array([float(0 <= (t - thx[t.year]).days <= 4) for t in index])
    d["cal_ramadan"] = _in_windows(index, RAMADAN, 29)
    d["cal_sport"] = np.maximum(_in_ranges(index, WORLDCUP + OLYMPICS), _in_windows(index, SUPERBOWL, 1, before=3))
    dst = []
    for y in range(index.year.min(), index.year.max() + 1):                                          # US clock change (2nd Sunday of March, 1st Sunday of November)
        dst += [pd.Timestamp(f"{y}-03-01") + pd.offsets.Week(weekday=6) * 2 - pd.Timedelta(days=7 if pd.Timestamp(f"{y}-03-01").dayofweek == 6 else 0),
                pd.Timestamp(f"{y}-11-01") + pd.offsets.Week(weekday=6) - pd.Timedelta(days=7 if pd.Timestamp(f"{y}-11-01").dayofweek == 6 else 0)]
    d["cal_dst"] = _in_windows(index, dst, 2)
    return d


RAMADAN = ["2012-07-20", "2013-07-09", "2014-06-28", "2015-06-18", "2016-06-06", "2017-05-27", "2018-05-16", "2019-05-06", "2020-04-24",
           "2021-04-13", "2022-04-02", "2023-03-23", "2024-03-11", "2025-03-01", "2026-02-18", "2027-02-08"]
SUPERBOWL = ["2012-02-05", "2013-02-03", "2014-02-02", "2015-02-01", "2016-02-07", "2017-02-05", "2018-02-04", "2019-02-03", "2020-02-02",
             "2021-02-07", "2022-02-13", "2023-02-12", "2024-02-11", "2025-02-09", "2026-02-08", "2027-02-14"]
WORLDCUP = [("2014-06-12", "2014-07-13"), ("2018-06-14", "2018-07-15"), ("2022-11-20", "2022-12-18"), ("2026-06-11", "2026-07-19")]
OLYMPICS = [("2012-07-27", "2012-08-12"), ("2016-08-05", "2016-08-21"), ("2021-07-23", "2021-08-08"), ("2024-07-26", "2024-08-11"),
            ("2014-02-07", "2014-02-23"), ("2018-02-09", "2018-02-25"), ("2022-02-04", "2022-02-20"), ("2026-02-06", "2026-02-22")]


def _in_windows(index, starts, days, before=0):
    st = pd.to_datetime(pd.Index(starts))
    return np.array([float(any(-before <= (t - s).days < days for s in st)) for t in index])


def _in_ranges(index, ranges):
    return np.array([float(any(pd.Timestamp(a) <= t <= pd.Timestamp(b) for a, b in ranges)) for t in index])


def mercury_retrograde(index):
    """Apparent retrograde motion of Mercury seen from Earth: computed from Keplerian orbits (J2000 elements, accurate to ~1 day)."""
    def helio(t, a, e, i, L, w, O):
        M = np.radians((L - w) % 360)
        E = M.copy()
        for _ in range(8):
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        x, y = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
        wp, Op, ip = np.radians(w - O), np.radians(O), np.radians(i)
        X = (np.cos(wp) * np.cos(Op) - np.sin(wp) * np.sin(Op) * np.cos(ip)) * x + (-np.sin(wp) * np.cos(Op) - np.cos(wp) * np.sin(Op) * np.cos(ip)) * y
        Y = (np.cos(wp) * np.sin(Op) + np.sin(wp) * np.cos(Op) * np.cos(ip)) * x + (-np.sin(wp) * np.sin(Op) + np.cos(wp) * np.cos(Op) * np.cos(ip)) * y
        return X, Y
    T = ((index - pd.Timestamp("2000-01-01 12:00")).days.to_numpy(float) + 0.5) / 36525
    mx, my = helio(T, 0.38709927, 0.20563593, 7.00497902, 252.25032350 + 149472.67411175 * T, 77.45779628 + 0.16047689 * T, 48.33076593 - 0.12534081 * T)
    ex, ey = helio(T, 1.00000261, 0.01671123, -0.00001531, 100.46457166 + 35999.37244981 * T, 102.93768193 + 0.32327364 * T, 0.0)
    lon = np.unwrap(np.arctan2(my - ey, mx - ex))
    return (np.gradient(lon) < 0).astype(float)


def transforms(raw, index):
    """Stationary transforms (returns, attention log-ratios, weather anomalies) aligned to the BTC axis. No peeking into the future."""
    F = {}
    try:                                                   # world event log (Wikipedia) as extra intensity series
        wl = pd.read_csv(C.DATA_DIR / "worldlog_daily.csv", index_col=0, parse_dates=True)
        raw = raw.join(wl[[c for c in WL if c in wl.columns]].rename(columns=lambda c: f"we_{c}"), how="outer")
    except Exception:
        pass
    try:                                                   # S&P 500 stocks, ETFs and market measures
        px = MK.load()
        if px is not None:
            sp = set(pd.read_csv(MK.SP500)["Symbol"]) if MK.SP500.exists() else set()
            etf = set(MK.ETF)
            ren = {c: (f"m_{c}" if c in etf else f"s_{c}" if c in sp else f"g_{c}") for c in px.columns}
            raw = raw.join(px.rename(columns=ren), how="outer")
        if MK.FEAT.exists():
            mf = pd.read_csv(MK.FEAT, index_col=0, parse_dates=True)
            raw = raw.join(mf.rename(columns=lambda c: f"mk_{c}"), how="outer")
    except Exception:
        pass
    try:                                                   # liquidity, CFTC, DeFi, Lightning, Hacker News, space weather
        if XT.OUT.exists():
            raw = raw.join(pd.read_csv(XT.OUT, index_col=0, parse_dates=True), how="outer")
    except Exception:
        pass
    raw = raw.reindex(index.union(raw.index)).sort_index()
    for c in raw.columns:
        s = raw[c]
        kind = DSER.get(c, XT.KIND.get(c, (None, None)))[1]
        if kind == "chg":
            ls = np.log(s.where(s > 0).ffill(limit=10).reindex(index))
            F[f"{c}|r7"], F[f"{c}|r30"] = ls.diff(7), ls.diff(30)
            continue
        if kind == "attn":
            s = s.where(s > 0).ffill(limit=5).reindex(index)
            ls = np.log(s)
            F[f"{c}|a7"] = ls.rolling(7).mean() - ls.rolling(90, min_periods=60).mean()
            F[f"{c}|a30"] = ls.rolling(30).mean() - ls.rolling(365, min_periods=200).mean()
            continue
        if kind == "lvl":
            s = s.ffill(limit=5).reindex(index)
            F[f"{c}|d7"] = s - s.shift(7)
            F[f"{c}|dev"] = s - s.rolling(365, min_periods=120).mean()
            continue
        if kind == "ratio":
            ls = np.log(s.where(s > 0).ffill(limit=5).reindex(index))
            F[f"{c}|r7"], F[f"{c}|r30"] = ls.diff(7), ls.diff(30)
            continue
        if c.startswith("g_"):
            s = s.where(s > 0).ffill(limit=5).reindex(index)
            F[f"{c}|r30"] = np.log(s).diff(30)
            continue
        if c.startswith(("y_", "m_", "mk_ratio_")):
            s = s.where(s > 0).ffill(limit=5).reindex(index)
            ls = np.log(s)
            if not c.startswith("mk_"):
                F[f"{c}|r1"] = ls.diff(1)
            F[f"{c}|r7"], F[f"{c}|r30"] = ls.diff(7), ls.diff(30)
        elif c.startswith("s_"):
            s = s.where(s > 0).ffill(limit=5).reindex(index)
            F[f"{c}|r30"] = np.log(s).diff(30)
        elif c.startswith("mk_"):
            s = s.ffill(limit=5).reindex(index)
            F[f"{c}|d7"] = s - s.shift(7)
            F[f"{c}|dev"] = s - s.rolling(365, min_periods=200).mean()
        elif c in ATTN:
            s = s.reindex(index)
            if c == "sunspots":
                s = s.ffill(limit=5)
            ls = np.log1p(s.clip(lower=0))
            F[f"{c}|a7"] = ls.rolling(7).mean() - ls.rolling(90, min_periods=60).mean()
            F[f"{c}|a30"] = ls.rolling(30).mean() - ls.rolling(365, min_periods=200).mean()
        elif c in WX:
            s = s.reindex(index)
            doy = s.groupby(s.index.dayofyear).transform(lambda x: x.expanding(min_periods=2).mean().shift(1))
            an = (s - doy)
            F[f"{c}|w7"] = an.rolling(7).mean()
            F[f"{c}|w30"] = an.rolling(30).mean()
    st = static_series(index)
    for c in st.columns:
        F[f"{c}|s"] = st[c]
    return pd.DataFrame(F, index=index)
