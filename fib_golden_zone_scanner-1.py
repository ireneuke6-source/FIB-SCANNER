"""
Fibonacci Golden Zone Scanner
-----------------------------
Detects the last swing leg on H1, builds fib retracement levels, and fires
a signal when M15 price retraces into the golden zone (0.618-0.65) with
rejection + RSI confirmation.

Railway deployment: runs as a persistent worker process, looping on an
interval (SCAN_INTERVAL_SECONDS, default 900s / 15 min). Each loop
iteration is wrapped so a single bad cycle (WS hiccup, rate limit, etc.)
logs and continues instead of crash-looping the whole service.

cooldown.json persists on local disk only. Railway's filesystem is
ephemeral across redeploys/restarts unless you attach a Volume mounted
at the working directory -- without one, cooldown state resets on every
deploy (functionally harmless, just means a leg could refire once after
a redeploy).

ENV VARS:
    TELEGRAM_BOT_TOKEN       (required)
    TELEGRAM_CHAT_ID         (required)
    SCAN_INTERVAL_SECONDS    (optional, default 900)
"""

import os
import json
import asyncio
import websockets
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

# ---------------- CONFIG ----------------
DERIV_APP_ID = 1089
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

SYMBOLS = [
    "frxEURUSD",
    "frxGBPUSD",
    "frxXAUUSD",
    "frxUSDJPY",
    "frxGBPJPY",
]

SWING_GRANULARITY = 3600      # H1 candles for swing/leg detection
SWING_COUNT = 200              # candles to pull for swing scan
FRACTAL_WIDTH = 3               # bars each side required to confirm a pivot

ENTRY_GRANULARITY = 900        # M15 candles for entry confirmation
ENTRY_COUNT = 100
RSI_PERIOD = 14

GOLDEN_ZONE = (0.618, 0.65)     # strict golden zone (highest score)
EXTENDED_ZONE = (0.5, 0.786)    # looser zone (partial score)

TP1_EXT = 0.272                 # 127.2% extension beyond the leg
TP2_EXT = 0.618                 # 161.8% extension beyond the leg
SL_BUFFER = 0.05                # extra SL buffer, as fraction of leg range

MIN_SCORE_TO_FIRE = 6           # out of 9, see scoring breakdown below
COOLDOWN_DIR = os.environ.get("COOLDOWN_DIR", ".")   # point this at your Railway Volume mount path
COOLDOWN_FILE = os.path.join(COOLDOWN_DIR, "fib_cooldown.json")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
SCAN_INTERVAL_SECONDS = int(os.environ.get("SCAN_INTERVAL_SECONDS", 900))


# ---------------- DATA FETCH ----------------
async def fetch_candles(symbol: str, granularity: int, count: int) -> pd.DataFrame:
    async with websockets.connect(DERIV_WS_URL, ping_interval=20) as ws:
        req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "candles",
            "granularity": granularity,
        }
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())

    if "error" in resp:
        raise RuntimeError(f"{symbol}: {resp['error']['message']}")

    candles = resp["candles"]
    df = pd.DataFrame(candles)
    df.rename(columns={"epoch": "time", "open": "open", "high": "high",
                        "low": "low", "close": "close"}, inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def closed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the still-forming last candle."""
    return df.iloc[:-1].reset_index(drop=True)


# ---------------- SWING / FIB LOGIC ----------------
def find_swings(df: pd.DataFrame, width: int = FRACTAL_WIDTH):
    """Fractal pivots: a bar is a swing high/low if it's the max/min of
    its (width*2+1)-bar window."""
    highs, lows = [], []
    n = len(df)
    for i in range(width, n - width):
        window = df.iloc[i - width: i + width + 1]
        if df["high"].iloc[i] == window["high"].max():
            highs.append((i, df["time"].iloc[i], df["high"].iloc[i]))
        if df["low"].iloc[i] == window["low"].min():
            lows.append((i, df["time"].iloc[i], df["low"].iloc[i]))
    return highs, lows


def get_last_leg(highs, lows):
    """Combine highs+lows chronologically, return the most recent
    alternating high->low or low->high pair as the active leg."""
    points = [("H", *h) for h in highs] + [("L", *l) for l in lows]
    points.sort(key=lambda p: p[1])  # sort by bar index
    if len(points) < 2:
        return None

    # walk backwards to find the last two points of opposite type
    last = points[-1]
    for p in reversed(points[:-1]):
        if p[0] != last[0]:
            leg_start, leg_end = p, last
            direction = "UP" if leg_start[0] == "L" else "DOWN"
            return {
                "direction": direction,
                "start_idx": leg_start[1],
                "start_time": leg_start[2],
                "start_price": leg_start[3],
                "end_idx": last[1],
                "end_time": last[2],
                "end_price": last[3],
            }
    return None


def fib_levels(leg: dict) -> dict:
    """Retracement levels as prices. For an UP leg (low->high), retracement
    measures back down from the high; for DOWN leg, back up from the low."""
    start, end = leg["start_price"], leg["end_price"]
    rng = end - start  # signed
    levels = {}
    for lvl in [0, 0.236, 0.382, 0.5, 0.618, 0.65, 0.786, 1.0]:
        levels[lvl] = end - rng * lvl
    return levels


def zone_bounds(levels: dict, zone: tuple) -> tuple:
    lo, hi = levels[zone[1]], levels[zone[0]]
    return (min(lo, hi), max(lo, hi))


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------- SIGNAL EVALUATION ----------------
def evaluate_symbol(symbol: str, h1_df: pd.DataFrame, m15_df: pd.DataFrame):
    h1 = closed_only(h1_df)
    m15 = closed_only(m15_df)

    highs, lows = find_swings(h1)
    leg = get_last_leg(highs, lows)
    if leg is None:
        return None

    levels = fib_levels(leg)
    golden_lo, golden_hi = zone_bounds(levels, GOLDEN_ZONE)
    ext_lo, ext_hi = zone_bounds(levels, EXTENDED_ZONE)

    last_close = m15["close"].iloc[-1]
    last_open = m15["open"].iloc[-1]
    last_high = m15["high"].iloc[-1]
    last_low = m15["low"].iloc[-1]
    m15["rsi"] = rsi(m15["close"])
    last_rsi = m15["rsi"].iloc[-1]

    score = 0
    zone_hit = None
    if golden_lo <= last_close <= golden_hi:
        score += 4
        zone_hit = "GOLDEN"
    elif ext_lo <= last_close <= ext_hi:
        score += 2
        zone_hit = "EXTENDED"
    else:
        return None  # price not in any retracement zone, skip

    # rejection wick confirmation, direction-aware
    body = abs(last_close - last_open)
    if leg["direction"] == "UP":  # expecting bullish rejection (buy)
        lower_wick = min(last_open, last_close) - last_low
        if lower_wick > body:
            score += 2
    else:  # DOWN leg, expecting bearish rejection (sell)
        upper_wick = last_high - max(last_open, last_close)
        if upper_wick > body:
            score += 2

    # RSI not exhausted against the trade direction
    if leg["direction"] == "UP" and last_rsi < 65:
        score += 2
    elif leg["direction"] == "DOWN" and last_rsi > 35:
        score += 2

    # body strength (avoid doji entries)
    candle_range = last_high - last_low
    if candle_range > 0 and body / candle_range > 0.35:
        score += 1

    if score < MIN_SCORE_TO_FIRE:
        return None

    leg_range = abs(leg["end_price"] - leg["start_price"])
    if leg["direction"] == "UP":
        side = "BUY"
        sl = leg["start_price"] - leg_range * SL_BUFFER
        tp1 = leg["end_price"] + leg_range * TP1_EXT
        tp2 = leg["end_price"] + leg_range * TP2_EXT
    else:
        side = "SELL"
        sl = leg["start_price"] + leg_range * SL_BUFFER
        tp1 = leg["end_price"] - leg_range * TP1_EXT
        tp2 = leg["end_price"] - leg_range * TP2_EXT

    rating = "PRIME" if score >= 8 else "STRONG" if score >= 7 else "GOOD"

    return {
        "symbol": symbol,
        "side": side,
        "zone": zone_hit,
        "rating": rating,
        "score": score,
        "entry": last_close,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": round(last_rsi, 1),
        "leg_id": f"{leg['start_time'].isoformat()}_{leg['end_time'].isoformat()}",
    }


# ---------------- COOLDOWN ----------------
def load_cooldown() -> dict:
    if os.path.exists(COOLDOWN_FILE):
        with open(COOLDOWN_FILE) as f:
            return json.load(f)
    return {}


def save_cooldown(data: dict):
    os.makedirs(COOLDOWN_DIR, exist_ok=True)
    with open(COOLDOWN_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------- TELEGRAM ----------------
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()


def format_signal(sig: dict) -> str:
    emoji = "🟢" if sig["side"] == "BUY" else "🔴"
    return (
        f"{emoji} <b>{sig['symbol']} {sig['side']}</b> — {sig['rating']} "
        f"({sig['score']}/9)\n"
        f"Zone: <b>{sig['zone']}</b> | RSI: {sig['rsi']}\n"
        f"Entry: {sig['entry']:.5f}\n"
        f"SL: {sig['sl']:.5f}\n"
        f"TP1: {sig['tp1']:.5f}\n"
        f"TP2: {sig['tp2']:.5f}\n"
        f"<i>Fib Golden Zone Scanner</i>"
    )


# ---------------- MAIN ----------------
async def scan_once():
    cooldown = load_cooldown()
    fired_any = False

    for symbol in SYMBOLS:
        try:
            h1_df = await fetch_candles(symbol, SWING_GRANULARITY, SWING_COUNT)
            m15_df = await fetch_candles(symbol, ENTRY_GRANULARITY, ENTRY_COUNT)
        except Exception as e:
            print(f"[{symbol}] fetch error: {e}")
            continue

        try:
            sig = evaluate_symbol(symbol, h1_df, m15_df)
        except Exception as e:
            print(f"[{symbol}] evaluation error: {e}")
            continue

        if sig is None:
            continue

        key = f"{symbol}_{sig['leg_id']}"
        if cooldown.get(key):
            print(f"[{symbol}] signal on cooldown for this leg, skipping")
            continue

        try:
            send_telegram(format_signal(sig))
        except Exception as e:
            print(f"[{symbol}] telegram send error: {e}")
            continue

        cooldown[key] = datetime.now(timezone.utc).isoformat()
        fired_any = True
        print(f"[{symbol}] fired {sig['side']} {sig['rating']} ({sig['score']}/9)")

    # prune cooldown entries older than 3 days to keep file small
    cutoff = datetime.now(timezone.utc).timestamp() - (3 * 86400)
    cooldown = {
        k: v for k, v in cooldown.items()
        if datetime.fromisoformat(v).timestamp() > cutoff
    }
    save_cooldown(cooldown)

    if not fired_any:
        print("No golden zone signals this cycle.")


async def run_forever():
    """Persistent worker loop for Railway. A single cycle's failure is
    caught and logged so the process keeps running rather than crashing
    and forcing Railway to restart it."""
    print(f"Fib Golden Zone Scanner starting. Interval: {SCAN_INTERVAL_SECONDS}s")

    try:
        startup_msg = (
            "✅ <b>Fib Golden Zone Scanner</b> is online.\n"
            f"Watching: {', '.join(SYMBOLS)}\n"
            f"Scan interval: {SCAN_INTERVAL_SECONDS}s\n"
            f"Golden zone: {GOLDEN_ZONE[0]}–{GOLDEN_ZONE[1]}"
        )
        send_telegram(startup_msg)
    except Exception as e:
        print(f"Startup telegram message failed: {e}")

    while True:
        cycle_start = datetime.now(timezone.utc)
        try:
            await scan_once()
        except Exception as e:
            print(f"Unhandled error in scan cycle: {e}")

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        sleep_for = max(5, SCAN_INTERVAL_SECONDS - elapsed)
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    asyncio.run(run_forever())
