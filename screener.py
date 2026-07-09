"""
Swing Trade Screener — 1-3 week horizon
Uses Kite Connect API for live NSE data
Target: ~5% return in 1-3 weeks

Weekly trend filter: only shows stocks where the weekly chart
is also bullish (price above 50 EMA, 50 EMA rising, weekly RSI > 50).
This avoids catching dead cat bounces in downtrending stocks.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from typing import Optional
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect

import event_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_N50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "TITAN", "WIPRO", "ULTRACEMCO", "BAJFINANCE", "NESTLEIND", "POWERGRID",
    "ONGC", "NTPC", "TECHM", "SUNPHARMA", "TMPV", "HCLTECH", "M&M",
    "ADANIENT", "BAJAJFINSV", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
    "HEROMOTOCO", "INDUSINDBK", "JSWSTEEL", "CIPLA", "COALINDIA", "BRITANNIA",
    "APOLLOHOSP", "TATACONSUM", "HINDALCO", "BPCL", "UPL", "BAJAJ-AUTO",
    "SHREECEM", "ADANIPORTS", "SBILIFE", "HDFCLIFE", "TATASTEEL",
]

_N_NEXT50 = [
    "AMBUJACEM", "AUROPHARMA", "BANKBARODA", "BERGEPAINT", "BOSCHLTD",
    "CHOLAFIN", "COLPAL", "CUMMINSIND", "DLF", "DMART", "GAIL", "GODREJCP",
    "GODREJPROP", "HAVELLS", "HINDZINC", "ICICIGI", "ICICIPRULI", "INDHOTEL",
    "IOC", "IRCTC", "LTM", "LTTS", "LUPIN", "UNITDSPR", "MOTHERSON",
    "MUTHOOTFIN", "NAUKRI", "NMDC", "OFSS", "PAGEIND", "PERSISTENT",
    "PIDILITIND", "POLYCAB", "RECLTD", "SAIL", "SIEMENS", "TATAELXSI",
    "TORNTPHARM", "TRENT", "TVSMOTOR", "VEDL", "VBL", "VOLTAS", "ETERNAL",
    "BAJAJHLDNG", "ZYDUSLIFE", "ATGL", "BHEL", "HINDPETRO", "PGHH",
]

_MIDCAP = [
    "ABCAPITAL", "ALKEM", "APOLLOTYRE", "BALKRISIND", "BANKINDIA", "BHARATFORG",
    "COFORGE", "CONCOR", "CROMPTON", "DEEPAKNTR", "DIXON", "EMAMILTD",
    "ESCORTS", "FEDERALBNK", "GLENMARK", "GRANULES", "IDFCFIRSTB", "INDUSTOWER",
    "JKCEMENT", "JUBLFOOD", "KALYANKJIL", "KPITTECH", "LALPATHLAB", "LICI",
    "MANAPPURAM", "MAXHEALTH", "MCX", "MPHASIS", "OBEROIRLTY", "PHOENIXLTD",
    "PRESTIGE", "SOBHA", "BRIGADE", "SUNDARMFIN", "SUNTECK", "TANLA",
    "TIINDIA", "TRIDENT", "UNIONBANK", "WHIRLPOOL", "NAUKRI", "CRISIL",
    "KPITTECH", "NBCC", "RVNL", "NYKAA", "CANBK", "PNB", "IPCALAB", "BIOCON",
]

UNIVERSES = {
    # ── Broad market ────────────────────────────────────────────────────────
    "nifty50":      _N50,
    "nifty_next50": _N_NEXT50,
    "nifty100":     list(dict.fromkeys(_N50 + _N_NEXT50)),   # deduped union
    "midcap":       _MIDCAP,

    # ── Sectoral ─────────────────────────────────────────────────────────────
    "nifty_bank": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK",
        "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK", "AUBANK", "PNB", "BANKBARODA",
    ],
    "nifty_it": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTM",
        "MPHASIS", "COFORGE", "PERSISTENT", "TATAELXSI",
    ],
    "nifty_auto": [
        "MARUTI", "TMPV", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
        "TVSMOTOR", "ASHOKLEY", "BOSCHLTD", "MOTHERSON", "BHARATFORG",
        "BALKRISIND", "TIINDIA", "MRF", "APOLLOTYRE",
    ],
    "nifty_pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
        "AUROPHARMA", "BIOCON", "TORNTPHARM", "IPCALAB", "ALKEM",
    ],
    "nifty_fmcg": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "GODREJCP", "MARICO", "COLPAL", "EMAMILTD", "TATACONSUM",
        "VBL", "RADICO", "UBL", "PGHH", "JYOTHYLAB",
    ],
    "nifty_financial": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIGI",
        "ICICIPRULI", "MUTHOOTFIN", "CHOLAFIN", "ABCAPITAL", "MANAPPURAM",
        "LICI", "IDFCFIRSTB", "FEDERALBNK", "RECLTD", "PFC",
    ],
    "nifty_metal": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "SAIL", "NMDC", "HINDZINC",
        "COALINDIA", "NATIONALUM", "VEDL", "JINDALSTEL", "APLAPOLLO",
        "RATNAMANI", "WELSPUNLIV", "HINDCOPPER",
    ],
    "nifty_realty": [
        "DLF", "OBEROIRLTY", "PRESTIGE", "GODREJPROP", "PHOENIXLTD",
        "SOBHA", "MAHLIFE", "BRIGADE", "SUNTECK", "NESCO",
    ],
    "nifty_psu_bank": [
        "SBIN", "BANKBARODA", "PNB", "CANBK", "UCOBANK",
        "MAHABANK", "UNIONBANK", "INDIANB", "IOB", "BANKINDIA", "CENTRALBK",
    ],
    "nifty_energy": [
        "RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "IOC",
        "GAIL", "TATAPOWER", "ADANIGREEN", "ADANIPOWER", "NHPC", "SJVN",
    ],
    "nifty_infra": [
        "LT", "ADANIPORTS", "ULTRACEMCO", "ADANIENT", "POWERGRID", "NTPC",
        "ONGC", "BPCL", "IOC", "GAIL", "NHPC", "ADANIGREEN", "TATAPOWER",
        "SJVN", "KEC", "PFC", "RECLTD", "IRCON", "CONCOR", "COALINDIA",
        "HINDALCO", "JSWSTEEL", "TATASTEEL", "SAIL", "ASHOKLEY",
        "IRB", "NBCC", "GMRAIRPORT", "RVNL", "NCC",
    ],
    "nifty_media": [
        "ZEEL", "SUNTV", "PVRINOX", "NETWORK18", "SAREGAMA", "NAZARA",
    ],
    "nifty_healthcare": [
        "APOLLOHOSP", "MAXHEALTH", "FORTIS", "DIVISLAB", "METROPOLIS",
        "THYROCARE", "ASTERDM", "NH", "LALPATHLAB", "SUNPHARMA",
    ],
    "nifty_consumer_durables": [
        "HAVELLS", "DIXON", "VOLTAS", "CROMPTON", "AMBER", "BLUESTARCO",
        "VGUARD", "WHIRLPOOL",
    ],
    "nifty_chemicals": [
        "PIDILITIND", "DEEPAKNTR", "SRF", "NAVINFLUOR", "AARTIIND",
        "TATACHEM", "COROMANDEL", "GNFC",
    ],
}

# All sectoral indices combined — excludes broad market to avoid duplicates
_SECTORAL_KEYS = [
    "nifty_bank", "nifty_it", "nifty_auto", "nifty_pharma", "nifty_fmcg",
    "nifty_financial", "nifty_metal", "nifty_realty", "nifty_psu_bank",
    "nifty_energy", "nifty_infra", "nifty_media", "nifty_healthcare",
    "nifty_consumer_durables", "nifty_chemicals",
]
UNIVERSES["all_sectors"] = list(dict.fromkeys(
    s for key in _SECTORAL_KEYS for s in UNIVERSES[key]
))

# All indices combined — deduplicated, preserving first-seen order
UNIVERSES["all"] = list(dict.fromkeys(
    s for key in UNIVERSES for s in UNIVERSES[key]
))


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9) -> dict:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
        "bullish": bool(histogram.iloc[-1] > 0 and histogram.iloc[-1] > histogram.iloc[-2]),
    }


def calc_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    current = close.iloc[-1]
    band_width = float(upper.iloc[-1] - lower.iloc[-1])
    position = (current - float(lower.iloc[-1])) / band_width if band_width > 0 else 0.5
    return {
        "upper": round(float(upper.iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower": round(float(lower.iloc[-1]), 2),
        "position": round(position, 3),
    }


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    atr = tr.ewm(span=period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return round(float(adx.iloc[-1]), 2)


def calc_volume_ratio(volume: pd.Series, lookback: int = 20) -> float:
    avg_vol = volume.iloc[-lookback-1:-1].mean()
    return round(float(volume.iloc[-1] / avg_vol), 2) if avg_vol > 0 else 1.0


def calc_momentum(close: pd.Series, days: int) -> float:
    """% change over the last `days` completed sessions.

    close[-1] vs close[-days-1]: a "5D" momentum spans exactly 5 sessions.
    (iloc[-days] would span days-1 sessions, and days=1 would compare the
    last close against itself — always 0.)
    """
    if len(close) <= days:
        return 0.0
    return round(float((close.iloc[-1] / close.iloc[-days - 1] - 1) * 100), 2)


def calc_52w(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """52-week high/low proximity using up to 252 trading days."""
    lookback = min(252, len(high))
    h52 = float(high.iloc[-lookback:].max())
    l52 = float(low.iloc[-lookback:].min())
    cur = float(close.iloc[-1])
    return {
        "high_52w":           round(h52, 2),
        "low_52w":            round(l52, 2),
        "dist_from_high_pct": round((cur - h52) / h52 * 100, 2),  # negative = below 52W high
        "dist_from_low_pct":  round((cur - l52) / l52 * 100, 2),  # positive = above 52W low
    }


def calc_golden_cross(close: pd.Series) -> dict:
    """Daily MA50 / MA200 golden-cross and death-cross detection."""
    if len(close) < 201:
        return {"ma50": 0.0, "ma200": 0.0, "above_ma200": False,
                "golden_cross": False, "death_cross": False, "cross_days_ago": None}
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    cur50, cur200 = float(ma50.iloc[-1]), float(ma200.iloc[-1])
    golden_cross = death_cross = False
    cross_days_ago = None
    for d in range(1, 21):
        if len(ma50) <= d + 1:
            break
        p50, p200 = float(ma50.iloc[-(d+1)]), float(ma200.iloc[-(d+1)])
        c50, c200 = float(ma50.iloc[-d]),     float(ma200.iloc[-d])
        if p50 <= p200 and c50 > c200:
            golden_cross = True;  cross_days_ago = d; break
        if p50 >= p200 and c50 < c200:
            death_cross  = True;  cross_days_ago = d; break
    return {
        "ma50":           round(cur50,  2),
        "ma200":          round(cur200, 2),
        "above_ma200":    float(close.iloc[-1]) > cur200,
        "golden_cross":   golden_cross,
        "death_cross":    death_cross,
        "cross_days_ago": cross_days_ago,
    }


def calc_weekly_vol_ratio(df: pd.DataFrame) -> float:
    """Last week's total volume vs 4-week average."""
    if df is None or len(df) < 25:
        return 1.0
    try:
        d = df.copy()
        d["_w"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.to_period("W")
        wv = d.groupby("_w")["volume"].sum()
        if len(wv) < 5:
            return 1.0
        avg4 = float(wv.iloc[-5:-1].mean())
        return round(float(wv.iloc[-1]) / avg4, 2) if avg4 > 0 else 1.0
    except Exception:
        return 1.0


def calc_support_resistance(close: pd.Series, high: pd.Series, low: pd.Series) -> dict:
    recent_high = float(high.iloc[-20:].max())
    recent_low = float(low.iloc[-20:].min())
    current = float(close.iloc[-1])
    return {
        "resistance": round(recent_high, 2),
        "support": round(recent_low, 2),
        "resistance_dist_pct": round((recent_high - current) / current * 100, 2),
        "support_dist_pct": round((current - recent_low) / current * 100, 2),
    }


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return round(float(atr.iloc[-1]), 2)


def calc_weekly_trend(weekly_df: Optional[pd.DataFrame]) -> dict:
    """
    Analyse weekly candles to determine the bigger-picture trend.

    Scoring (0-100, capped):

      PRIMARY TREND — EMA50 vs EMA200 relationship (30 pts max)
        30 pts — weekly EMA50 > EMA200 by >= 10% (strong golden cross)
        25 pts — weekly EMA50 > EMA200 by 3-10%  (developing golden cross)
        15 pts — weekly EMA50 > EMA200 by 0-3%   (just crossed, fragile)
         0 pts — weekly EMA50 < EMA200            (death cross = no credit)

      PRICE POSITION vs EMA50 (25 pts max)
        25 pts — price above EMA50                (momentum confirmed)
        15 pts — price within 5% below EMA50 AND golden cross
                 (Dow Theory secondary correction — highest-quality pullback)
         8 pts — price 5-10% below EMA50 AND golden cross  (deeper pullback)
         0 pts — price > 10% below EMA50, or death cross

      PRICE vs EMA200 (15 pts)
        15 pts — price above weekly EMA200        (long-term structural support)

      EMA50 SLOPE (15 pts max)
        15 pts — slope > +0.3%/week  (strongly rising trend)
        10 pts — slope > 0           (gently rising)
         5 pts — slope -1% to 0 AND golden cross
                 (slight decline = normal secondary correction, not breakdown)

      WEEKLY RSI (10 pts max)
        10 pts — RSI > 55
         5 pts — RSI 42-55

      WEEKLY MACD (10 pts)
        10 pts — bullish crossover

    Thresholds: bullish >= 70 | neutral 36-69 | bearish <= 35

    The key change vs the old version: PRIMARY TREND (EMA50 vs EMA200
    relationship) is now scored first and carries the most weight.  A stock
    in a Dow Theory secondary correction (price pulling back below EMA50
    while weekly golden cross is intact) is recognised as "pullback in
    uptrend" and can qualify as bullish — e.g. Nifty50 large-caps after
    a 25-30% correction from peak within a multi-year uptrend.
    """
    empty = {
        "trend": "neutral", "score": 50,
        "above_50ema": False, "above_200ema": False,
        "ema50_slope": 0.0, "weekly_rsi": 50.0,
        "weekly_macd_bull": False, "ema50": 0.0, "ema200": 0.0,
        "weekly_golden_cross": False, "pullback_in_uptrend": False,
        "momentum_confirming": False, "trend_deteriorating": False,
    }
    if weekly_df is None or len(weekly_df) < 20:
        return empty

    close = weekly_df["close"]
    current = float(close.iloc[-1])

    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    ema50_val  = float(ema50.iloc[-1])
    ema200_val = float(ema200.iloc[-1])

    above_50  = current > ema50_val
    above_200 = current > ema200_val

    weekly_golden_cross = ema50_val > ema200_val
    gc_margin_pct = (ema50_val - ema200_val) / ema200_val * 100 if ema200_val > 0 else 0.0

    dist_below_50_pct = (ema50_val - current) / ema50_val * 100 if ema50_val > 0 else 0.0

    ema50_slope = round(
        float(ema50.iloc[-1] - ema50.iloc[-5]) / float(ema50.iloc[-5]) * 100, 3
    ) if len(ema50) >= 5 else 0.0

    weekly_rsi  = calc_rsi(close, period=14)
    weekly_macd = calc_macd(close)

    # Momentum confirmation gate: at least ONE weekly momentum indicator must
    # support the "pullback is buyable" thesis. If RSI is weak, MACD bearish,
    # and EMA50 slope actively falling — the stock is in a correction, not a
    # textbook pullback, and should NOT score like one.
    momentum_confirming = (
        weekly_rsi > 50 or
        weekly_macd["bullish"] or
        ema50_slope > -0.3
    )
    trend_deteriorating = (
        weekly_golden_cross and not above_50 and not momentum_confirming
    )

    # Only label "pullback in uptrend" when momentum confirms the thesis.
    # A crash with bearish RSI + bearish MACD + falling EMA50 is trend
    # deterioration, not a Dow Theory secondary correction.
    pullback_in_uptrend = (
        weekly_golden_cross and
        not above_50 and
        above_200 and
        dist_below_50_pct <= 15 and
        momentum_confirming
    )

    score = 0

    # 1. Primary trend: EMA50 vs EMA200 (30 pts max)
    if weekly_golden_cross:
        if gc_margin_pct >= 10:    score += 30
        elif gc_margin_pct >= 3:   score += 25
        else:                      score += 15
    # death cross → 0 pts (no negative to allow recovery stocks to still score)

    # 2. Price position vs EMA50 (25 pts max)
    # Within golden cross territory, pulls below EMA50 CAN be secondary
    # corrections — but only if weekly momentum is confirming. Without
    # confirmation, halve the credit: structure exists but quality is poor.
    if above_50:
        score += 25
    elif weekly_golden_cross:
        if dist_below_50_pct <= 5:
            score += 15 if momentum_confirming else 8
        elif dist_below_50_pct <= 10:
            score += 10 if momentum_confirming else 5
        elif dist_below_50_pct <= 15:
            score += 5 if momentum_confirming else 2
        # else: > 15% below EMA50 → price breakdown, no credit

    # 3. Price vs EMA200 — long-term structural support (15 pts)
    if above_200:
        score += 15

    # 4. EMA50 slope (15 pts max)
    # Slope credit for negative slopes ONLY when momentum is confirming the
    # pullback thesis. A falling EMA50 + bearish RSI/MACD = trend change,
    # no slope credit.
    if ema50_slope > 0.3:
        score += 15
    elif ema50_slope > 0:
        score += 10
    elif momentum_confirming and weekly_golden_cross:
        if ema50_slope > -1.0:   score += 5   # gentle correction + momentum confirming
        elif ema50_slope > -3.0: score += 3   # deeper correction + momentum confirming

    # 5. Weekly RSI (10 pts max)
    if weekly_rsi > 55:           score += 10
    elif weekly_rsi > 42:         score += 5

    # 6. Weekly MACD (10 pts)
    if weekly_macd["bullish"]:    score += 10

    # 7. Pullback-in-uptrend zone bonus (5 pts) — now momentum-gated
    if pullback_in_uptrend:
        score += 5

    score = min(100, score)
    # Raised threshold to 70: a marginal "pullback" without momentum
    # confirmation used to land at 65 and get labelled bullish. Requiring 70
    # means the stock must have at least one momentum tailwind OR price above
    # EMA50 to be called bullish.
    trend = "bullish" if score >= 70 else "bearish" if score <= 35 else "neutral"

    return {
        "trend": trend,
        "score": score,
        "above_50ema": above_50,
        "above_200ema": above_200,
        "ema50_slope": ema50_slope,
        "weekly_rsi": round(weekly_rsi, 1),
        "weekly_macd_bull": weekly_macd["bullish"],
        "ema50": round(ema50_val, 2),
        "ema200": round(ema200_val, 2),
        "weekly_golden_cross": weekly_golden_cross,
        "pullback_in_uptrend": pullback_in_uptrend,
        "momentum_confirming": momentum_confirming,
        "trend_deteriorating": trend_deteriorating,
    }



# ---------------------------------------------------------------------------
# Price action detectors
# ---------------------------------------------------------------------------

def detect_price_action(df) -> dict:
    """
    Detect candlestick patterns on the last 20 daily candles.

    Bullish (contribute to pa_score):
      1.  Hammer                — long lower wick reversal
      2.  Bullish engulfing     — today fully engulfs prior red candle
      3.  Resistance breakout   — close above 20-day high with volume surge
      4.  Support bounce        — touched 20-day low, closed upper 40% green
      5.  Inside bar            — range contained in prior candle, green close
      6.  Higher high + HL      — last 3-candle window beats prior window
      7.  Inverted hammer       — long upper wick after downtrend
      8.  Morning star          — 3-candle bullish reversal
      9.  Piercing line         — green candle closes above 50% of prior red body
      10. Three white soldiers  — 3 consecutive rising green candles
      11. Marubozu              — near-full-body green candle, no wicks
      12. Rising three methods  — continuation after small pullback
      13. Bullish Harami        — small green body contained within prior large red body
      14. NR4 Squeeze           — narrowest range of last 4 bars (volatility squeeze)

    Bearish warnings (shown but NOT added to pa_score):
      15. Shooting star         — long upper wick after uptrend
      16. Bearish engulfing     — red candle fully engulfs prior green
      17. Evening star          — 3-candle bearish reversal
      18. Dark cloud cover      — red candle closes below 50% of prior green body

    pa_score: 0-30 bonus on top of the base 100-pt score.
    """
    if df is None or len(df) < 5:
        return _empty_pa()

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values

    patterns = {}
    active   = []

    # ── Bullish patterns ──────────────────────────────────────────────────────

    # 1. Hammer
    body         = abs(c[-1] - o[-1])
    candle_range = h[-1] - l[-1]
    if candle_range > 0 and body > 0:
        lower_shadow = min(c[-1], o[-1]) - l[-1]
        upper_shadow = h[-1] - max(c[-1], o[-1])
        hammer = (lower_shadow >= 2 * body and
                  upper_shadow <= 0.3 * body and
                  body / candle_range <= 0.35 and
                  c[-1] > o[-1])
    else:
        hammer = False
    patterns["hammer"] = hammer
    if hammer:
        active.append("Hammer")

    # 2. Bullish engulfing
    if len(c) >= 2:
        prev_bear  = c[-2] < o[-2]
        curr_bull  = c[-1] > o[-1]
        engulfs    = o[-1] <= c[-2] and c[-1] >= o[-2]
        body_ratio = abs(c[-1] - o[-1]) / max(abs(c[-2] - o[-2]), 1)
        engulfing  = prev_bear and curr_bull and engulfs and body_ratio >= 1.1
    else:
        engulfing = False
    patterns["bullish_engulfing"] = engulfing
    if engulfing:
        active.append("Bullish engulfing")

    # 3. Resistance breakout
    if len(h) >= 20:
        prior_high = float(h[-20:-1].max())
        vol_avg    = float(v[-20:-1].mean())
        breakout   = c[-1] > prior_high and v[-1] >= 1.5 * vol_avg
    else:
        breakout = False
    patterns["breakout"] = breakout
    if breakout:
        active.append("Resistance breakout")

    # 4. Support bounce
    if len(l) >= 20:
        support       = float(l[-20:-1].min())
        support_zone  = support * 1.015
        candle_range2 = h[-1] - l[-1]
        bounce = (l[-1] <= support_zone and
                  candle_range2 > 0 and
                  c[-1] > (l[-1] + candle_range2 * 0.6) and
                  c[-1] > o[-1])
    else:
        bounce = False
    patterns["support_bounce"] = bounce
    if bounce:
        active.append("Support bounce")

    # 5. Inside bar (bullish)
    if len(h) >= 3:
        inside = h[-1] <= h[-2] and l[-1] >= l[-2] and c[-1] > o[-1]
    else:
        inside = False
    patterns["inside_bar"] = inside
    if inside:
        active.append("Inside bar")

    # 6. Higher high + higher low
    if len(h) >= 6:
        recent_high = float(h[-3:].max())
        prior_high2 = float(h[-6:-3].max())
        recent_low  = float(l[-3:].min())
        prior_low2  = float(l[-6:-3].min())
        hh_hl = recent_high > prior_high2 and recent_low > prior_low2
    else:
        hh_hl = False
    patterns["hh_hl"] = hh_hl
    if hh_hl:
        active.append("Higher high + HL")

    # 7. Inverted hammer
    if len(c) >= 4:
        ih_body  = abs(c[-1] - o[-1])
        ih_range = h[-1] - l[-1]
        ih_upper = h[-1] - max(c[-1], o[-1])
        ih_lower = min(c[-1], o[-1]) - l[-1]
        ih_down  = c[-4] > c[-3] > c[-2]   # prior 3-candle downtrend
        inv_hammer = (ih_range > 0 and ih_body > 0 and
                      ih_upper >= 2 * ih_body and
                      ih_lower <= 0.3 * ih_body and
                      ih_down)
    else:
        inv_hammer = False
    patterns["inverted_hammer"] = inv_hammer
    if inv_hammer:
        active.append("Inverted hammer")

    # 8. Morning star
    if len(c) >= 3:
        ms_c3_body = abs(c[-3] - o[-3])
        ms_c2_body = abs(c[-2] - o[-2])
        ms_c3_mid  = min(o[-3], c[-3]) + ms_c3_body / 2
        morning_star = (ms_c3_body > 0 and
                        c[-3] < o[-3] and                   # candle[-3] red
                        ms_c2_body <= 0.30 * ms_c3_body and # candle[-2] doji-like
                        c[-1] > o[-1] and                   # candle[-1] green
                        c[-1] > ms_c3_mid)                  # closes above midpoint
    else:
        morning_star = False
    patterns["morning_star"] = morning_star
    if morning_star:
        active.append("Morning star")

    # 9. Piercing line
    if len(c) >= 2:
        pl_body = abs(c[-2] - o[-2])
        pl_mid  = min(o[-2], c[-2]) + pl_body / 2
        piercing = (c[-2] < o[-2] and           # candle[-2] red
                    o[-1] < l[-2] and            # opens below prior low
                    c[-1] > o[-1] and            # candle[-1] green
                    c[-1] > pl_mid and           # closes above 50% of prior body
                    c[-1] < o[-2])               # but below prior open
    else:
        piercing = False
    patterns["piercing_line"] = piercing
    if piercing:
        active.append("Piercing line")

    # 10. Three white soldiers
    if len(c) >= 3:
        tws = (c[-3] > o[-3] and c[-2] > o[-2] and c[-1] > o[-1] and
               o[-2] >= o[-3] and o[-2] <= c[-3] and
               o[-1] >= o[-2] and o[-1] <= c[-2] and
               c[-1] > c[-2] > c[-3] and
               (h[-3] - c[-3]) <= 0.1 * max(c[-3] - o[-3], 1) and
               (h[-2] - c[-2]) <= 0.1 * max(c[-2] - o[-2], 1) and
               (h[-1] - c[-1]) <= 0.1 * max(c[-1] - o[-1], 1))
    else:
        tws = False
    patterns["three_white_soldiers"] = tws
    if tws:
        active.append("Three white soldiers")

    # 11. Marubozu (strong full-body green candle)
    mz_body  = abs(c[-1] - o[-1])
    mz_range = h[-1] - l[-1]
    marubozu = (mz_range > 0 and
                c[-1] > o[-1] and
                mz_body / mz_range >= 0.90 and
                mz_body / c[-1] >= 0.015)
    patterns["marubozu"] = marubozu
    if marubozu:
        active.append("Marubozu")

    # 12. Rising three methods
    if len(c) >= 5:
        r3_body = abs(c[-5] - o[-5])
        rtm = (c[-5] > o[-5] and r3_body > 0 and
               all(h[j] <= h[-5] and l[j] >= l[-5] and
                   abs(c[j] - o[j]) <= 0.5 * r3_body
                   for j in [-4, -3, -2]) and
               c[-1] > o[-1] and
               c[-1] > c[-5])
    else:
        rtm = False
    patterns["rising_three_methods"] = rtm
    if rtm:
        active.append("Rising three methods")

    # 13. Bullish Harami  (small green candle contained within a prior large red body — indecision
    #     shifting to buyers; per NSE TA textbook: further confirmation required but high probability
    #     when combined with volume pickup on the harami day)
    if len(c) >= 2:
        bh_prev = abs(c[-2] - o[-2])
        bh_curr = abs(c[-1] - o[-1])
        bh_harami = (bh_prev > 0 and
                     c[-2] < o[-2] and                  # prior candle red
                     c[-1] > o[-1] and                  # current candle green
                     bh_curr <= 0.5 * bh_prev and       # current body ≤ 50% of prior
                     o[-1] >= c[-2] and c[-1] <= o[-2]) # current body inside prior body
    else:
        bh_harami = False
    patterns["bullish_harami"] = bh_harami
    if bh_harami:
        active.append("Bullish Harami")

    # 14. Narrow Range 4 (NR4)  — Fidelity chart patterns textbook: "New trends often begin from
    #     periods of low volatility."  The last candle has the narrowest range of the past 4 bars;
    #     this volatility squeeze often precedes a directional expansion move.
    if len(h) >= 4:
        bar_ranges = [h[j] - l[j] for j in range(-4, 0)]
        nr4 = bar_ranges[-1] > 0 and bar_ranges[-1] < min(bar_ranges[:3])
    else:
        nr4 = False
    patterns["narrow_range"] = nr4
    if nr4:
        active.append("NR4 Squeeze")

    # ── Bearish warning patterns (not added to pa_score) ─────────────────────

    # 15. Shooting star
    if len(c) >= 4:
        ss_body  = abs(c[-1] - o[-1])
        ss_range = h[-1] - l[-1]
        ss_floor = max(ss_body, ss_range * 0.05)
        ss_upper = h[-1] - max(c[-1], o[-1])
        ss_lower = min(c[-1], o[-1]) - l[-1]
        ss_up    = c[-4] < c[-3] < c[-2]   # prior uptrend
        shooting_star = (ss_range > 0 and
                         ss_upper >= 2 * ss_floor and
                         ss_lower <= 0.3 * ss_floor and
                         (c[-1] <= o[-1] or ss_body <= ss_range * 0.25) and
                         ss_up)
    else:
        shooting_star = False
    patterns["shooting_star"] = shooting_star

    # 16. Bearish engulfing
    if len(c) >= 2:
        be = (c[-2] > o[-2] and c[-1] < o[-1] and
              o[-1] >= c[-2] and c[-1] <= o[-2])
    else:
        be = False
    patterns["bearish_engulfing"] = be

    # 17. Evening star
    if len(c) >= 3:
        es_c3_body = abs(c[-3] - o[-3])
        es_c2_body = abs(c[-2] - o[-2])
        es_c3_mid  = min(o[-3], c[-3]) + es_c3_body / 2
        evening_star = (es_c3_body > 0 and
                        c[-3] > o[-3] and                    # candle[-3] green
                        es_c2_body <= 0.30 * es_c3_body and  # candle[-2] small
                        c[-1] < o[-1] and                    # candle[-1] red
                        c[-1] < es_c3_mid)                   # closes below midpoint
    else:
        evening_star = False
    patterns["evening_star"] = evening_star

    # 18. Dark cloud cover
    if len(c) >= 2:
        dc_body = abs(c[-2] - o[-2])
        dc_mid  = min(o[-2], c[-2]) + dc_body / 2
        dark_cloud = (c[-2] > o[-2] and          # candle[-2] green
                      o[-1] > h[-2] and           # opens above prior high
                      c[-1] < o[-1] and           # candle[-1] red
                      c[-1] < dc_mid and          # closes below 50% of prior body
                      c[-1] > o[-2])              # but above prior open
    else:
        dark_cloud = False
    patterns["dark_cloud_cover"] = dark_cloud

    # ── PA score — bullish patterns only (max 30) ─────────────────────────────
    pa_score = 0
    if patterns["morning_star"]:          pa_score += 14
    if patterns["bullish_engulfing"]:     pa_score += 12
    if patterns["breakout"]:              pa_score += 12
    if patterns["three_white_soldiers"]:  pa_score += 12
    if patterns["piercing_line"]:         pa_score += 10
    if patterns["rising_three_methods"]:  pa_score += 10
    if patterns["hammer"]:                pa_score += 8
    if patterns["support_bounce"]:        pa_score += 8
    if patterns["marubozu"]:              pa_score += 8
    if patterns["hh_hl"]:                 pa_score += 6
    if patterns["inverted_hammer"]:       pa_score += 6
    if patterns["bullish_harami"]:        pa_score += 6
    if patterns["inside_bar"]:            pa_score += 4
    if patterns["narrow_range"]:          pa_score += 4
    pa_score = min(30, pa_score)

    bearish_patterns = []
    if patterns["shooting_star"]:    bearish_patterns.append("Shooting star")
    if patterns["bearish_engulfing"]:bearish_patterns.append("Bearish engulfing")
    if patterns["evening_star"]:     bearish_patterns.append("Evening star")
    if patterns["dark_cloud_cover"]: bearish_patterns.append("Dark cloud cover")

    return {
        "patterns":           patterns,
        "active":             active,
        "pa_score":           pa_score,
        "has_pattern":        len(active) > 0,
        "pattern_summary":    ", ".join(active) if active else "None",
        "bearish_patterns":   bearish_patterns,
        "is_bearish_warning": len(bearish_patterns) > 0,
    }


def _empty_pa() -> dict:
    return {
        "patterns":           {},
        "active":             [],
        "pa_score":           0,
        "has_pattern":        False,
        "pattern_summary":    "None",
        "bearish_patterns":   [],
        "is_bearish_warning": False,
    }


# ---------------------------------------------------------------------------
# Scoring engine (daily signals)
# ---------------------------------------------------------------------------

def score_stock(indicators: dict, filters: dict) -> dict:
    score = 0
    breakdown = {}

    rsi = indicators["rsi"]
    if 50 <= rsi <= 65:       pts = 22
    elif 45 <= rsi <= 70:     pts = 15
    elif 40 <= rsi <= 75:     pts = 8
    else:                     pts = 0
    score += pts
    breakdown["rsi"] = {"value": rsi, "points": pts, "max": 22}

    vol_ratio = indicators["volume_ratio"]
    if vol_ratio >= 2.0:      pts = 18
    elif vol_ratio >= 1.5:    pts = 12
    elif vol_ratio >= 1.2:    pts = 6
    else:                     pts = 0
    score += pts
    breakdown["volume"] = {"value": vol_ratio, "points": pts, "max": 18}

    if indicators["macd"]["bullish"]:           pts = 15
    elif indicators["macd"]["histogram"] > 0:   pts = 8
    else:                                       pts = 0
    score += pts
    breakdown["macd"] = {"value": indicators["macd"]["histogram"], "points": pts, "max": 15}

    ma_dist = indicators["ma20_dist_pct"]
    if 0 < ma_dist <= 3:      pts = 15   # ideal — just above MA20, clean entry
    elif ma_dist > 3:         pts = 8    # extended above MA20
    elif -1 <= ma_dist <= 0:  pts = 5    # at MA20 — testing as support
    elif ma_dist < -1 and vol_ratio >= 2.0 and 35 <= rsi <= 60:
        # Below MA20 but: massive volume + RSI recovering from oversold
        # = volume-confirmed reversal setup (NSE TA: volume is the primary
        #   confirmation for any bottoming pattern).  Give partial credit.
        pts = 5
    else:
        pts = 0
    score += pts
    breakdown["ma20"] = {"value": ma_dist, "points": pts, "max": 15}

    adx = indicators["adx"]
    if adx >= 30:     pts = 15
    elif adx >= 25:   pts = 10
    elif adx >= 20:   pts = 5
    else:             pts = 0
    score += pts
    breakdown["adx"] = {"value": adx, "points": pts, "max": 15}

    mom5 = indicators["momentum_5d"]
    if 1 <= mom5 <= 5:        pts = 10
    elif 0 < mom5 <= 8:       pts = 6
    else:                     pts = 0
    score += pts
    breakdown["momentum_5d"] = {"value": mom5, "points": pts, "max": 10}

    # 20-day momentum — captures sustained multi-week recovery that 5D misses.
    # This is the key metric for bottom-reversal plays: a stock can have poor
    # 5D momentum (already spiked today) but excellent 20D momentum (been
    # recovering steadily for 3 weeks).  Fidelity PDF: trend confirmation
    # requires SUSTAINED price improvement, not single-day moves.
    mom20 = indicators["momentum_20d"]
    if 5 <= mom20 <= 25:      pts = 10   # solid sustained recovery
    elif 0 < mom20 < 5:       pts = 5    # early positive trend
    elif mom20 > 25:          pts = 3    # extended but still bullish
    else:                     pts = 0    # negative 20D = still in downtrend
    score += pts
    breakdown["momentum_20d"] = {"value": mom20, "points": pts, "max": 10}

    # Relative strength vs Nifty 50 (10 pts max). For a 1-2 week horizon,
    # outperformance vs the index is the strongest continuation signal there
    # is — absolute momentum in a stock that is merely tracking the market
    # tells you about the market, not the stock. rs = stock 20D return minus
    # Nifty 20D return, in percentage points. None = index data unavailable
    # this cycle → neutral credit so scores stay comparable.
    rs20 = indicators.get("rs_nifty_20d")
    rs5  = indicators.get("rs_nifty_5d")
    if rs20 is None:
        pts = 5
    elif rs20 > 3 and (rs5 is None or rs5 > 0):
        pts = 10    # clear leader and still leading this week
    elif rs20 > 3:
        pts = 7     # leader, but cooling off short-term
    elif rs20 > 0:
        pts = 6
    elif rs20 > -3:
        pts = 2
    else:
        pts = 0     # clear laggard — momentum is the market's, not the stock's
    score += pts
    breakdown["rs_nifty"] = {"value": rs20, "points": pts, "max": 10}

    bb_pos = indicators["bollinger"]["position"]
    if 0.35 <= bb_pos <= 0.65:    pts = 5
    elif 0.25 <= bb_pos <= 0.75:  pts = 3
    else:                         pts = 0
    score += pts
    breakdown["bollinger"] = {"value": bb_pos, "points": pts, "max": 5}

    score = min(100, score)

    # ── Death-cross regime penalty (applied after cap) ──────────────────────
    # Three tiers, from worst to best:
    #   -20  Fresh death-cross (<= 30 days): trend just turned — avoid.
    #   -18  Price ABOVE MA200 but MA50 still below: "false breakout" risk.
    #        Momentum metrics all light up on a dead-cat bounce; the long-term
    #        trend hasn't reversed — MA50 must recross MA200 to confirm.
    #   -12  Price BELOW MA200 and MA50 below: genuine bottom recovery.
    #        Stock is still under the long-term average so there's real upside
    #        if the reversal proves out — lighter penalty rewards early entry.
    gc = indicators.get("golden_cross", {})
    if gc and gc.get("ma200", 0) > 0:
        ma50_val    = gc.get("ma50",  0)
        ma200_val   = gc.get("ma200", 0)
        above_ma200 = gc.get("above_ma200", False)
        current_px  = indicators.get("current_price", 0)
        if gc.get("death_cross"):
            dc_penalty = 20
            dc_label   = f"fresh death-cross (MA50 {ma50_val:.0f} < MA200 {ma200_val:.0f})"
        elif ma50_val > 0 and ma50_val < ma200_val:
            if above_ma200:
                # Price poked above MA200 but trend not confirmed — false breakout risk
                dc_penalty = 18
                dc_label   = (f"false-breakout risk: price above MA200 ({ma200_val:.0f}) "
                              f"but MA50 ({ma50_val:.0f}) still below")
            else:
                # Price still under MA200 — genuine recovery, lighter penalty
                dc_penalty = 12
                dc_label   = (f"recovery below MA200 ({ma200_val:.0f}), "
                              f"MA50 ({ma50_val:.0f}) still below")
        else:
            dc_penalty = 0
            dc_label   = ""
        if dc_penalty:
            score = max(0, score - dc_penalty)
            breakdown["death_cross_penalty"] = {
                "value": dc_label, "points": -dc_penalty, "max": 0,
            }

    atr = indicators["atr"]
    price = indicators["current_price"]
    target_pct = round((atr * 3) / price * 100, 2)
    sl_pct     = round((atr * 1.5) / price * 100, 2)
    rr         = round(target_pct / sl_pct, 2) if sl_pct > 0 else 0
    # target_pct stays the honest 3x ATR number — no flooring to
    # min_target_pct. Flooring made the min_target_pct rejection in
    # analyse_stock unreachable, so low-volatility stocks that cannot
    # plausibly move 5% in 1-2 weeks passed with an inflated target.

    return {
        "score": score,
        "breakdown": breakdown,
        "target_pct": target_pct,
        "sl_pct": sl_pct,
        "rr": rr,
        "signal": "strong_buy" if score >= 75 else "moderate" if score >= 60 else "watch",
    }


# ---------------------------------------------------------------------------
# Main Screener class
# ---------------------------------------------------------------------------

IST = timezone(timedelta(hours=5, minutes=30))
NSE_CLOSE = dtime(15, 30)


def _drop_forming_candle(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Drop the still-forming last candle so indicators see completed bars only.

    Kite's historical API includes today's partial candle when to_date is
    today (and the current week's partial candle for weekly data). Computing
    RSI/MACD/volume-ratio/momentum on a half-formed bar makes every value
    drift intraday — scan results end up depending on the time of day the
    scan runs, and partial-day volume vs full-day averages systematically
    fails the volume-surge filter in the morning. Live price still reaches
    the UI via the LTP override in analyse_stock.
    """
    if df is None or len(df) == 0:
        return df
    now = datetime.now(IST)
    last = df["date"].iloc[-1]
    if getattr(last, "tzinfo", None) is not None:
        last = last.tz_convert(IST)
    else:
        now = now.replace(tzinfo=None)

    if interval == "day":
        forming = last.date() == now.date() and now.time() < NSE_CLOSE
    elif interval == "week":
        # Weekly candles are dated at the week's first session; the candle
        # completes at Friday's close of that week.
        week_close = (last + pd.Timedelta(days=4 - last.weekday())).replace(
            hour=NSE_CLOSE.hour, minute=NSE_CLOSE.minute)
        forming = now < week_close
    else:
        forming = False

    return df.iloc[:-1].reset_index(drop=True) if forming else df


class SwingScreener:
    INDEX_SYMBOL = "NIFTY 50"

    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._instrument_map: dict = {}
        self._index_mom_cache: Optional[dict] = None

    def _index_momentum(self) -> dict:
        """Nifty 50 5D/20D momentum for relative-strength scoring.

        Cached 10 min on success, 5 min on failure. Returns
        {"mom5": float | None, "mom20": float | None} — Nones mean the
        index fetch failed and RS scoring falls back to neutral credit.
        """
        now = time.time()
        cached = self._index_mom_cache
        if cached is not None:
            ttl = 600 if cached["mom20"] is not None else 300
            if now - cached["ts"] < ttl:
                return cached
        mom5 = mom20 = None
        try:
            if not self._instrument_map:
                self.load_instruments()
            df = self._fetch(self.INDEX_SYMBOL, "day", 40)
            if df is not None and len(df) >= 21:
                mom5  = calc_momentum(df["close"], 5)
                mom20 = calc_momentum(df["close"], 20)
        except Exception as e:
            log.warning(f"Nifty momentum fetch failed (RS neutral this cycle): {e}")
        self._index_mom_cache = {"ts": now, "mom5": mom5, "mom20": mom20}
        return self._index_mom_cache

    def load_instruments(self):
        log.info("Loading NSE instruments...")
        instruments = self.kite.instruments("NSE")
        # Build map: prefer EQ type but fall back to any entry for the symbol
        tmp: dict = {}
        for inst in instruments:
            sym = inst["tradingsymbol"]
            if sym not in tmp or inst.get("instrument_type") == "EQ":
                tmp[sym] = inst["instrument_token"]
        self._instrument_map = tmp
        log.info(f"Loaded {len(self._instrument_map)} NSE instruments")

    def _get_token(self, symbol: str) -> Optional[int]:
        return self._instrument_map.get(symbol)

    def _fetch(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        token = self._get_token(symbol)
        if not token:
            log.warning(f"Token not found: {symbol}")
            return None
        to_date = datetime.now()
        if interval == "day":
            # `days` means TRADING days. NSE has ~250 sessions/year, so the
            # calendar window must be ~1.5x wider (plus holiday buffer) or a
            # request for 265 sessions returns only ~190 — which silently
            # disabled every MA200-based check (calc_golden_cross needs 201).
            from_date = to_date - timedelta(days=int(days * 1.55) + 20)
        else:
            from_date = to_date - timedelta(days=days + 14)
        try:
            candles = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d"),
                interval=interval,
            )
            if not candles:
                return None
            df = pd.DataFrame(candles)
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return _drop_forming_candle(df, interval)
        except Exception as e:
            log.warning(f"Fetch error {symbol} ({interval}): {e}")
            return None

    def fetch_ohlcv(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        df = self._fetch(symbol, "day", days)
        return df.tail(days) if df is not None else None

    def fetch_weekly_ohlcv(self, symbol: str, weeks: int = 250) -> Optional[pd.DataFrame]:
        """Fetch weekly candles (~5 years for reliable 200 EMA)."""
        df = self._fetch(symbol, "week", weeks * 7)
        return df.tail(weeks) if df is not None else None

    def analyse_stock(self, symbol: str, sector: str = "—", filters: dict = None) -> Optional[dict]:
        if filters is None:
            filters = {}

        # ── Surveillance screen (before any data fetch — it's a set lookup) ──
        # ASM/GSM stocks carry 100% margin and tight circuits; a swing exit
        # can get trapped for days. Label is always computed so non-excluding
        # callers (custom watchlists) can still flag it in the UI.
        surveillance = event_risk.surveillance_label(symbol)
        if surveillance and filters.get("exclude_surveillance", True):
            log.info(f"  {symbol} rejected — NSE surveillance: {surveillance}")
            return None

        # ── Daily analysis ──────────────────────────────────────────────────
        df = self.fetch_ohlcv(symbol, days=265)
        if df is None or len(df) < 30:
            return None

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        current_price = float(close.iloc[-1])   # fallback: last historical close

        # Override with live LTP so the displayed price matches the Kite app
        try:
            ltp_data = self.kite.ltp([f"NSE:{symbol}"])
            live_px  = ltp_data.get(f"NSE:{symbol}", {}).get("last_price", 0)
            if live_px and live_px > 0:
                current_price = float(live_px)
        except Exception:
            pass   # market closed or API error — historical close is fine

        ma20 = float(close.rolling(20).mean().iloc[-1])

        indicators = {
            "current_price":    current_price,
            "rsi":              calc_rsi(close),
            "macd":             calc_macd(close),
            "bollinger":        calc_bollinger(close),
            "adx":              calc_adx(high, low, close),
            "volume_ratio":     calc_volume_ratio(volume),
            "momentum_5d":      calc_momentum(close, 5),
            "momentum_20d":     calc_momentum(close, 20),
            "ma20":             round(ma20, 2),
            "ma20_dist_pct":    round((current_price - ma20) / ma20 * 100, 2),
            "atr":              calc_atr(high, low, close),
            "sr":               calc_support_resistance(close, high, low),
            "chg_1d":           calc_momentum(close, 1),
            "chg_5d":           calc_momentum(close, 5),
            "52w":              calc_52w(high, low, close),
            "golden_cross":     calc_golden_cross(close),
            "weekly_vol_ratio": calc_weekly_vol_ratio(df),
        }

        # Relative strength vs Nifty 50 (percentage points of out/underperformance)
        idx = self._index_momentum()
        indicators["nifty_mom_5d"]  = idx["mom5"]
        indicators["nifty_mom_20d"] = idx["mom20"]
        indicators["rs_nifty_5d"] = (
            round(indicators["momentum_5d"] - idx["mom5"], 2)
            if idx["mom5"] is not None else None)
        indicators["rs_nifty_20d"] = (
            round(indicators["momentum_20d"] - idx["mom20"], 2)
            if idx["mom20"] is not None else None)

        # Pre-filters
        rsi = indicators["rsi"]
        if not (filters.get("rsi_min", 40) <= rsi <= filters.get("rsi_max", 75)):
            return None
        if indicators["volume_ratio"] < filters.get("min_vol_surge", 1.2):
            return None

        # ── Daily scoring ────────────────────────────────────────────────────
        scoring = score_stock(indicators, filters)
        if scoring["score"] < filters.get("min_score", 60):
            return None
        # Volatility gate: 3x ATR must span the mandated move. A stock whose
        # ATR-based target is below min_target_pct cannot plausibly deliver
        # the 5-10% swing in 1-2 weeks regardless of its score.
        if scoring["target_pct"] < filters.get("min_target_pct", 5.0):
            return None

        # ── Earnings gate (after score gates — one yfinance call per real
        # candidate, cached 24h). A 1-2 week hold that straddles a results
        # day is gap-risk, not a technical trade. 0 = screen off.
        earnings_window = int(filters.get("exclude_earnings_days", 10))
        if earnings_window > 0:
            earnings = event_risk.earnings_info(symbol)
            days_to = earnings["days_to"]
            if days_to is not None and 0 <= days_to <= earnings_window:
                log.info(f"  {symbol} rejected — earnings in {days_to}d "
                         f"({earnings['date']})")
                return None
        else:
            earnings = {"date": None, "days_to": None}

        # ── Price action detection + score boost ─────────────────────────────
        pa = detect_price_action(df)
        final_score = min(130, scoring["score"] + pa["pa_score"])  # PA can push score above 100
        scoring["base_score"]  = scoring["score"]
        scoring["pa_score"]    = pa["pa_score"]
        scoring["final_score"] = final_score
        scoring["score"]       = final_score   # used for ranking

        log.info(f"  {symbol} — base {scoring['base_score']} + PA {pa['pa_score']} "
                 f"= {final_score}  [{pa['pattern_summary']}]")

        # ── Weekly trend filter ─────────────────────────────────────────────
        weekly_trend = {
            "trend": "neutral", "score": 50,
            "above_50ema": False, "above_200ema": False,
            "ema50_slope": 0.0, "weekly_rsi": 50.0,
            "weekly_macd_bull": False, "ema50": 0.0, "ema200": 0.0,
        }

        if filters.get("weekly_trend_filter", False):
            time.sleep(0.25)
            weekly_df    = self.fetch_weekly_ohlcv(symbol)
            weekly_trend = calc_weekly_trend(weekly_df)

            required = filters.get("weekly_trend_required", "bullish")
            if required == "bullish" and weekly_trend["trend"] != "bullish":
                log.info(f"  {symbol} rejected — weekly trend: {weekly_trend['trend']}")
                return None
            if required == "not_bearish" and weekly_trend["trend"] == "bearish":
                log.info(f"  {symbol} rejected — weekly trend bearish")
                return None

        return {
            "symbol":     symbol,
            "sector":     sector,
            "price":      current_price,
            "indicators": indicators,
            "scoring":    scoring,
            "weekly":     weekly_trend,
            "price_action": pa,
            "event_risk": {
                "surveillance":     surveillance,
                "earnings_date":    earnings["date"],
                "days_to_earnings": earnings["days_to"],
            },
        }

    def run(self, universe: str = "nifty50", filters: dict = None) -> list[dict]:
        if filters is None:
            filters = {}
        if not self._instrument_map:
            self.load_instruments()

        symbols = UNIVERSES.get(universe, UNIVERSES["nifty50"])
        weekly_on = filters.get("weekly_trend_filter", False)
        log.info(f"Screening {len(symbols)} stocks in {universe} "
                 f"[weekly filter: {'ON' if weekly_on else 'OFF'}]...")

        results = []
        for i, symbol in enumerate(symbols):
            log.info(f"  [{i+1}/{len(symbols)}] {symbol}...")
            result = self.analyse_stock(symbol, filters=filters)
            if result:
                results.append(result)
            time.sleep(0.35)

        results.sort(key=lambda x: x["scoring"]["score"], reverse=True)
        log.info(f"Done. {len(results)} qualified out of {len(symbols)} scanned.")
        return results


# ---------------------------------------------------------------------------
# Pretty printer / report
# ---------------------------------------------------------------------------

def print_report(results: list[dict], top_n: int = 10):
    print("\n" + "="*70)
    print(f"  SWING SCREENER RESULTS — Top {min(top_n, len(results))} picks")
    print("="*70)

    for i, r in enumerate(results[:top_n], 1):
        s   = r["scoring"]
        ind = r["indicators"]
        w   = r.get("weekly", {})
        print(f"\n#{i}  {r['symbol']}  ({r['sector']})  —  Score: {s['score']}/100  [{s['signal'].upper()}]")
        print(f"     Price:  ₹{r['price']:,.2f}")
        print(f"     Target: +{s['target_pct']:.1f}%  |  Stop: -{s['sl_pct']:.1f}%  |  R:R = 1:{s['rr']}")
        print(f"     RSI: {ind['rsi']}  |  Volume: {ind['volume_ratio']}x  |  ADX: {ind['adx']}")
        print(f"     MACD: {'Bullish' if ind['macd']['bullish'] else 'Bearish'}  |  "
              f"vs 20MA: {ind['ma20_dist_pct']:+.1f}%  |  5D mom: {ind['momentum_5d']:+.1f}%")
        print(f"     Support: ₹{ind['sr']['support']:,.0f}  |  "
              f"Resistance: ₹{ind['sr']['resistance']:,.0f}")
        if w.get("trend"):
            print(f"     Weekly: {w['trend'].upper()}  |  "
                  f"50EMA {'above' if w['above_50ema'] else 'below'}  |  "
                  f"200EMA {'above' if w['above_200ema'] else 'below'}  |  "
                  f"Slope {w['ema50_slope']:+.2f}%")

    print("\n" + "="*70)
    print("  DISCLAIMER: For educational purposes only. Not financial advice.")
    print("="*70 + "\n")


class _NumpyEncoder(json.JSONEncoder):
    """Handles numpy scalars that standard json cannot serialise."""
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.generic):   return obj.item()   # catches np.bool_, np.bool, all generics
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return super().default(obj)


def to_json(results: list[dict]) -> str:
    return json.dumps(results, cls=_NumpyEncoder, indent=2)
