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
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UNIVERSES = {
    "nifty50": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
        "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
        "TITAN", "WIPRO", "ULTRACEMCO", "BAJFINANCE", "NESTLEIND", "POWERGRID",
        "ONGC", "NTPC", "TECHM", "SUNPHARMA", "TATAMOTORS", "HCLTECH", "M&M",
        "ADANIENT", "BAJAJFINSV", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
        "HEROMOTOCO", "INDUSINDBK", "JSWSTEEL", "CIPLA", "COALINDIA", "BRITANNIA",
        "APOLLOHOSP", "TATACONSUM", "HINDALCO", "BPCL", "UPL", "BAJAJ-AUTO",
        "SHREECEM", "ADANIPORTS", "SBILIFE", "HDFCLIFE", "TATASTEEL",
    ],
    "nifty100": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
        "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
        "TITAN", "WIPRO", "ULTRACEMCO", "BAJFINANCE", "NESTLEIND", "POWERGRID",
        "ONGC", "NTPC", "TECHM", "SUNPHARMA", "TATAMOTORS", "HCLTECH", "M&M",
        "BAJAJFINSV", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HEROMOTOCO",
        "INDUSINDBK", "JSWSTEEL", "CIPLA", "COALINDIA", "BRITANNIA", "APOLLOHOSP",
        "TATACONSUM", "HINDALCO", "BPCL", "BAJAJ-AUTO", "SHREECEM", "ADANIPORTS",
        "MUTHOOTFIN", "PERSISTENT", "TATAELXSI", "CHOLAFIN", "DIXON",
        "ABCAPITAL", "ESCORTS", "VOLTAS", "LALPATHLAB", "NAUKRI",
        "BANDHANBNK", "IDFCFIRSTB", "FEDERALBNK", "CANBK", "PNB",
        "SAIL", "NMDC", "GAIL", "IOC", "HPCL",
        "PIDILITIND", "BERGEPAINT", "WHIRLPOOL", "HAVELLS", "POLYCAB",
        "MCDOWELL-N", "COLPAL", "MARICO", "DABUR", "GODREJCP",
        "LUPIN", "AUROPHARMA", "BIOCON", "TORNTPHARM", "IPCALAB",
        "IRCTC", "CONCOR", "OBEROIRLTY", "PRESTIGE", "DLF",
        "CRISIL", "CREDITACC", "AAVAS", "HOMEFIRST", "FIVE-STAR",
        "TANLA", "COFORGE", "MPHASIS", "LTTS", "KPIT",
    ],
    "midcap": [
        "MUTHOOTFIN", "PERSISTENT", "TATAELXSI", "CHOLAFIN", "DIXON",
        "ABCAPITAL", "ESCORTS", "VOLTAS", "LALPATHLAB", "NAUKRI",
        "TANLA", "COFORGE", "MPHASIS", "LTTS", "KPIT",
        "CRISIL", "IRCTC", "CONCOR", "OBEROIRLTY", "PRESTIGE",
        "HAVELLS", "POLYCAB", "PIDILITIND", "BERGEPAINT", "WHIRLPOOL",
        "LUPIN", "AUROPHARMA", "BIOCON", "TORNTPHARM", "IPCALAB",
    ],
}


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
    if len(close) <= days:
        return 0.0
    return round(float((close.iloc[-1] / close.iloc[-days] - 1) * 100), 2)


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

    Scoring (0-100):
      35 pts — price above weekly 50 EMA  (most important)
      25 pts — price above weekly 200 EMA (long-term uptrend)
      20 pts — 50 EMA slope is rising     (trend still intact)
      10 pts — weekly RSI > 50            (weekly momentum positive)
      10 pts — weekly MACD bullish        (weekly momentum accelerating)

    trend:
      >= 70  → bullish   (daily setups in these stocks are high-quality)
      <= 35  → bearish   (daily signals here are likely dead cat bounces)
      middle → neutral
    """
    empty = {
        "trend": "neutral", "score": 50,
        "above_50ema": False, "above_200ema": False,
        "ema50_slope": 0.0, "weekly_rsi": 50.0,
        "weekly_macd_bull": False, "ema50": 0.0, "ema200": 0.0,
    }
    if weekly_df is None or len(weekly_df) < 20:
        return empty

    close = weekly_df["close"]
    current = float(close.iloc[-1])

    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    above_50  = current > float(ema50.iloc[-1])
    above_200 = current > float(ema200.iloc[-1])

    ema50_slope = round(
        float(ema50.iloc[-1] - ema50.iloc[-5]) / float(ema50.iloc[-5]) * 100, 3
    ) if len(ema50) >= 5 else 0.0

    weekly_rsi = calc_rsi(close, period=14)
    weekly_macd = calc_macd(close)

    score = 0
    if above_50:                    score += 35
    if above_200:                   score += 25
    if ema50_slope > 0:             score += 20
    if weekly_rsi > 50:             score += 10
    if weekly_macd["bullish"]:      score += 10

    trend = "bullish" if score >= 70 else "bearish" if score <= 35 else "neutral"

    return {
        "trend": trend,
        "score": score,
        "above_50ema": above_50,
        "above_200ema": above_200,
        "ema50_slope": ema50_slope,
        "weekly_rsi": round(weekly_rsi, 1),
        "weekly_macd_bull": weekly_macd["bullish"],
        "ema50": round(float(ema50.iloc[-1]), 2),
        "ema200": round(float(ema200.iloc[-1]), 2),
    }



# ---------------------------------------------------------------------------
# Price action detectors
# ---------------------------------------------------------------------------

def detect_price_action(df) -> dict:
    """
    Detect key price action patterns on the last 20 daily candles.

    Patterns:
      1. Hammer              — long lower wick, reversal after pullback
      2. Bullish engulfing   — today fully engulfs prior red candle
      3. Resistance breakout — close above 20-day high with volume surge
      4. Support bounce      — touched 20-day low, closed in upper 40% green
      5. Inside bar (bullish)— range contained in prior candle, closes green
      6. Higher high + HL    — last 3-candle window beats prior window

    pa_score: 0-30 bonus added on top of the base 100-pt score.
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
        support      = float(l[-20:-1].min())
        support_zone = support * 1.015
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

    # PA score — bonus points (max 30)
    pa_score = 0
    if patterns["bullish_engulfing"]: pa_score += 12
    if patterns["breakout"]:          pa_score += 12
    if patterns["hammer"]:            pa_score += 8
    if patterns["support_bounce"]:    pa_score += 8
    if patterns["hh_hl"]:             pa_score += 6
    if patterns["inside_bar"]:        pa_score += 4
    pa_score = min(30, pa_score)

    return {
        "patterns":        patterns,
        "active":          active,
        "pa_score":        pa_score,
        "has_pattern":     len(active) > 0,
        "pattern_summary": ", ".join(active) if active else "None",
    }


def _empty_pa() -> dict:
    return {
        "patterns":        {},
        "active":          [],
        "pa_score":        0,
        "has_pattern":     False,
        "pattern_summary": "None",
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
    if 0 < ma_dist <= 3:      pts = 15
    elif ma_dist > 3:         pts = 8
    elif -1 <= ma_dist <= 0:  pts = 5
    else:                     pts = 0
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

    bb_pos = indicators["bollinger"]["position"]
    if 0.35 <= bb_pos <= 0.65:    pts = 5
    elif 0.25 <= bb_pos <= 0.75:  pts = 3
    else:                         pts = 0
    score += pts
    breakdown["bollinger"] = {"value": bb_pos, "points": pts, "max": 5}

    score = min(100, score)

    atr = indicators["atr"]
    price = indicators["current_price"]
    target_pct = round((atr * 3) / price * 100, 2)
    sl_pct     = round((atr * 1.5) / price * 100, 2)
    rr         = round(target_pct / sl_pct, 2) if sl_pct > 0 else 0
    target_pct = max(target_pct, filters.get("min_target_pct", 5.0))

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

class SwingScreener:
    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._instrument_map: dict = {}

    def load_instruments(self):
        log.info("Loading NSE instruments...")
        instruments = self.kite.instruments("NSE")
        self._instrument_map = {
            inst["tradingsymbol"]: inst["instrument_token"]
            for inst in instruments
            if inst["segment"] == "NSE"
        }
        log.info(f"Loaded {len(self._instrument_map)} NSE instruments")

    def _get_token(self, symbol: str) -> Optional[int]:
        return self._instrument_map.get(symbol)

    def _fetch(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        token = self._get_token(symbol)
        if not token:
            log.warning(f"Token not found: {symbol}")
            return None
        to_date   = datetime.now()
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
            return df.sort_values("date").reset_index(drop=True)
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

        # ── Daily analysis ──────────────────────────────────────────────────
        df = self.fetch_ohlcv(symbol)
        if df is None or len(df) < 30:
            return None

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        current_price = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])

        indicators = {
            "current_price": current_price,
            "rsi":           calc_rsi(close),
            "macd":          calc_macd(close),
            "bollinger":     calc_bollinger(close),
            "adx":           calc_adx(high, low, close),
            "volume_ratio":  calc_volume_ratio(volume),
            "momentum_5d":   calc_momentum(close, 5),
            "momentum_20d":  calc_momentum(close, 20),
            "ma20":          round(ma20, 2),
            "ma20_dist_pct": round((current_price - ma20) / ma20 * 100, 2),
            "atr":           calc_atr(high, low, close),
            "sr":            calc_support_resistance(close, high, low),
            "chg_1d":        calc_momentum(close, 1),
            "chg_5d":        calc_momentum(close, 5),
        }

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
        if scoring["target_pct"] < filters.get("min_target_pct", 5.0):
            return None

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


def to_json(results: list[dict]) -> str:
    def clean(obj):
        if isinstance(obj, (np.integer,)):        return int(obj)
        if isinstance(obj, (np.floating,)):       return float(obj)
        if isinstance(obj, (np.bool_,)):          return bool(obj)
        if isinstance(obj, bool):                 return obj
        if isinstance(obj, dict):                 return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):        return [clean(v) for v in obj]
        return obj
    return json.dumps(clean(results), indent=2)
