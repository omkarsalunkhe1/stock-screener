"""
Kite Stock Analyzer - Swing Trade Recommender
==============================================
Fetches historical & live data from Kite API v3, applies technical analysis,
detects chart patterns, and recommends stocks for 1-3 week holds targeting ≥5% return.

Usage:
    python kite_analyzer.py                     # analyze default watchlist
    python kite_analyzer.py --symbols INFY TCS  # analyze specific symbols
    python kite_analyzer.py --top 10            # show top N recommendations
    python kite_analyzer.py --chart INFY        # open chart for a symbol
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MCP bridge — when running inside the Claude MCP context these are injected.
# When running standalone, install kiteconnect and set env vars.
# ---------------------------------------------------------------------------
try:
    from mcp_tools import (  # type: ignore[import]
        mcp_kite_get_historical_data,
        mcp_kite_get_quotes,
        mcp_kite_search_instruments,
        mcp_kite_get_holdings,
        mcp_kite_get_ltp,
    )
    MCP_MODE = True
except ImportError:
    MCP_MODE = False
    try:
        from kiteconnect import KiteConnect  # type: ignore[import]
    except ImportError:
        KiteConnect = None


# ---------------------------------------------------------------------------
# Default NSE watchlist (Nifty 50 + midcap picks)
# ---------------------------------------------------------------------------
DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "AXISBANK", "BAJFINANCE", "KOTAKBANK", "LT",
    "WIPRO", "HCLTECH", "TECHM", "MARUTI", "M&M",
    "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO", "ADANIENT",
    "ADANIPORTS", "POWERGRID", "NTPC", "COALINDIA", "ONGC",
    "BPCL", "IOC", "GAIL", "VEDL", "NATIONALUM",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "TITAN", "ASIANPAINT", "NESTLEIND", "HINDUNILVR", "ITC",
    "PIDILITIND", "BERGEPAINT", "DABUR", "MARICO", "COLPAL",
    "HDFCLIFE", "SBILIFE", "ICICIGI", "BAJAJFINSV", "SHRIRAMFIN",
    "ZOMATO", "PAYTM", "NYKAA", "POLICYBZR", "DELHIVERY",
    "IRCTC", "ABCAPITAL", "MUTHOOTFIN", "CHOLAFIN", "MANAPPURAM",
    "TATAPOWER", "TORNTPOWER", "CESC", "IEX", "NHPC",
    "UPL", "PI", "CHAMBLFERT", "COROMANDEL", "AARTIIND",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StockData:
    symbol: str
    token: int
    df: pd.DataFrame           # OHLCV with indicators
    quote: dict = field(default_factory=dict)


@dataclass
class Recommendation:
    symbol: str
    score: float               # 0-100
    signals: list[str]
    entry: float
    target: float
    stop_loss: float
    expected_return_pct: float
    risk_reward: float
    horizon: str               # "1 week" / "2 weeks" / "3 weeks"
    pattern: str
    current_price: float
    volume_signal: str


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def volume_ma(volume: pd.Series, period=20) -> pd.Series:
    return volume.rolling(period).mean()


def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    df["ema9"]  = ema(c, 9)
    df["ema21"] = ema(c, 21)
    df["ema50"] = ema(c, 50)
    df["ema200"]= ema(c, 200)

    df["rsi"] = rsi(c)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(c)

    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(c)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["atr"] = atr(h, lo, c)
    df["vol_ma20"] = volume_ma(v)
    df["vol_ratio"] = v / df["vol_ma20"]

    df["stoch_k"], df["stoch_d"] = stochastic(h, lo, c)

    # Supertrend (simplified)
    multiplier = 3.0
    basic_upper = (h + lo) / 2 + multiplier * df["atr"]
    basic_lower = (h + lo) / 2 - multiplier * df["atr"]
    df["supertrend_upper"] = basic_upper
    df["supertrend_lower"] = basic_lower
    df["supertrend_dir"] = np.where(c > basic_upper.shift(), 1,
                           np.where(c < basic_lower.shift(), -1, 0))

    # Price momentum
    df["momentum_5"]  = c.pct_change(5)  * 100
    df["momentum_10"] = c.pct_change(10) * 100
    df["momentum_20"] = c.pct_change(20) * 100

    return df


# ---------------------------------------------------------------------------
# Pattern Detection
# ---------------------------------------------------------------------------

def detect_patterns(df: pd.DataFrame) -> list[str]:
    """Return list of detected chart patterns from the last 30 candles."""
    patterns = []
    if len(df) < 30:
        return patterns

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    c = df["close"]
    h = df["high"]
    lo = df["low"]
    o = df["open"]

    # --- Trend ---
    if last["ema9"] > last["ema21"] > last["ema50"]:
        patterns.append("Uptrend (EMA 9>21>50)")
    if last["ema21"] > last["ema50"] > last["ema200"]:
        patterns.append("Strong Uptrend (above EMA200)")

    # --- Golden / Death Cross ---
    if prev["ema50"] <= prev["ema200"] and last["ema50"] > last["ema200"]:
        patterns.append("Golden Cross (EMA50 x EMA200)")
    if prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]:
        patterns.append("Bullish EMA Crossover (9x21)")

    # --- MACD ---
    if prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        patterns.append("MACD Bullish Crossover")
    if last["macd"] > 0 and last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
        patterns.append("MACD Histogram Expanding Bullish")

    # --- RSI ---
    if 30 < last["rsi"] < 50 and last["rsi"] > prev["rsi"]:
        patterns.append(f"RSI Recovering from Oversold ({last['rsi']:.1f})")
    if 50 < last["rsi"] < 70:
        patterns.append(f"RSI Bullish Zone ({last['rsi']:.1f})")
    if prev["rsi"] < 30 and last["rsi"] > 30:
        patterns.append("RSI Oversold Reversal")

    # --- Bollinger Bands ---
    if last["close"] > last["bb_upper"] and prev["close"] <= prev["bb_upper"]:
        patterns.append("Bollinger Band Breakout")
    if last["close"] < last["bb_lower"]:
        patterns.append("Bollinger Lower Band Bounce (Mean Reversion)")

    # --- Volume confirmation ---
    if last["vol_ratio"] > 2.0:
        patterns.append(f"High Volume Surge ({last['vol_ratio']:.1f}x avg)")
    elif last["vol_ratio"] > 1.5:
        patterns.append(f"Above Average Volume ({last['vol_ratio']:.1f}x avg)")

    # --- Support / Resistance Breakout (20-day high) ---
    recent_high = h.iloc[-21:-1].max()
    if last["close"] > recent_high:
        patterns.append("20-Day High Breakout")

    recent_low_20 = lo.iloc[-21:-1].min()
    if last["close"] < recent_low_20 * 1.02:
        pass  # near 20-day low — not bullish

    # --- Candlestick patterns ---
    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"] + 1e-9
    body_ratio = body / range_

    # Bullish engulfing
    if (prev["close"] < prev["open"] and          # prev red
        last["close"] > last["open"] and           # last green
        last["open"] < prev["close"] and
        last["close"] > prev["open"]):
        patterns.append("Bullish Engulfing Candle")

    # Hammer
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    upper_shadow = last["high"] - max(last["open"], last["close"])
    if lower_shadow > 2 * body and upper_shadow < body * 0.5 and last["close"] > last["open"]:
        patterns.append("Hammer / Dragonfly Doji")

    # Morning star (3 candle)
    if (prev2["close"] < prev2["open"] and          # bearish
        abs(prev["close"] - prev["open"]) < range_ * 0.3 and  # small body
        last["close"] > last["open"] and            # bullish
        last["close"] > (prev2["open"] + prev2["close"]) / 2):
        patterns.append("Morning Star Pattern")

    # Stochastic oversold cross
    if prev["stoch_k"] < 20 and last["stoch_k"] > last["stoch_d"] and prev["stoch_k"] <= prev["stoch_d"]:
        patterns.append("Stochastic Oversold Crossover")

    return patterns


# ---------------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------------

def score_stock(df: pd.DataFrame, patterns: list[str]) -> tuple[float, list[str]]:
    """
    Score a stock 0-100 for 1-3 week swing trade potential.
    Returns (score, triggered_signals).
    """
    score = 0.0
    signals = []

    if len(df) < 20:
        return 0.0, []

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # --- Trend alignment (max 25 pts) ---
    if last["ema9"] > last["ema21"]:
        score += 8; signals.append("Short-term uptrend")
    if last["ema21"] > last["ema50"]:
        score += 8; signals.append("Mid-term uptrend")
    if last["close"] > last["ema200"]:
        score += 9; signals.append("Above 200 EMA (long-term bullish)")

    # --- Momentum (max 20 pts) ---
    if last["momentum_5"] > 0:
        score += min(last["momentum_5"] * 0.8, 8)
        signals.append(f"5-day momentum: +{last['momentum_5']:.1f}%")
    if last["momentum_20"] > 0:
        score += min(last["momentum_20"] * 0.4, 12)

    # --- RSI (max 15 pts) ---
    rsi_val = last["rsi"]
    if 40 < rsi_val < 65:
        score += 15; signals.append(f"RSI ideal zone ({rsi_val:.0f})")
    elif 30 < rsi_val <= 40:
        score += 10; signals.append(f"RSI recovering ({rsi_val:.0f})")
    elif rsi_val > 65:
        score += 5   # overbought, less upside

    # --- MACD (max 15 pts) ---
    if last["macd"] > last["macd_signal"]:
        score += 8; signals.append("MACD bullish")
    if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
        score += 7; signals.append("MACD histogram expanding")

    # --- Volume (max 15 pts) ---
    vol_ratio = last["vol_ratio"]
    if vol_ratio > 1.5:
        score += min(vol_ratio * 4, 15); signals.append(f"Volume surge ({vol_ratio:.1f}x)")
    elif vol_ratio > 1.0:
        score += 5

    # --- Pattern bonus (max 10 pts) ---
    high_value_patterns = {
        "Bullish Engulfing Candle": 5,
        "Morning Star Pattern": 5,
        "MACD Bullish Crossover": 4,
        "Bullish EMA Crossover (9x21)": 4,
        "Golden Cross (EMA50 x EMA200)": 5,
        "Bollinger Band Breakout": 4,
        "20-Day High Breakout": 4,
        "RSI Oversold Reversal": 3,
        "Stochastic Oversold Crossover": 3,
        "Hammer / Dragonfly Doji": 3,
        "High Volume Surge": 4,
    }
    pattern_bonus = 0
    for p in patterns:
        for key, pts in high_value_patterns.items():
            if key in p:
                pattern_bonus += pts
    score += min(pattern_bonus, 10)

    return min(score, 100.0), signals


# ---------------------------------------------------------------------------
# Entry / Target / Stop-Loss Calculator
# ---------------------------------------------------------------------------

def calc_trade_levels(df: pd.DataFrame, current_price: float) -> tuple[float, float, float, float, str]:
    """
    Returns (entry, target, stop_loss, rr_ratio, horizon).
    Uses ATR-based method for realistic levels.
    """
    last = df.iloc[-1]
    atr_val = last["atr"]

    entry = current_price
    # Target: 5-10% above, but also ATR-based
    target_pct = max(0.05, min(atr_val / entry * 5, 0.12))
    target = round(entry * (1 + target_pct), 2)

    # Stop: 1.5x ATR below entry
    stop_loss = round(entry - 1.5 * atr_val, 2)
    stop_loss = max(stop_loss, entry * 0.93)  # max 7% loss

    risk = entry - stop_loss
    reward = target - entry
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    # Horizon based on ATR relative to price
    volatility = atr_val / entry
    if volatility > 0.025:
        horizon = "1 week"
    elif volatility > 0.015:
        horizon = "2 weeks"
    else:
        horizon = "3 weeks"

    return entry, target, stop_loss, rr_ratio, horizon


# ---------------------------------------------------------------------------
# Kite Data Fetcher
# ---------------------------------------------------------------------------

class KiteFetcher:
    def __init__(self, kite=None):
        self.kite = kite  # KiteConnect instance (standalone mode)
        self._token_cache: dict[str, int] = {}

    def get_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        key = f"{exchange}:{symbol}"
        if key in self._token_cache:
            return self._token_cache[key]

        try:
            results = mcp_kite_search_instruments(query=f"{exchange}:{symbol}", filter_on="id", limit=5)
            for r in (results if isinstance(results, list) else []):
                if r.get("tradingsymbol") == symbol and r.get("exchange") == exchange:
                    token = r["instrument_token"]
                    self._token_cache[key] = token
                    return token
        except Exception as e:
            print(f"  [WARN] Token lookup failed for {symbol}: {e}", file=sys.stderr)
        return None

    def fetch_history(self, symbol: str, days: int = 90, interval: str = "day") -> Optional[pd.DataFrame]:
        token = self.get_token(symbol)
        if token is None:
            return None

        to_date   = datetime.now()
        from_date = to_date - timedelta(days=days)

        try:
            data = mcp_kite_get_historical_data(
                instrument_token=token,
                from_date=from_date.strftime("%Y-%m-%d 09:00:00"),
                to_date=to_date.strftime("%Y-%m-%d 15:30:00"),
                interval=interval,
            )
        except Exception as e:
            print(f"  [WARN] History fetch failed for {symbol}: {e}", file=sys.stderr)
            return None

        if not data:
            return None

        rows = []
        for candle in data:
            rows.append({
                "date":   candle[0],
                "open":   float(candle[1]),
                "high":   float(candle[2]),
                "low":    float(candle[3]),
                "close":  float(candle[4]),
                "volume": float(candle[5]),
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    def fetch_quote(self, symbol: str, exchange: str = "NSE") -> dict:
        try:
            key = f"{exchange}:{symbol}"
            result = mcp_kite_get_quotes(instruments=[key])
            return result.get(key, {})
        except Exception:
            return {}

    def fetch_ltp(self, symbols: list[str], exchange: str = "NSE") -> dict[str, float]:
        keys = [f"{exchange}:{s}" for s in symbols]
        try:
            result = mcp_kite_get_ltp(instruments=keys)
            return {s.split(":")[1]: v.get("last_price", 0) for s, v in result.items()}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------

class StockAnalyzer:
    def __init__(self, fetcher: KiteFetcher):
        self.fetcher = fetcher

    def analyze_symbol(self, symbol: str) -> Optional[Recommendation]:
        df = self.fetcher.fetch_history(symbol, days=120)
        if df is None or len(df) < 30:
            return None

        df = add_indicators(df)
        patterns = detect_patterns(df)
        score, signals = score_stock(df, patterns)

        quote = self.fetcher.fetch_quote(symbol)
        current_price = quote.get("last_price") or df["close"].iloc[-1]

        entry, target, stop_loss, rr, horizon = calc_trade_levels(df, current_price)
        expected_return = round((target - entry) / entry * 100, 2)

        vol_ratio = df["vol_ratio"].iloc[-1]
        if vol_ratio > 2.0:
            vol_signal = f"HIGH ({vol_ratio:.1f}x avg)"
        elif vol_ratio > 1.2:
            vol_signal = f"ABOVE AVG ({vol_ratio:.1f}x)"
        else:
            vol_signal = f"Normal ({vol_ratio:.1f}x)"

        return Recommendation(
            symbol=symbol,
            score=score,
            signals=signals,
            entry=entry,
            target=target,
            stop_loss=stop_loss,
            expected_return_pct=expected_return,
            risk_reward=rr,
            horizon=horizon,
            pattern=", ".join(patterns[:3]) if patterns else "No strong pattern",
            current_price=current_price,
            volume_signal=vol_signal,
        )

    def run(self, symbols: list[str], top_n: int = 10, min_score: float = 40.0) -> list[Recommendation]:
        recs = []
        total = len(symbols)
        for i, sym in enumerate(symbols, 1):
            print(f"  Analyzing {sym} ({i}/{total})...", end="\r")
            try:
                rec = self.analyze_symbol(sym)
                if rec and rec.score >= min_score and rec.expected_return_pct >= 5.0 and rec.risk_reward >= 1.5:
                    recs.append(rec)
            except Exception as e:
                print(f"\n  [ERROR] {sym}: {e}", file=sys.stderr)
            time.sleep(0.05)  # gentle rate limiting

        print(" " * 60, end="\r")  # clear progress line
        recs.sort(key=lambda r: r.score, reverse=True)
        return recs[:top_n]


# ---------------------------------------------------------------------------
# Report Printer
# ---------------------------------------------------------------------------

def print_report(recs: list[Recommendation]) -> None:
    if not recs:
        print("\nNo stocks met the criteria (score≥40, return≥5%, RR≥1.5).")
        return

    divider = "=" * 100

    print(f"\n{divider}")
    print(f"  SWING TRADE RECOMMENDATIONS  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Target: 5%+ in 1-3 weeks")
    print(divider)

    header = f"{'#':<3} {'Symbol':<12} {'Score':>6} {'Price':>9} {'Entry':>9} {'Target':>9} {'SL':>9} {'Return%':>8} {'R:R':>6} {'Horizon':<10} {'Volume'}"
    print(header)
    print("-" * 100)

    for i, r in enumerate(recs, 1):
        print(
            f"{i:<3} {r.symbol:<12} {r.score:>6.1f} "
            f"{r.current_price:>9.2f} {r.entry:>9.2f} {r.target:>9.2f} "
            f"{r.stop_loss:>9.2f} {r.expected_return_pct:>7.1f}% "
            f"{r.risk_reward:>5.1f}x  {r.horizon:<10} {r.volume_signal}"
        )

    print(divider)

    print("\n--- DETAILED SIGNALS ---")
    for r in recs:
        print(f"\n[{r.symbol}] Score: {r.score:.0f}/100 | Pattern: {r.pattern}")
        for s in r.signals:
            print(f"   + {s}")

    print(f"\n{divider}")
    print("DISCLAIMER: This is algorithmic analysis only. NOT financial advice.")
    print("Always do your own due diligence before trading.")
    print(divider)


def export_json(recs: list[Recommendation], path: str = "recommendations.json") -> None:
    data = []
    for r in recs:
        data.append({
            "symbol": r.symbol,
            "score": r.score,
            "current_price": r.current_price,
            "entry": r.entry,
            "target": r.target,
            "stop_loss": r.stop_loss,
            "expected_return_pct": r.expected_return_pct,
            "risk_reward": r.risk_reward,
            "horizon": r.horizon,
            "pattern": r.pattern,
            "signals": r.signals,
            "volume_signal": r.volume_signal,
            "generated_at": datetime.now().isoformat(),
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nExported {len(data)} recommendations -> {path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kite Swing Trade Analyzer")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to analyze")
    parser.add_argument("--top", type=int, default=10, help="Top N recommendations")
    parser.add_argument("--min-score", type=float, default=40.0, help="Minimum score threshold")
    parser.add_argument("--export", action="store_true", help="Export results to JSON")
    parser.add_argument("--chart", help="Open interactive chart for a symbol")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_WATCHLIST

    print(f"\nKite Swing Trade Analyzer")
    print(f"Analyzing {len(symbols)} stocks...")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    fetcher = KiteFetcher()
    analyzer = StockAnalyzer(fetcher)

    if args.chart:
        # Import dashboard for charting
        try:
            from dashboard import show_chart
            df = fetcher.fetch_history(args.chart, days=120)
            if df is not None:
                df = add_indicators(df)
                show_chart(args.chart, df)
            else:
                print(f"Could not fetch data for {args.chart}")
        except ImportError:
            print("Install plotly to use charts: pip install plotly")
        return

    recs = analyzer.run(symbols, top_n=args.top, min_score=args.min_score)
    print_report(recs)

    if args.export:
        export_json(recs)

    return recs


if __name__ == "__main__":
    main()
