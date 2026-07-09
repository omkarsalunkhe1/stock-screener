"""
Kite Stock Analysis Web Tool
=============================
A standalone web application for real-time NSE stock analysis.

Setup:
  1. python get_token.py          <- generate your daily access token
  2. python app.py                <- start the server
  3. Open http://localhost:8000   <- use the dashboard

Requirements: pip install -r requirements.txt
"""

import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# ── FastAPI ──────────────────────────────────────────────────────────────────
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

# ── KiteConnect ───────────────────────────────────────────────────────────────
try:
    from kiteconnect import KiteConnect
    KITE_OK = True
except ImportError:
    KITE_OK = False
    print("kiteconnect not installed. Run: pip install kiteconnect")
    KiteConnect = None

# ── Plotly ────────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    print("plotly not installed. Run: pip install plotly")


# =============================================================================
# CONFIG
# =============================================================================

CONFIG_FILE = Path(__file__).parent / "kite_config.json"

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        default = {
            "api_key": "YOUR_API_KEY_HERE",
            "api_secret": "YOUR_API_SECRET_HERE",
            "access_token": "YOUR_ACCESS_TOKEN_HERE"
        }
        CONFIG_FILE.write_text(json.dumps(default, indent=2))
        print(f"[!] Created {CONFIG_FILE}. Please fill in your Kite API credentials.")
        return default
    return json.loads(CONFIG_FILE.read_text())

config = load_config()


# =============================================================================
# KITE SETUP
# =============================================================================

kite: Optional["KiteConnect"] = None
KITE_CONNECTED = False
KITE_USER = "—"

def init_kite() -> bool:
    global kite, KITE_CONNECTED, KITE_USER
    if not KITE_OK:
        return False
    if config.get("access_token", "").startswith("YOUR_"):
        print("[!] Access token not configured. Run: python get_token.py")
        return False
    try:
        kite = KiteConnect(api_key=config["api_key"])
        kite.set_access_token(config["access_token"])
        profile = kite.profile()
        KITE_USER = profile.get("user_name", "User")
        KITE_CONNECTED = True
        print(f"[✓] Connected to Kite as: {KITE_USER}")
        return True
    except Exception as e:
        print(f"[✗] Kite connection failed: {e}")
        print("    Run: python get_token.py to refresh your access token")
        KITE_CONNECTED = False
        return False

KITE_CONNECTED = init_kite()


# =============================================================================
# INSTRUMENT TOKEN LOOKUP
# =============================================================================

_token_cache: Dict[str, int] = {}
_instruments_nse: List[dict] = []

def get_instrument_token(symbol: str, exchange: str = "NSE") -> Optional[int]:
    key = f"{exchange}:{symbol}"
    if key in _token_cache:
        return _token_cache[key]

    global _instruments_nse
    if not kite:
        return None

    # Download instruments list once and cache
    if not _instruments_nse:
        try:
            print(f"[~] Downloading NSE instrument list...")
            _instruments_nse = kite.instruments("NSE")
            print(f"[✓] Loaded {len(_instruments_nse)} instruments")
        except Exception as e:
            print(f"[✗] Could not load instruments: {e}")
            return None

    for inst in _instruments_nse:
        if inst.get("tradingsymbol") == symbol and inst.get("instrument_type") == "EQ":
            _token_cache[key] = inst["instrument_token"]
            return inst["instrument_token"]

    return None


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi_calc(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_calc(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std

def atr_calc(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]

    df["ema9"]   = ema(c, 9)
    df["ema21"]  = ema(c, 21)
    df["ema50"]  = ema(c, 50)
    df["ema200"] = ema(c, 200)

    df["rsi"] = rsi_calc(c)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_calc(c)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(c)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["atr"]      = atr_calc(h, lo, c)
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]

    df["momentum_5"]  = c.pct_change(5)  * 100
    df["momentum_10"] = c.pct_change(10) * 100
    df["momentum_20"] = c.pct_change(20) * 100

    stoch_low  = lo.rolling(14).min()
    stoch_high = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - stoch_low) / (stoch_high - stoch_low + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    return df


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def detect_patterns(df: pd.DataFrame) -> List[str]:
    if len(df) < 30:
        return []

    patterns = []
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Trend structure
    if last["ema9"] > last["ema21"] > last["ema50"]:
        patterns.append("Uptrend (EMA 9 > 21 > 50)")
    if last["ema21"] > last["ema50"] > last["ema200"]:
        patterns.append("Strong Uptrend (above all EMAs)")
    if last["ema9"] < last["ema21"] < last["ema50"]:
        patterns.append("Downtrend (EMA 9 < 21 < 50)")

    # EMA crossovers
    if prev["ema50"] <= prev["ema200"] and last["ema50"] > last["ema200"]:
        patterns.append("Golden Cross (EMA50 x EMA200)")
    if prev["ema50"] >= prev["ema200"] and last["ema50"] < last["ema200"]:
        patterns.append("Death Cross (EMA50 x EMA200)")
    if prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]:
        patterns.append("Bullish EMA Crossover (9 x 21)")

    # MACD
    if prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        patterns.append("MACD Bullish Crossover")
    if last["macd"] > 0 and last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
        patterns.append("MACD Momentum Building")
    if last["macd"] < 0 and last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]:
        patterns.append("MACD Bearish Momentum")

    # RSI
    rsi_val = last["rsi"]
    if prev["rsi"] < 30 and rsi_val > 30:
        patterns.append("RSI Oversold Reversal")
    elif 40 < rsi_val < 65:
        patterns.append(f"RSI Bullish Zone ({rsi_val:.0f})")
    elif rsi_val > 70:
        patterns.append(f"RSI Overbought ({rsi_val:.0f}) — Caution")
    elif rsi_val < 30:
        patterns.append(f"RSI Oversold ({rsi_val:.0f})")

    # Bollinger Bands
    if last["close"] > last["bb_upper"]:
        patterns.append("Bollinger Band Breakout (Upper)")
    elif last["close"] < last["bb_lower"]:
        patterns.append("Bollinger Lower Band — Mean Reversion Setup")
    if last["bb_width"] < df["bb_width"].rolling(20).mean().iloc[-1] * 0.7:
        patterns.append("Bollinger Squeeze — Volatility Breakout Imminent")

    # Volume
    vol_ratio = last["vol_ratio"]
    if not np.isnan(vol_ratio):
        if vol_ratio > 3.0:
            patterns.append(f"Exceptional Volume Surge ({vol_ratio:.1f}x avg)")
        elif vol_ratio > 2.0:
            patterns.append(f"High Volume Day ({vol_ratio:.1f}x avg)")
        elif vol_ratio > 1.5:
            patterns.append(f"Above Average Volume ({vol_ratio:.1f}x avg)")

    # 20-day breakout
    if last["close"] > df["high"].iloc[-21:-1].max():
        patterns.append("20-Day High Breakout")

    # Candlestick patterns
    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"] + 1e-9
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    upper_shadow = last["high"] - max(last["open"], last["close"])

    # Bullish engulfing
    if (prev["close"] < prev["open"] and
            last["close"] > last["open"] and
            last["open"] < prev["close"] and
            last["close"] > prev["open"]):
        patterns.append("Bullish Engulfing Candle")

    # Hammer
    if lower_shadow > 2 * body and upper_shadow < body * 0.5 and last["close"] > last["open"]:
        patterns.append("Hammer / Dragonfly Doji")

    # Morning star
    if (prev2["close"] < prev2["open"] and
            abs(prev["close"] - prev["open"]) < range_ * 0.3 and
            last["close"] > last["open"] and
            last["close"] > (prev2["open"] + prev2["close"]) / 2):
        patterns.append("Morning Star Pattern")

    # Stochastic oversold cross
    if prev["stoch_k"] < 20 and last["stoch_k"] > last["stoch_d"] and prev["stoch_k"] <= prev["stoch_d"]:
        patterns.append("Stochastic Oversold Crossover")

    return patterns


# =============================================================================
# SCORING ENGINE (0-100)
# =============================================================================

def score_stock(df: pd.DataFrame, patterns: List[str]) -> tuple:
    if len(df) < 20:
        return 0.0, []

    score = 0.0
    signals = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── Trend (max 25 pts) ──
    if last["ema9"] > last["ema21"]:
        score += 8;  signals.append("Short-term uptrend (EMA9 > EMA21)")
    if last["ema21"] > last["ema50"]:
        score += 8;  signals.append("Mid-term uptrend (EMA21 > EMA50)")
    if last["close"] > last["ema200"]:
        score += 9;  signals.append("Above 200 EMA — long-term bullish")

    # ── Momentum (max 20 pts) ──
    if last["momentum_5"] > 0:
        score += min(last["momentum_5"] * 0.8, 8)
        signals.append(f"5-day momentum: +{last['momentum_5']:.1f}%")
    if last["momentum_20"] > 0:
        score += min(last["momentum_20"] * 0.4, 12)
        signals.append(f"20-day momentum: +{last['momentum_20']:.1f}%")

    # ── RSI (max 15 pts) ──
    rsi_val = last["rsi"]
    if 40 < rsi_val < 65:
        score += 15;  signals.append(f"RSI in ideal buy zone ({rsi_val:.0f})")
    elif 30 < rsi_val <= 40:
        score += 10;  signals.append(f"RSI recovering from oversold ({rsi_val:.0f})")
    elif rsi_val > 65:
        score += 5    # overbought — less upside

    # ── MACD (max 15 pts) ──
    if last["macd"] > last["macd_signal"]:
        score += 8;  signals.append("MACD above signal line — bullish")
    if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
        score += 7;  signals.append("MACD histogram expanding — momentum building")

    # ── Volume (max 15 pts) ──
    vol_ratio = last["vol_ratio"]
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.5:
            score += min(vol_ratio * 4, 15)
            signals.append(f"Volume surge: {vol_ratio:.1f}x average")
        elif vol_ratio > 1.0:
            score += 5

    # ── Pattern bonus (max 10 pts) ──
    bonus_map = {
        "Bullish Engulfing": 5,
        "Morning Star": 5,
        "MACD Bullish Crossover": 4,
        "Bullish EMA Crossover": 4,
        "Golden Cross": 5,
        "Bollinger Band Breakout": 4,
        "20-Day High Breakout": 4,
        "RSI Oversold Reversal": 3,
        "Hammer": 3,
        "Stochastic Oversold": 3,
        "Volume Surge": 3,
    }
    bonus = 0
    for p in patterns:
        for key, pts in bonus_map.items():
            if key in p:
                bonus += pts
                break
    score += min(bonus, 10)

    return min(score, 100.0), signals


# =============================================================================
# TRADE LEVEL CALCULATOR
# =============================================================================

def calc_trade_levels(df: pd.DataFrame, price: float):
    last = df.iloc[-1]
    atr_val = float(last["atr"])

    entry = price
    target_pct = max(0.05, min(atr_val / entry * 5, 0.12))
    target    = round(entry * (1 + target_pct), 2)
    stop_loss = round(max(entry - 1.5 * atr_val, entry * 0.93), 2)

    risk   = entry - stop_loss
    reward = target - entry
    rr     = round(reward / risk, 2) if risk > 0 else 0

    volatility = atr_val / entry
    horizon = "1 week" if volatility > 0.025 else ("2 weeks" if volatility > 0.015 else "3 weeks")

    return entry, target, stop_loss, rr, horizon


# =============================================================================
# CHART GENERATOR
# =============================================================================

def generate_chart_json(symbol: str, df: pd.DataFrame, rec: dict = None) -> dict:
    if not PLOTLY_OK or df is None or df.empty:
        return {}

    dates = df.index.strftime("%Y-%m-%d").tolist()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        vertical_spacing=0.02,
        subplot_titles=[f"{symbol} — Price & Indicators", "Volume", "RSI (14)", "MACD (12,26,9)"],
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # EMAs
    for col, color, label in [
        ("ema9",   "#ff9800", "EMA9"),
        ("ema21",  "#2196f3", "EMA21"),
        ("ema50",  "#9c27b0", "EMA50"),
        ("ema200", "#f44336", "EMA200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=df[col].round(2), name=label,
                line=dict(color=color, width=1.3),
            ), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["bb_upper"].round(2), name="BB Upper",
            line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df["bb_lower"].round(2), name="BB Lower",
            line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,255,0.04)",
        ), row=1, col=1)

    # Trade levels
    if rec:
        for level, color, label in [
            (rec.get("entry"),     "#ffd600", f"Entry {rec.get('entry', 0):.0f}"),
            (rec.get("target"),    "#00e676", f"Target {rec.get('target', 0):.0f} (+{rec.get('return_pct', 0):.1f}%)"),
            (rec.get("stop_loss"), "#ff1744", f"SL {rec.get('stop_loss', 0):.0f}"),
        ]:
            if level:
                fig.add_hline(
                    y=level, line_dash="dash", line_color=color,
                    annotation_text=label,
                    annotation_font_color=color,
                    annotation_bgcolor="rgba(0,0,0,0.5)",
                    row=1, col=1,
                )

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=dates, y=df["volume"], name="Volume",
        marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)
    if "vol_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["vol_ma20"].round(0), name="Vol MA20",
            line=dict(color="#ff9800", width=1.2),
        ), row=2, col=1)

    # ── RSI ──────────────────────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["rsi"].round(2), name="RSI",
            line=dict(color="#90caf9", width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.6)",  row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.6)", row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(144,202,249,0.04)", row=3, col=1)

    # ── MACD ─────────────────────────────────────────────────────────────────
    if "macd" in df.columns:
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350"
                       for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=dates, y=df["macd_hist"].round(3), name="Histogram",
            marker_color=hist_colors, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df["macd"].round(3), name="MACD",
            line=dict(color="#2196f3", width=1.5),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df["macd_signal"].round(3), name="Signal",
            line=dict(color="#ff9800", width=1.5),
        ), row=4, col=1)

    # ── Styling ───────────────────────────────────────────────────────────────
    title = symbol
    if rec:
        title += f"  |  Score: {rec.get('score', 0):.0f}/100  |  {rec.get('recommendation', '')}  |  {rec.get('trend', '')}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="white")),
        template="plotly_dark",
        height=900,
        paper_bgcolor="#0d1526",
        plot_bgcolor="#111827",
        font=dict(color="#ccd6f6"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=60, r=60, t=80, b=30),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", showspikes=True,
                     spikemode="across", spikesnap="cursor", spikecolor="rgba(255,255,255,0.3)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")

    return json.loads(fig.to_json())


# =============================================================================
# CORE STOCK ANALYSIS
# =============================================================================

def fetch_and_analyze(symbol: str, days: int = 120) -> dict:
    result = {
        "symbol": symbol,
        "status": "error",
        "error": None,
        "ltp": 0,
        "score": 0,
        "recommendation": "—",
        "rec_color": "#888",
        "entry": 0, "target": 0, "stop_loss": 0,
        "return_pct": 0, "rr": 0,
        "rsi": 0, "rsi_signal": "—",
        "macd_signal": "—",
        "trend": "—",
        "patterns": [],
        "signals": [],
        "horizon": "—",
        "vol_ratio": 0,
        "ema9": 0, "ema21": 0, "ema50": 0, "ema200": 0,
        "atr": 0,
        "momentum_5": 0, "momentum_20": 0,
        "_df": None,
    }

    if not kite:
        result["error"] = "Kite not connected — run python get_token.py"
        return result

    try:
        token = get_instrument_token(symbol)
        if token is None:
            result["error"] = f"Symbol not found on NSE: {symbol}"
            return result

        # Historical data (extra buffer for indicator warmup)
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=days + 60)

        hist = kite.historical_data(
            instrument_token=token,
            from_date=from_dt,
            to_date=to_dt,
            interval="day",
        )

        if not hist or len(hist) < 30:
            result["error"] = f"Insufficient data ({len(hist) if hist else 0} candles)"
            return result

        df = pd.DataFrame(hist)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        # Keep only needed columns
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.apply(pd.to_numeric, errors="coerce")

        # Live price
        try:
            ltp_data = kite.ltp(f"NSE:{symbol}")
            ltp = float(ltp_data[f"NSE:{symbol}"]["last_price"])
        except Exception:
            ltp = float(df["close"].iloc[-1])

        # Indicators
        df = add_indicators(df)

        # Patterns & score
        patterns = detect_patterns(df)
        score, signals = score_stock(df, patterns)

        # Trade levels
        entry, target, stop_loss, rr, horizon = calc_trade_levels(df, ltp)
        return_pct = round((target - entry) / entry * 100, 2)

        # Recommendation
        if score >= 65 and return_pct >= 5 and rr >= 1.5:
            rec, rec_color = "BUY",   "#00e676"
        elif score >= 45:
            rec, rec_color = "WATCH", "#ffd600"
        else:
            rec, rec_color = "AVOID", "#ef5350"

        # Trend label
        last = df.iloc[-1]
        if last["ema9"] > last["ema21"] > last["ema50"] > last["ema200"]:
            trend = "Strong Uptrend"
        elif last["ema9"] > last["ema21"] > last["ema50"]:
            trend = "Uptrend"
        elif last["ema9"] < last["ema21"] < last["ema50"] < last["ema200"]:
            trend = "Strong Downtrend"
        elif last["ema9"] < last["ema21"] < last["ema50"]:
            trend = "Downtrend"
        else:
            trend = "Sideways"

        # MACD label
        if last["macd"] > last["macd_signal"] and last["macd_hist"] > 0:
            macd_sig = "Bullish"
        elif last["macd"] > last["macd_signal"]:
            macd_sig = "Crossing Up"
        elif last["macd"] < last["macd_signal"] and last["macd_hist"] < 0:
            macd_sig = "Bearish"
        else:
            macd_sig = "Neutral"

        rsi_val = float(last["rsi"])
        rsi_signal = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Normal")

        def safe(val):
            v = float(val)
            return 0 if np.isnan(v) or np.isinf(v) else round(v, 2)

        result.update({
            "status": "ok",
            "ltp": round(ltp, 2),
            "score": round(score, 1),
            "recommendation": rec,
            "rec_color": rec_color,
            "entry": entry,
            "target": target,
            "stop_loss": stop_loss,
            "return_pct": return_pct,
            "rr": rr,
            "rsi": round(rsi_val, 1),
            "rsi_signal": rsi_signal,
            "macd_signal": macd_sig,
            "trend": trend,
            "patterns": patterns,
            "signals": signals,
            "horizon": horizon,
            "vol_ratio": safe(last["vol_ratio"]),
            "ema9":   safe(last["ema9"]),
            "ema21":  safe(last["ema21"]),
            "ema50":  safe(last["ema50"]),
            "ema200": safe(last["ema200"]),
            "atr": safe(last["atr"]),
            "momentum_5":  safe(last["momentum_5"]),
            "momentum_20": safe(last["momentum_20"]),
            "_df": df,
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="Kite Stock Analyzer", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    symbols: List[str]
    days: int = 120


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


@app.get("/api/status")
async def get_status():
    return {
        "kite_connected": KITE_CONNECTED,
        "user": KITE_USER,
        "kite_ok": KITE_OK,
        "plotly_ok": PLOTLY_OK,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/reconnect")
async def reconnect():
    global config
    config = load_config()
    KITE_CONNECTED = init_kite()
    return {"connected": KITE_CONNECTED, "user": KITE_USER}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    if not kite:
        raise HTTPException(503, "Kite not connected. Run: python get_token.py")

    symbols = [s.strip().upper() for s in req.symbols if s.strip()]
    if not symbols:
        raise HTTPException(400, "No symbols provided")

    results = []
    for symbol in symbols:
        r = fetch_and_analyze(symbol, req.days)
        r.pop("_df", None)
        results.append(r)

    # Sort: BUY first, then by score desc
    order = {"BUY": 0, "WATCH": 1, "AVOID": 2, "—": 3}
    results.sort(key=lambda x: (order.get(x["recommendation"], 3), -x["score"]))

    return {
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "user": KITE_USER,
    }


@app.get("/api/chart/{symbol}")
async def chart(symbol: str, days: int = 120):
    if not kite:
        raise HTTPException(503, "Kite not connected")

    r = fetch_and_analyze(symbol.upper(), days)
    df = r.pop("_df", None)

    if r["status"] == "error":
        raise HTTPException(400, r.get("error", "Analysis failed"))

    chart_data = generate_chart_json(symbol.upper(), df, r)
    return {"chart": chart_data, "analysis": r}


@app.get("/api/portfolio")
async def portfolio():
    if not kite:
        raise HTTPException(503, "Kite not connected. Run: python get_token.py")

    try:
        holdings = kite.holdings()
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch holdings: {e}")

    items = []
    total_invested = 0.0
    total_current  = 0.0
    total_day_pnl  = 0.0

    for h in holdings:
        qty = h.get("quantity", 0)
        if qty == 0:
            continue
        avg   = float(h.get("average_price", 0))
        ltp   = float(h.get("last_price", 0))
        close = float(h.get("close_price", 0))

        invested = avg * qty
        current  = ltp * qty
        pnl      = current - invested
        pnl_pct  = (pnl / invested * 100) if invested else 0
        day_pnl  = (ltp - close) * qty

        total_invested += invested
        total_current  += current
        total_day_pnl  += day_pnl

        items.append({
            "symbol":         h.get("tradingsymbol", ""),
            "exchange":       h.get("exchange", ""),
            "isin":           h.get("isin", ""),
            "quantity":       qty,
            "avg_price":      round(avg, 2),
            "ltp":            round(ltp, 2),
            "close_price":    round(close, 2),
            "invested":       round(invested, 2),
            "current":        round(current, 2),
            "pnl":            round(pnl, 2),
            "pnl_pct":        round(pnl_pct, 2),
            "day_change":     round(float(h.get("day_change", 0)), 2),
            "day_change_pct": round(float(h.get("day_change_percentage", 0)), 2),
            "day_pnl":        round(day_pnl, 2),
        })

    # Sort by absolute invested value descending
    items.sort(key=lambda x: x["invested"], reverse=True)

    total_pnl     = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0
    total_day_pct = (total_day_pnl / total_current * 100) if total_current else 0

    return {
        "holdings": items,
        "summary": {
            "total_invested":  round(total_invested, 2),
            "total_current":   round(total_current, 2),
            "total_pnl":       round(total_pnl, 2),
            "total_pnl_pct":   round(total_pnl_pct, 2),
            "total_day_pnl":   round(total_day_pnl, 2),
            "total_day_pct":   round(total_day_pct, 2),
            "count":           len(items),
        },
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# HTML DASHBOARD (embedded)
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kite Stock Analyzer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080d1a;color:#ccd6f6;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#0d1526}
::-webkit-scrollbar-thumb{background:#2a3a5c;border-radius:3px}

/* ── Header ── */
header{background:linear-gradient(135deg,#0d1526,#111827);border-bottom:1px solid #1e2d4a;padding:14px 32px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.logo{display:flex;align-items:center;gap:10px}
.logo h1{font-size:18px;font-weight:700;color:#fff;letter-spacing:-0.3px}
.logo p{font-size:11px;color:#8892b0;margin-top:1px}
.hdr-right{display:flex;align-items:center;gap:16px}
.status-badge{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;cursor:pointer}
.s-ok{background:rgba(0,230,118,0.12);color:#00e676;border:1px solid #00e67640}
.s-err{background:rgba(239,83,80,0.12);color:#ef5350;border:1px solid #ef535040}
.s-warn{background:rgba(255,214,0,0.12);color:#ffd600;border:1px solid #ffd60040}
#clock{font-size:12px;color:#8892b0;font-variant-numeric:tabular-nums}
#market-status{font-size:11px;padding:3px 8px;border-radius:10px;font-weight:600}
.mkt-open{background:rgba(0,230,118,0.15);color:#00e676}
.mkt-closed{background:rgba(239,83,80,0.12);color:#ef5350}

/* ── Container ── */
.container{max-width:1500px;margin:0 auto;padding:24px 28px}

/* ── Input Section ── */
.input-card{background:#0d1526;border:1px solid #1e2d4a;border-radius:12px;padding:22px;margin-bottom:20px}
.input-card h2{font-size:13px;font-weight:600;color:#8892b0;text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px}
.input-row{display:flex;gap:10px;flex-wrap:wrap}
.stock-input{flex:1;min-width:280px;background:#060b14;border:1px solid #1e2d4a;color:#ccd6f6;padding:10px 16px;border-radius:8px;font-size:14px;height:46px;transition:border .2s}
.stock-input:focus{outline:none;border-color:#64ffda;box-shadow:0 0 0 2px rgba(100,255,218,.08)}
.stock-input::placeholder{color:#3a4a6a}
.btn{padding:10px 22px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all .15s;white-space:nowrap}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-primary{background:#64ffda;color:#040810}
.btn-primary:not(:disabled):hover{background:#00e676;transform:translateY(-1px);box-shadow:0 4px 12px rgba(100,255,218,.25)}
.btn-ghost{background:transparent;color:#64ffda;border:1px solid #2a3a5a}
.btn-ghost:hover{background:rgba(100,255,218,.06);border-color:#64ffda}
.btn-sm{padding:5px 12px;font-size:12px;border-radius:6px}

.quick-row{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.ql{font-size:11px;color:#3a4a6a;margin-right:2px}
.qbtn{padding:4px 10px;border-radius:5px;font-size:11px;background:#0d1a30;color:#64ffda;border:1px solid #1e2d4a;cursor:pointer;transition:all .15s}
.qbtn:hover{background:#1a2a45;border-color:#64ffda}

/* ── Stats Row ── */
.stats-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.stat{background:#0d1526;border:1px solid #1e2d4a;border-radius:10px;padding:12px 18px;text-align:center;min-width:88px}
.stat .v{font-size:24px;font-weight:700;line-height:1.1}
.stat .l{font-size:10px;color:#8892b0;text-transform:uppercase;letter-spacing:.5px;margin-top:3px}
.c-buy{color:#00e676}.c-watch{color:#ffd600}.c-avoid{color:#ef5350}.c-info{color:#64ffda}
.stats-actions{flex:1;display:flex;justify-content:flex-end;align-items:center;gap:8px}

/* ── Table ── */
.tbl-wrap{border-radius:12px;border:1px solid #1e2d4a;overflow:hidden;overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:1100px}
thead th{background:#0b1220;padding:10px 14px;text-align:left;font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.4px;white-space:nowrap;border-bottom:1px solid #1e2d4a}
thead th:first-child{position:sticky;left:0;background:#0b1220;z-index:2}
tbody tr{border-top:1px solid #0f1d33;cursor:pointer;transition:background .12s}
tbody tr:hover{background:rgba(100,255,218,.03)}
tbody td{padding:11px 14px;font-size:13px;vertical-align:middle;white-space:nowrap}
tbody td:first-child{position:sticky;left:0;background:#080d1a;z-index:1}
tbody tr:hover td:first-child{background:#0a1224}

.sym{font-weight:700;color:#e2e8f0;font-size:14px}
.sym-arrow{color:#3a4a6a;font-size:10px;margin-left:4px;transition:transform .2s}
.expanded .sym-arrow{transform:rotate(180deg)}

.score-pill{display:inline-flex;align-items:center;justify-content:center;width:42px;height:26px;border-radius:6px;font-weight:700;font-size:12px}
.sp-hi{background:rgba(0,230,118,.14);color:#00e676;border:1px solid #00e67630}
.sp-md{background:rgba(255,214,0,.14);color:#ffd600;border:1px solid #ffd60030}
.sp-lo{background:rgba(239,83,80,.14);color:#ef5350;border:1px solid #ef535030}

.rec-pill{padding:3px 9px;border-radius:10px;font-size:10px;font-weight:700;letter-spacing:.4px;text-transform:uppercase}
.rp-buy{background:rgba(0,230,118,.18);color:#00e676;border:1px solid #00e67650}
.rp-watch{background:rgba(255,214,0,.18);color:#ffd600;border:1px solid #ffd60050}
.rp-avoid{background:rgba(239,83,80,.18);color:#ef5350;border:1px solid #ef535050}
.rp-na{background:rgba(100,100,100,.15);color:#888;border:1px solid #44444440}

.rsi-wrap{display:flex;align-items:center;gap:7px}
.rsi-bar{width:52px;height:5px;border-radius:3px;background:#1e2d4a;overflow:hidden}
.rsi-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#26a69a,#64ffda)}
.rsi-fill.rb-high{background:linear-gradient(90deg,#ffd600,#ef5350)}
.rsi-fill.rb-low{background:linear-gradient(90deg,#5c6bc0,#26a69a)}

.c-up{color:#26a69a}.c-dn{color:#ef5350}.c-sd{color:#64748b}
.c-pos{color:#26a69a}.c-neg{color:#ef5350}
.c-vol-hi{color:#ffd600}.c-vol-lo{color:#64748b}

.chart-btn{padding:4px 10px;border-radius:5px;font-size:11px;background:#0d1a30;color:#64ffda;border:1px solid #1e2d4a;cursor:pointer;transition:all .15s}
.chart-btn:hover{background:#1a2a45;border-color:#64ffda}

/* ── Detail row ── */
.detail-row > td{padding:0 !important;background:#060c1a !important;border:none}
.detail-inner{display:none;padding:16px 20px;border-top:1px solid #0f1d33}
.detail-inner.open{display:block}
.det-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-bottom:14px}
.det-item .dl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-bottom:3px}
.det-item .dv{font-size:13px;color:#ccd6f6;font-weight:500}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.tag{padding:3px 8px;border-radius:4px;font-size:11px;background:rgba(100,255,218,.08);color:#64ffda;border:1px solid rgba(100,255,218,.2)}
.tag.tag-warn{background:rgba(255,214,0,.08);color:#ffd600;border-color:rgba(255,214,0,.2)}
.tag.tag-neg{background:rgba(239,83,80,.08);color:#ef5350;border-color:rgba(239,83,80,.2)}
.sig-list{margin-top:8px}
.sig-item{font-size:12px;color:#94a3b8;padding:2px 0}
.sig-item::before{content:"+ ";color:#26a69a;font-weight:700}

/* ── Loading ── */
.loading-box{text-align:center;padding:60px 20px}
.spinner{display:inline-block;width:36px;height:36px;border:3px solid #1e2d4a;border-top-color:#64ffda;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-box p{margin-top:14px;color:#64748b;font-size:14px}
.loading-sym{font-size:11px;color:#3a4a6a;margin-top:6px}

/* ── Empty state ── */
.empty{text-align:center;padding:80px 20px}
.empty .icon{font-size:44px;margin-bottom:16px;opacity:.6}
.empty h3{color:#ccd6f6;font-size:18px;margin-bottom:8px}
.empty p{color:#64748b;font-size:14px}

/* ── Modal ── */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:500;align-items:center;justify-content:center}
.modal-bg.open{display:flex}
.modal{background:#0d1526;border:1px solid #1e2d4a;border-radius:14px;width:96vw;max-width:1300px;max-height:96vh;display:flex;flex-direction:column;overflow:hidden}
.modal-hdr{padding:14px 18px;border-bottom:1px solid #1e2d4a;display:flex;justify-content:space-between;align-items:center}
.modal-hdr h3{color:#ccd6f6;font-size:15px;font-weight:600}
.modal-close{cursor:pointer;color:#64748b;background:none;border:none;font-size:18px;padding:2px 6px;border-radius:4px}
.modal-close:hover{color:#ef5350;background:rgba(239,83,80,.1)}
.modal-body{flex:1;overflow:auto}
#chart-box{width:100%;min-height:600px}

/* ── Error cell ── */
.err-cell{color:#ef5350;font-size:12px}
.timestamp{font-size:11px;color:#3a4a6a;margin-top:8px;text-align:right}

/* ── Tabs ── */
.tab-bar{display:flex;gap:4px;margin-bottom:20px;background:#0d1526;border:1px solid #1e2d4a;border-radius:10px;padding:4px;width:fit-content}
.tab-btn{padding:8px 22px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all .15s;background:transparent;color:#64748b}
.tab-btn.active{background:#1a2a45;color:#64ffda;box-shadow:0 1px 4px rgba(0,0,0,.3)}
.tab-btn:not(.active):hover{color:#ccd6f6;background:rgba(255,255,255,.03)}
.tab-content{display:none}
.tab-content.active{display:block}

/* ── Portfolio ── */
.pf-summary{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}
.pf-card{background:#0d1526;border:1px solid #1e2d4a;border-radius:10px;padding:16px 22px;min-width:140px;flex:1}
.pf-card .pf-label{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.pf-card .pf-val{font-size:22px;font-weight:700;line-height:1.1}
.pf-card .pf-sub{font-size:12px;margin-top:3px}

.pf-tbl-wrap{border-radius:12px;border:1px solid #1e2d4a;overflow:hidden;overflow-x:auto}
.pf-tbl{width:100%;border-collapse:collapse;min-width:900px}
.pf-tbl thead th{background:#0b1220;padding:10px 14px;text-align:left;font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.4px;white-space:nowrap;border-bottom:1px solid #1e2d4a}
.pf-tbl tbody tr{border-top:1px solid #0f1d33;transition:background .12s}
.pf-tbl tbody tr:hover{background:rgba(100,255,218,.03)}
.pf-tbl tbody td{padding:11px 14px;font-size:13px;white-space:nowrap}

.pf-pnl-bar{width:80px;height:6px;border-radius:3px;background:#1e2d4a;overflow:hidden;display:inline-block;vertical-align:middle;margin-left:6px}
.pf-pnl-fill{height:100%;border-radius:3px}

.pf-actions{display:flex;gap:8px;margin-top:14px;align-items:center}

@media(max-width:768px){header{padding:10px 14px}.container{padding:14px}.pf-summary{flex-direction:column}}
</style>
</head>
<body>

<header>
  <div class="logo">
    <span style="font-size:26px">📈</span>
    <div>
      <h1>Kite Stock Analyzer</h1>
      <p>Real-time NSE technical analysis &amp; swing trade recommendations</p>
    </div>
  </div>
  <div class="hdr-right">
    <span id="market-badge" class="market-status">—</span>
    <span id="status-badge" class="status-badge s-warn" onclick="checkStatus()">● Checking...</span>
    <span id="clock"></span>
  </div>
</header>

<div class="container">

  <!-- Tab Bar -->
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('analyzer')">⚡ Analyzer</button>
    <button class="tab-btn" onclick="switchTab('portfolio')">💼 My Portfolio</button>
  </div>

  <!-- ═══ TAB: ANALYZER ═══ -->
  <div id="tab-analyzer" class="tab-content active">

  <!-- Input -->
  <div class="input-card">
    <h2>Analyze Stocks</h2>
    <div class="input-row">
      <input id="stock-input" class="stock-input"
        placeholder="RELIANCE, TCS, JIOFIN, TEJASNET, BAJFINANCE ..."
        onkeydown="if(event.key==='Enter')analyzeStocks()" />
      <button class="btn btn-primary" id="analyze-btn" onclick="analyzeStocks()">
        ⚡&nbsp;Analyze
      </button>
      <button class="btn btn-ghost" onclick="clearAll()">✕ Clear</button>
    </div>
    <div class="quick-row">
      <span class="ql">Quick:</span>
      <button class="qbtn" onclick="setStocks('RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK')">Nifty Top 5</button>
      <button class="qbtn" onclick="setStocks('INFY,TCS,WIPRO,HCLTECH,TECHM')">IT</button>
      <button class="qbtn" onclick="setStocks('SBIN,HDFCBANK,ICICIBANK,AXISBANK,KOTAKBANK')">Banking</button>
      <button class="qbtn" onclick="setStocks('SUNPHARMA,DRREDDY,CIPLA,DIVISLAB,APOLLOHOSP')">Pharma</button>
      <button class="qbtn" onclick="setStocks('RELIANCE,ADANIENT,LT,NTPC,POWERGRID,ONGC')">Energy</button>
      <button class="qbtn" onclick="setStocks('JIOFIN,BAJFINANCE,BAJAJFINSV,SHRIRAMFIN,CHOLAFIN')">NBFC</button>
      <button class="qbtn" onclick="setStocks('TEJASNET,IRCTC,ZOMATO,TATAPOWER,ADANIPORTS')">Momentum</button>
      <button class="qbtn" onclick="setStocks('TITAN,MARUTI,ASIANPAINT,NESTLEIND,HINDUNILVR')">Consumer</button>
    </div>
  </div>

  <!-- Stats -->
  <div id="stats-row" class="stats-row" style="display:none">
    <div class="stat"><div class="v c-info" id="st-total">0</div><div class="l">Analyzed</div></div>
    <div class="stat"><div class="v c-buy"  id="st-buy">0</div><div class="l">BUY</div></div>
    <div class="stat"><div class="v c-watch" id="st-watch">0</div><div class="l">WATCH</div></div>
    <div class="stat"><div class="v c-avoid" id="st-avoid">0</div><div class="l">AVOID</div></div>
    <div class="stat"><div class="v c-info" id="st-avg">0</div><div class="l">Avg Score</div></div>
    <div class="stats-actions">
      <button class="btn btn-ghost btn-sm" onclick="sortResults('score')">Sort Score</button>
      <button class="btn btn-ghost btn-sm" onclick="sortResults('rsi')">Sort RSI</button>
      <button class="btn btn-ghost btn-sm" onclick="sortResults('return_pct')">Sort Return</button>
      <button class="btn btn-ghost btn-sm" onclick="exportCSV()">⬇ CSV</button>
    </div>
  </div>

  <!-- Results -->
  <div id="results-wrap" style="display:none">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Symbol</th><th>Score</th><th>Rec</th>
            <th>LTP ₹</th><th>Entry ₹</th><th>Target ₹</th><th>Stop Loss ₹</th>
            <th>Return %</th><th>R:R</th><th>RSI</th>
            <th>MACD</th><th>Trend</th><th>Horizon</th><th>Vol Ratio</th>
            <th>Chart</th>
          </tr>
        </thead>
        <tbody id="tbl-body"></tbody>
      </table>
    </div>
    <div id="ts-row" class="timestamp"></div>
  </div>

  <!-- Loading -->
  <div id="loading" style="display:none" class="loading-box">
    <div class="spinner"></div>
    <p id="load-text">Fetching data and computing indicators...</p>
    <div id="load-sym" class="loading-sym"></div>
  </div>

  <!-- Empty state -->
  <div id="empty-state" class="empty">
    <div class="icon">📊</div>
    <h3>Ready to Analyze</h3>
    <p>Enter stock symbols above and click Analyze to get real-time technical analysis with buy/sell recommendations</p>
  </div>

  </div><!-- /tab-analyzer -->

  <!-- ═══ TAB: PORTFOLIO ═══ -->
  <div id="tab-portfolio" class="tab-content">

    <!-- Summary Cards -->
    <div class="pf-summary" id="pf-summary" style="display:none">
      <div class="pf-card">
        <div class="pf-label">Invested Value</div>
        <div class="pf-val" id="pf-invested" style="color:#ccd6f6">—</div>
      </div>
      <div class="pf-card">
        <div class="pf-label">Current Value</div>
        <div class="pf-val" id="pf-current" style="color:#ccd6f6">—</div>
      </div>
      <div class="pf-card">
        <div class="pf-label">Total P&amp;L</div>
        <div class="pf-val" id="pf-pnl">—</div>
        <div class="pf-sub" id="pf-pnl-pct"></div>
      </div>
      <div class="pf-card">
        <div class="pf-label">Today's Change</div>
        <div class="pf-val" id="pf-day">—</div>
        <div class="pf-sub" id="pf-day-pct"></div>
      </div>
      <div class="pf-card">
        <div class="pf-label">Holdings</div>
        <div class="pf-val c-info" id="pf-count">0</div>
      </div>
    </div>

    <!-- Holdings Table -->
    <div id="pf-table-wrap" style="display:none">
      <div class="pf-tbl-wrap">
        <table class="pf-tbl">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Exchange</th>
              <th>Qty</th>
              <th>Avg Price ₹</th>
              <th>LTP ₹</th>
              <th>Invested ₹</th>
              <th>Current ₹</th>
              <th>P&amp;L ₹</th>
              <th>P&amp;L %</th>
              <th>Day Change</th>
              <th>Day P&amp;L ₹</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="pf-body"></tbody>
        </table>
      </div>
      <div class="pf-actions">
        <button class="btn btn-primary btn-sm" onclick="analyzeHoldings()">⚡ Analyze All Holdings</button>
        <button class="btn btn-ghost btn-sm" onclick="loadPortfolio()">↻ Refresh</button>
        <div class="timestamp" id="pf-ts" style="flex:1"></div>
      </div>
    </div>

    <!-- Portfolio Loading -->
    <div id="pf-loading" class="loading-box" style="display:none">
      <div class="spinner"></div>
      <p>Fetching your holdings...</p>
    </div>

    <!-- Portfolio Empty -->
    <div id="pf-empty" class="empty">
      <div class="icon">💼</div>
      <h3>My Portfolio</h3>
      <p>Click the button below to fetch your Zerodha holdings</p>
      <button class="btn btn-primary" style="margin-top:20px" onclick="loadPortfolio()">Load Portfolio</button>
    </div>

  </div><!-- /tab-portfolio -->

</div><!-- container -->

<!-- Chart Modal -->
<div class="modal-bg" id="chart-modal" onclick="if(event.target===this)closeChart()">
  <div class="modal">
    <div class="modal-hdr">
      <h3 id="chart-title">Chart</h3>
      <div style="display:flex;gap:10px;align-items:center">
        <span id="chart-rec" class="rec-pill rp-na" style="font-size:12px">—</span>
        <button class="modal-close" onclick="closeChart()">✕</button>
      </div>
    </div>
    <div class="modal-body">
      <div id="chart-box">
        <div class="loading-box"><div class="spinner"></div><p>Loading chart...</p></div>
      </div>
    </div>
  </div>
</div>

<script>
// ── Globals ─────────────────────────────────────────────────────────────────
let allData = [];
let sortField = 'score', sortDir = -1;

// ── Clock & market hours ─────────────────────────────────────────────────────
function tick() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toLocaleString('en-IN', {
    timeZone:'Asia/Kolkata', hour:'2-digit', minute:'2-digit', second:'2-digit',
    day:'2-digit', month:'short'
  });
  const ist = new Date(now.toLocaleString('en-US',{timeZone:'Asia/Kolkata'}));
  const h=ist.getHours(), m=ist.getMinutes(), day=ist.getDay();
  const mins = h*60+m;
  const isWeekday = day>=1 && day<=5;
  const isMarket = isWeekday && mins>=555 && mins<930; // 9:15-15:30
  const badge = document.getElementById('market-badge');
  if(isMarket){ badge.textContent='● Market Open'; badge.className='market-status mkt-open'; }
  else { badge.textContent='● Market Closed'; badge.className='market-status mkt-closed'; }
}
setInterval(tick,1000); tick();

// ── Status check ─────────────────────────────────────────────────────────────
async function checkStatus(){
  const badge = document.getElementById('status-badge');
  try{
    const d = await fetch('/api/status').then(r=>r.json());
    if(d.kite_connected){
      badge.textContent='● '+d.user; badge.className='status-badge s-ok';
    } else {
      badge.textContent='● Disconnected'; badge.className='status-badge s-err';
    }
  } catch(e){
    badge.textContent='● Offline'; badge.className='status-badge s-err';
  }
}
checkStatus();

// ── Helpers ──────────────────────────────────────────────────────────────────
function fmt(n){ return typeof n==='number'? n.toLocaleString('en-IN',{maximumFractionDigits:2}) : n; }
function pct(n){ const c=n>0?'c-pos':n<0?'c-neg':''; return '<span class="'+c+'">'+(n>0?'+':'')+n+'%</span>'; }

function setStocks(s){ document.getElementById('stock-input').value=s; }

function clearAll(){
  document.getElementById('stock-input').value='';
  document.getElementById('results-wrap').style.display='none';
  document.getElementById('stats-row').style.display='none';
  document.getElementById('empty-state').style.display='block';
  document.getElementById('tbl-body').innerHTML='';
  allData=[];
}

// ── Analyze ──────────────────────────────────────────────────────────────────
async function analyzeStocks(){
  const raw = document.getElementById('stock-input').value;
  const symbols = raw.split(/[,\\s]+/).map(s=>s.trim().toUpperCase()).filter(Boolean);
  if(!symbols.length){ alert('Enter at least one stock symbol'); return; }

  document.getElementById('empty-state').style.display='none';
  document.getElementById('results-wrap').style.display='none';
  document.getElementById('stats-row').style.display='none';
  document.getElementById('loading').style.display='block';
  document.getElementById('load-text').textContent=
    'Analyzing '+symbols.length+' stock'+(symbols.length>1?'s':'')+'...';
  document.getElementById('load-sym').textContent='Fetching data, computing indicators...';
  document.getElementById('analyze-btn').disabled=true;

  try{
    const resp = await fetch('/api/analyze',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({symbols, days:120})
    });
    if(!resp.ok){ const e=await resp.json(); throw new Error(e.detail||'Server error'); }
    const data = await resp.json();
    allData = data.results;
    renderAll(data);
  } catch(e){
    alert('Error: '+e.message);
    document.getElementById('empty-state').style.display='block';
  } finally{
    document.getElementById('loading').style.display='none';
    document.getElementById('analyze-btn').disabled=false;
  }
}

// ── Render ───────────────────────────────────────────────────────────────────
function renderAll(data){
  const r=data.results;
  const ok=r.filter(x=>x.status==='ok');
  document.getElementById('st-total').textContent=r.length;
  document.getElementById('st-buy').textContent=ok.filter(x=>x.recommendation==='BUY').length;
  document.getElementById('st-watch').textContent=ok.filter(x=>x.recommendation==='WATCH').length;
  document.getElementById('st-avoid').textContent=ok.filter(x=>x.recommendation==='AVOID').length;
  const avg=ok.length?Math.round(ok.reduce((s,x)=>s+x.score,0)/ok.length):0;
  document.getElementById('st-avg').textContent=avg;
  document.getElementById('stats-row').style.display='flex';
  renderBody(allData);
  document.getElementById('results-wrap').style.display='block';
  document.getElementById('ts-row').textContent=
    'Last updated: '+new Date(data.timestamp).toLocaleString('en-IN',{timeZone:'Asia/Kolkata'})
    +' | User: '+data.user;
}

function sortResults(field){
  if(sortField===field) sortDir*=-1; else { sortField=field; sortDir=-1; }
  const sorted=[...allData].sort((a,b)=>sortDir*((b[field]||0)-(a[field]||0)));
  renderBody(sorted);
}

function renderBody(results){
  const tbody=document.getElementById('tbl-body');
  tbody.innerHTML='';
  results.forEach((r,idx)=>{
    const main=document.createElement('tr');
    main.id='row-'+idx;
    main.onclick=()=>toggleDetail(idx);

    if(r.status==='error'){
      main.innerHTML='<td class="sym">'+r.symbol+'</td>'
        +'<td colspan="14" class="err-cell">⚠ '+(r.error||'Error')+'</td>';
      tbody.appendChild(main);
      return;
    }

    const sc=r.score>=65?'sp-hi':r.score>=45?'sp-md':'sp-lo';
    const rc=r.recommendation==='BUY'?'rp-buy':r.recommendation==='WATCH'?'rp-watch':'rp-avoid';
    const tr=r.trend.includes('Up')?'c-up':r.trend.includes('Down')?'c-dn':'c-sd';
    const rsiC=r.rsi>70?'rb-high':r.rsi<30?'rb-low':'';
    const rsiW=Math.min(r.rsi,100);
    const mC=r.macd_signal==='Bullish'?'c-up':r.macd_signal.includes('Bearish')?'c-dn':'c-sd';
    const mI=r.macd_signal==='Bullish'?'▲':r.macd_signal.includes('Bearish')?'▼':'—';
    const retC=r.return_pct>=5?'c-pos':r.return_pct>=3?'c-watch':'c-neg';
    const rrC=r.rr>=1.5?'c-pos':'c-sd';
    const volC=r.vol_ratio>2?'c-vol-hi':r.vol_ratio>1.5?'c-watch':'c-vol-lo';

    main.innerHTML=
      '<td><span class="sym">'+r.symbol+' <span class="sym-arrow">▾</span></span></td>'
      +'<td><span class="score-pill '+sc+'">'+r.score+'</span></td>'
      +'<td><span class="rec-pill '+rc+'">'+r.recommendation+'</span></td>'
      +'<td style="font-weight:600;color:#e2e8f0">₹'+fmt(r.ltp)+'</td>'
      +'<td>₹'+fmt(r.entry)+'</td>'
      +'<td class="c-pos" style="font-weight:600">₹'+fmt(r.target)+'</td>'
      +'<td class="c-neg">₹'+fmt(r.stop_loss)+'</td>'
      +'<td class="'+retC+'" style="font-weight:600">'+(r.return_pct>0?'+':'')+r.return_pct+'%</td>'
      +'<td class="'+rrC+'">'+r.rr+'x</td>'
      +'<td><div class="rsi-wrap"><div class="rsi-bar"><div class="rsi-fill '+rsiC+'" style="width:'+rsiW+'%"></div></div><span>'+r.rsi+'</span></div></td>'
      +'<td class="'+mC+'">'+mI+' '+r.macd_signal+'</td>'
      +'<td class="'+tr+'">'+r.trend+'</td>'
      +'<td style="color:#64748b;font-size:12px">'+r.horizon+'</td>'
      +'<td class="'+volC+'" style="font-size:12px">'+r.vol_ratio+'x</td>'
      +'<td><button class="chart-btn" onclick="openChart(event,\''+r.symbol+'\','+JSON.stringify(r).replace(/'/g,"\\'")+')">📊 Chart</button></td>';

    // Detail row
    const det=document.createElement('tr');
    det.className='detail-row';
    det.id='det-'+idx;

    const ptags=r.patterns.map(p=>{
      const cls=p.includes('Death')||p.includes('Down')||p.includes('Bear')||p.includes('Caution')?'tag-neg':'';
      return '<span class="tag '+cls+'">'+p+'</span>';
    }).join('');

    const sigs=r.signals.map(s=>'<div class="sig-item">'+s+'</div>').join('');

    const m5c=r.momentum_5>0?'c-pos':'c-neg', m20c=r.momentum_20>0?'c-pos':'c-neg';

    det.innerHTML='<td colspan="15"><div class="detail-inner" id="di-'+idx+'">'
      +'<div class="det-grid">'
      +'<div class="det-item"><div class="dl">EMA 9 / 21 / 50 / 200</div>'
      +'<div class="dv">₹'+r.ema9+' / ₹'+r.ema21+' / ₹'+r.ema50+' / ₹'+r.ema200+'</div></div>'
      +'<div class="det-item"><div class="dl">ATR (14)</div>'
      +'<div class="dv">₹'+r.atr+' ('+(r.ltp?((r.atr/r.ltp)*100).toFixed(1):'?')+'% of price)</div></div>'
      +'<div class="det-item"><div class="dl">5d / 20d Momentum</div>'
      +'<div class="dv"><span class="'+m5c+'">'+(r.momentum_5>0?'+':'')+r.momentum_5+'%</span>'
      +' &nbsp;/&nbsp; <span class="'+m20c+'">'+(r.momentum_20>0?'+':'')+r.momentum_20+'%</span></div></div>'
      +'<div class="det-item"><div class="dl">RSI Signal</div>'
      +'<div class="dv">'+r.rsi_signal+' ('+r.rsi+')</div></div>'
      +'</div>'
      +(ptags?'<div class="dl" style="margin-bottom:6px">Detected Patterns</div><div class="tags">'+ptags+'</div>':'')
      +(sigs?'<div class="dl" style="margin:12px 0 6px">Bullish Signals</div><div class="sig-list">'+sigs+'</div>':'')
      +'</div></td>';

    tbody.appendChild(main);
    tbody.appendChild(det);
  });
}

function toggleDetail(idx){
  const el=document.getElementById('di-'+idx);
  if(!el) return;
  el.classList.toggle('open');
  const row=document.getElementById('row-'+idx);
  if(row) row.classList.toggle('expanded');
}

// ── CSV Export ───────────────────────────────────────────────────────────────
function exportCSV(){
  if(!allData.length) return;
  const hdr=['Symbol','Score','Rec','LTP','Entry','Target','SL','Return%','RR','RSI','MACD','Trend','Horizon','VolRatio','Patterns'];
  const rows=allData.map(r=>[
    r.symbol,r.score,r.recommendation,r.ltp,r.entry,r.target,
    r.stop_loss,r.return_pct,r.rr,r.rsi,r.macd_signal,r.trend,
    r.horizon,r.vol_ratio,'"'+r.patterns.join('; ')+'"'
  ]);
  const csv=[hdr,...rows].map(r=>r.join(',')).join('\\n');
  const a=document.createElement('a');
  a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download='kite_analysis_'+new Date().toISOString().slice(0,16).replace('T','_')+'.csv';
  a.click();
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function openChart(e, symbol, rec){
  e.stopPropagation();
  const modal=document.getElementById('chart-modal');
  modal.classList.add('open');
  document.getElementById('chart-title').textContent=symbol+' — Technical Analysis';
  const recBadge=document.getElementById('chart-rec');
  const rc=rec&&rec.recommendation==='BUY'?'rp-buy':rec&&rec.recommendation==='WATCH'?'rp-watch':'rp-avoid';
  recBadge.textContent=rec?rec.recommendation:'—';
  recBadge.className='rec-pill '+rc;
  document.getElementById('chart-box').innerHTML=
    '<div class="loading-box" style="padding:100px 0"><div class="spinner"></div><p>Loading chart for '+symbol+'...</p></div>';

  fetch('/api/chart/'+symbol)
    .then(r=>{ if(!r.ok) throw new Error('Chart failed'); return r.json(); })
    .then(data=>{
      document.getElementById('chart-box').innerHTML='<div id="plotly-div" style="width:100%;height:880px"></div>';
      Plotly.newPlot('plotly-div', data.chart.data, data.chart.layout, {responsive:true, displayModeBar:true});
    })
    .catch(err=>{
      document.getElementById('chart-box').innerHTML=
        '<div class="empty" style="padding:80px 0"><div class="icon">⚠️</div><h3>Chart Error</h3><p>'+err.message+'</p></div>';
    });
}

function closeChart(){
  document.getElementById('chart-modal').classList.remove('open');
  document.getElementById('chart-box').innerHTML='';
}

document.addEventListener('keydown', e=>{ if(e.key==='Escape') closeChart(); });

// ── Tab Switching ────────────────────────────────────────────────────────────
function switchTab(name){
  document.querySelectorAll('.tab-content').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el=>el.classList.remove('active'));
  const tab=document.getElementById('tab-'+name);
  if(tab) tab.classList.add('active');
  // Activate the correct button
  const btns=document.querySelectorAll('.tab-btn');
  btns.forEach(b=>{ if((name==='analyzer'&&b.textContent.includes('Analyzer'))||(name==='portfolio'&&b.textContent.includes('Portfolio'))) b.classList.add('active'); });
  // Auto-load portfolio on first visit
  if(name==='portfolio' && !portfolioLoaded) loadPortfolio();
}

// ── Portfolio ────────────────────────────────────────────────────────────────
let portfolioLoaded = false;
let portfolioData = [];

async function loadPortfolio(){
  document.getElementById('pf-empty').style.display='none';
  document.getElementById('pf-summary').style.display='none';
  document.getElementById('pf-table-wrap').style.display='none';
  document.getElementById('pf-loading').style.display='block';

  try{
    const resp = await fetch('/api/portfolio');
    if(!resp.ok){ const e=await resp.json(); throw new Error(e.detail||'Server error'); }
    const data = await resp.json();
    portfolioLoaded = true;
    portfolioData = data.holdings;
    renderPortfolio(data);
  } catch(e){
    alert('Portfolio error: '+e.message);
    document.getElementById('pf-empty').style.display='block';
  } finally{
    document.getElementById('pf-loading').style.display='none';
  }
}

function renderPortfolio(data){
  const s = data.summary;
  const h = data.holdings;

  // Summary cards
  document.getElementById('pf-invested').textContent = '₹'+fmtLakh(s.total_invested);
  document.getElementById('pf-current').textContent  = '₹'+fmtLakh(s.total_current);

  const pnlEl = document.getElementById('pf-pnl');
  pnlEl.textContent = (s.total_pnl>=0?'+':'')+fmtLakh(s.total_pnl);
  pnlEl.style.color = s.total_pnl>=0?'#00e676':'#ef5350';
  document.getElementById('pf-pnl-pct').innerHTML =
    '<span style="color:'+(s.total_pnl_pct>=0?'#00e676':'#ef5350')+'">'
    +(s.total_pnl_pct>=0?'+':'')+s.total_pnl_pct.toFixed(2)+'%</span>';

  const dayEl = document.getElementById('pf-day');
  dayEl.textContent = (s.total_day_pnl>=0?'+':'')+'₹'+fmt(Math.abs(s.total_day_pnl));
  dayEl.style.color = s.total_day_pnl>=0?'#00e676':'#ef5350';
  document.getElementById('pf-day-pct').innerHTML =
    '<span style="color:'+(s.total_day_pct>=0?'#00e676':'#ef5350')+'">'
    +(s.total_day_pct>=0?'+':'')+s.total_day_pct.toFixed(2)+'%</span>';

  document.getElementById('pf-count').textContent = s.count;
  document.getElementById('pf-summary').style.display='flex';

  // Table
  const tbody = document.getElementById('pf-body');
  tbody.innerHTML='';

  h.forEach(r=>{
    const pnlC = r.pnl>=0?'c-pos':'c-neg';
    const dayC = r.day_change_pct>=0?'c-pos':'c-neg';
    const pnlPctAbs = Math.min(Math.abs(r.pnl_pct), 100);
    const barColor = r.pnl>=0?'#00e676':'#ef5350';

    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td style="font-weight:700;color:#e2e8f0">'+r.symbol+'</td>'
      +'<td style="color:#64748b;font-size:12px">'+r.exchange+'</td>'
      +'<td>'+r.quantity+'</td>'
      +'<td>₹'+fmt(r.avg_price)+'</td>'
      +'<td style="font-weight:600;color:#ccd6f6">₹'+fmt(r.ltp)+'</td>'
      +'<td>₹'+fmt(r.invested)+'</td>'
      +'<td>₹'+fmt(r.current)+'</td>'
      +'<td class="'+pnlC+'" style="font-weight:600">'
        +(r.pnl>=0?'+':'')+fmt(r.pnl)
        +'<div class="pf-pnl-bar"><div class="pf-pnl-fill" style="width:'+pnlPctAbs+'%;background:'+barColor+'"></div></div>'
      +'</td>'
      +'<td class="'+pnlC+'" style="font-weight:600">'+(r.pnl_pct>=0?'+':'')+r.pnl_pct+'%</td>'
      +'<td class="'+dayC+'">'+(r.day_change_pct>=0?'+':'')+r.day_change_pct.toFixed(2)+'%</td>'
      +'<td class="'+dayC+'">'+(r.day_pnl>=0?'+':'')+'₹'+fmt(Math.abs(r.day_pnl))+'</td>'
      +'<td><button class="chart-btn" onclick="analyzeOne(\''+r.symbol+'\')">⚡ Analyze</button></td>';
    tbody.appendChild(tr);
  });

  document.getElementById('pf-table-wrap').style.display='block';
  document.getElementById('pf-ts').textContent =
    'Updated: '+new Date(data.timestamp).toLocaleString('en-IN',{timeZone:'Asia/Kolkata'});
}

function fmtLakh(n){
  const abs=Math.abs(n);
  if(abs>=10000000) return (n/10000000).toFixed(2)+' Cr';
  if(abs>=100000) return (n/100000).toFixed(2)+' L';
  return fmt(n);
}

function analyzeOne(symbol){
  switchTab('analyzer');
  document.getElementById('stock-input').value=symbol;
  analyzeStocks();
}

function analyzeHoldings(){
  if(!portfolioData.length){ alert('Load portfolio first'); return; }
  const symbols = portfolioData.map(h=>h.symbol).join(',');
  switchTab('analyzer');
  document.getElementById('stock-input').value=symbols;
  analyzeStocks();
}
</script>
</body>
</html>"""


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if not FASTAPI_OK:
        print("Install dependencies: pip install -r requirements.txt")
        sys.exit(1)

    print("\n" + "="*55)
    print("  Kite Stock Analyzer")
    print("="*55)
    print(f"  Kite Connected : {'YES — ' + KITE_USER if KITE_CONNECTED else 'NO — run python get_token.py'}")
    print(f"  Plotly Charts  : {'YES' if PLOTLY_OK else 'NO — pip install plotly'}")
    print(f"  URL            : http://localhost:8000")
    print("="*55 + "\n")

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
