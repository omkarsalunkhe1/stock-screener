#!/usr/bin/env python3
"""
Stock Analysis Tool — Kite API v3
===================================
Real-time technical analysis with interactive HTML dashboard.

Usage:
    python stock_scanner.py                           # interactive prompt
    python stock_scanner.py RELIANCE INFY TEJASNET    # direct symbols
    python stock_scanner.py --setup                   # configure API key
    python stock_scanner.py --portfolio               # analyze your holdings
"""

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

CONFIG_FILE = Path(__file__).parent / "kite_config.json"
REPORT_FILE = Path(__file__).parent / "analysis_report.html"

# ════════════════════════════════════════════════════════════════════════════════
# AUTH
# ════════════════════════════════════════════════════════════════════════════════

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return None

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def setup_credentials():
    print("\n" + "="*50)
    print("  KITE API SETUP")
    print("="*50)
    print("Get credentials from: https://developers.kite.trade/")
    print()
    api_key = input("API Key     : ").strip()
    access_token = input("Access Token: ").strip()
    cfg = {"api_key": api_key, "access_token": access_token}
    save_config(cfg)
    print(f"✓ Saved to {CONFIG_FILE}")
    return cfg

def get_kite():
    if not KITE_AVAILABLE:
        print("ERROR: Run:  pip install kiteconnect")
        sys.exit(1)
    cfg = load_config()
    if not cfg:
        print("No config found.")
        cfg = setup_credentials()
    kite = KiteConnect(api_key=cfg["api_key"])
    kite.set_access_token(cfg["access_token"])
    try:
        p = kite.profile()
        print(f"✓ Logged in as {p['user_name']} ({p['user_id']})")
    except Exception as e:
        print(f"✗ Auth failed: {e}")
        print("Run:  python stock_scanner.py --setup")
        sys.exit(1)
    return kite

# ════════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ════════════════════════════════════════════════════════════════════════════════

def fetch_stock(kite, symbol):
    print(f"  Fetching {symbol}...", end="", flush=True)
    try:
        # Get LTP + OHLC for today
        inst_key = f"NSE:{symbol}"
        ltp_data  = kite.ltp(inst_key)
        ohlc_data = kite.ohlc(inst_key)

        if inst_key not in ltp_data:
            print(f" ✗ Not found")
            return None

        ltp       = ltp_data[inst_key]["last_price"]
        token     = ltp_data[inst_key]["instrument_token"]
        prev_cls  = ohlc_data[inst_key]["ohlc"]["close"]
        today_o   = ohlc_data[inst_key]["ohlc"]["open"]
        today_h   = ohlc_data[inst_key]["ohlc"]["high"]
        today_l   = ohlc_data[inst_key]["ohlc"]["low"]
        chg_pct   = ((ltp - prev_cls) / prev_cls * 100) if prev_cls else 0

        # Historical data — 120 days (extra for 200 EMA warm-up)
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=150)

        records = kite.historical_data(
            instrument_token=token,
            from_date=from_dt,
            to_date=to_dt,
            interval="day",
        )
        if not records:
            print(" ✗ No history")
            return None

        df = pd.DataFrame(records)
        df.rename(columns={"date": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        print(f" ✓  LTP={ltp}  Chg={chg_pct:+.2f}%")
        return {
            "symbol":     symbol,
            "token":      token,
            "df":         df,
            "ltp":        ltp,
            "today_open": today_o,
            "today_high": today_h,
            "today_low":  today_l,
            "prev_close": prev_cls,
            "chg_pct":    round(chg_pct, 2),
            "fetched_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
        }
    except Exception as e:
        print(f" ✗ {e}")
        return None

# ════════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════════════════════════

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d   = close.diff()
    g   = d.clip(lower=0)
    l   = -d.clip(upper=0)
    ag  = g.ewm(alpha=1/n, adjust=False).mean()
    al  = l.ewm(alpha=1/n, adjust=False).mean()
    rs  = ag / al.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)

def macd(close, f=12, s=26, sig=9):
    m  = ema(close, f) - ema(close, s)
    sg = ema(m, sig)
    return m, sg, m - sg

def bollinger(close, n=20, k=2):
    mid = close.rolling(n).mean()
    sd  = close.rolling(n).std()
    return mid + k*sd, mid, mid - k*sd

def atr(high, low, close, n=14):
    tr = pd.concat([high-low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def add_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema9"]        = ema(c, 9)
    df["ema21"]       = ema(c, 21)
    df["ema50"]       = ema(c, 50)
    df["ema200"]      = ema(c, 200)
    df["rsi"]         = rsi(c)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(c)
    df["bb_up"], df["bb_mid"], df["bb_lo"]      = bollinger(c)
    df["atr"]         = atr(h, l, c)
    df["vol_ma20"]    = v.rolling(20).mean()
    df["vol_ratio"]   = v / df["vol_ma20"].replace(0, 1)
    df["ret5"]        = c.pct_change(5)  * 100
    df["ret10"]       = c.pct_change(10) * 100
    df["ret20"]       = c.pct_change(20) * 100
    return df

# ════════════════════════════════════════════════════════════════════════════════
# SCORING
# ════════════════════════════════════════════════════════════════════════════════

def score_stock(df, ltp):
    r  = df.iloc[-1]
    r2 = df.iloc[-2]
    score   = 0
    signals = []

    # ── TREND  25 pts ────────────────────────────────────────────────────────
    t = 0
    if ltp > r["ema9"]:   t += 3
    if ltp > r["ema21"]:  t += 5;  signals.append("Above EMA21")
    if ltp > r["ema50"]:  t += 8;  signals.append("Above EMA50")
    if ltp > r["ema200"]: t += 9;  signals.append("Above EMA200 ✓")
    if r["ema9"] > r["ema21"] > r["ema50"]:
        t = min(t+5, 25);           signals.append("EMAs aligned bullish")
    elif r["ema9"] < r["ema21"] < r["ema50"]:
                                    signals.append("EMAs aligned bearish ✗")
    score += min(t, 25)

    # ── RSI  15 pts ───────────────────────────────────────────────────────────
    rv = r["rsi"]
    if   50 <  rv < 60:  score += 15; signals.append(f"RSI {rv:.0f} sweet spot")
    elif 60 <= rv < 70:  score += 12; signals.append(f"RSI {rv:.0f} bullish")
    elif 40 <= rv < 50:  score += 10; signals.append(f"RSI {rv:.0f} recovering")
    elif 30 <= rv < 40:  score +=  8; signals.append(f"RSI {rv:.0f} near oversold")
    elif rv >= 70:       score +=  5; signals.append(f"RSI {rv:.0f} overbought ⚠")
    else:                score +=  3; signals.append(f"RSI {rv:.0f} weak")

    # ── MACD  15 pts ──────────────────────────────────────────────────────────
    if r["macd"] > r["macd_sig"]:
        score += 8; signals.append("MACD above signal")
        if r["macd_hist"] > 0 and r2["macd_hist"] <= 0:
            score += 7; signals.append("MACD crossover! 🚀")
        elif r["macd_hist"] > r2["macd_hist"]:
            score += 4; signals.append("MACD hist expanding")
    else:
        if r["macd_hist"] > r2["macd_hist"]:
            score += 3; signals.append("MACD hist turning up")

    # ── BOLLINGER  10 pts ─────────────────────────────────────────────────────
    bb_range = r["bb_up"] - r["bb_lo"]
    bb_pct   = (ltp - r["bb_lo"]) / (bb_range if bb_range > 0 else 1)
    if   0.3 < bb_pct < 0.65: score += 10; signals.append("BB mid zone")
    elif bb_pct <= 0.2:        score +=  8; signals.append("Near BB lower (bounce)")
    elif bb_pct >= 0.85:       score +=  4; signals.append("Near BB upper ⚠")
    else:                      score +=  6

    # ── VOLUME  15 pts ────────────────────────────────────────────────────────
    vr = r["vol_ratio"]
    if   vr > 3.0:  score += 15; signals.append(f"Vol {vr:.1f}x surge! 🔥")
    elif vr > 2.0:  score += 12; signals.append(f"Vol {vr:.1f}x elevated")
    elif vr > 1.2:  score +=  8; signals.append(f"Vol {vr:.1f}x above avg")
    else:           score +=  4; signals.append(f"Vol {vr:.1f}x low")

    # ── MOMENTUM  20 pts ──────────────────────────────────────────────────────
    r5  = r["ret5"]
    r20 = r["ret20"]
    if   r5 >  5:  score += 10; signals.append(f"5d +{r5:.1f}%")
    elif r5 >  1:  score +=  7; signals.append(f"5d +{r5:.1f}%")
    elif r5 > -2:  score +=  4; signals.append(f"5d {r5:.1f}%")
    else:          score +=  1; signals.append(f"5d {r5:.1f}% weak")

    if   r20 >  10: score += 10; signals.append(f"20d +{r20:.1f}%")
    elif r20 >   3: score +=  7; signals.append(f"20d +{r20:.1f}%")
    elif r20 >  -3: score +=  4; signals.append(f"20d {r20:.1f}%")
    else:           score +=  1; signals.append(f"20d {r20:.1f}% bearish")

    return min(int(score), 100), signals


def get_rec(score):
    if   score >= 75: return "STRONG BUY",  "#00e676", "🚀"
    elif score >= 60: return "BUY",          "#69f0ae", "✅"
    elif score >= 45: return "WATCH",        "#ffeb3b", "👀"
    elif score >= 30: return "WEAK",         "#ff9800", "⚠️"
    else:             return "AVOID",        "#f44336", "⛔"


def trade_levels(df, ltp):
    r    = df.iloc[-1]
    atr_val = r["atr"]
    # Support levels
    ema21_val = r["ema21"]
    bb_lo_val = r["bb_lo"]
    recent_low = df["low"].tail(10).min()

    sl       = round(max(ltp - 2.0*atr_val, min(ema21_val*0.985, recent_low*0.99)), 2)
    risk     = ltp - sl
    target1  = round(ltp + 1.5*risk, 2)
    target2  = round(ltp + 3.0*risk, 2)
    rr1      = round((target1-ltp)/risk, 1) if risk > 0 else 0
    rr2      = round((target2-ltp)/risk, 1) if risk > 0 else 0
    return {
        "entry":   ltp,
        "target1": target1,
        "target2": target2,
        "sl":      sl,
        "risk_pct":    round(risk/ltp*100, 1),
        "t1_pct":      round((target1-ltp)/ltp*100, 1),
        "t2_pct":      round((target2-ltp)/ltp*100, 1),
        "rr1":     rr1,
        "rr2":     rr2,
    }


def full_analysis(raw):
    df  = add_indicators(raw["df"].copy())
    ltp = raw["ltp"]
    score, sigs = score_stock(df, ltp)
    rec, col, ico = get_rec(score)
    lvl = trade_levels(df, ltp)
    r   = df.iloc[-1]
    return {
        **raw,
        "df":       df,
        "score":    score,
        "signals":  sigs,
        "rec":      rec,
        "rec_col":  col,
        "rec_ico":  ico,
        "levels":   lvl,
        "ind": {
            "ema9":    round(float(r["ema9"]),    2),
            "ema21":   round(float(r["ema21"]),   2),
            "ema50":   round(float(r["ema50"]),   2),
            "ema200":  round(float(r["ema200"]),  2),
            "rsi":     round(float(r["rsi"]),     1),
            "macd":    round(float(r["macd"]),    3),
            "macd_sig":round(float(r["macd_sig"]),3),
            "bb_up":   round(float(r["bb_up"]),   2),
            "bb_lo":   round(float(r["bb_lo"]),   2),
            "atr":     round(float(r["atr"]),     2),
            "vr":      round(float(r["vol_ratio"]),2),
            "ret5":    round(float(r["ret5"]),    2),
            "ret20":   round(float(r["ret20"]),   2),
        },
    }

# ════════════════════════════════════════════════════════════════════════════════
# CHART GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def make_chart_html(a):
    df   = a["df"].tail(90)
    sym  = a["symbol"]
    ltp  = a["ltp"]
    lvl  = a["levels"]
    ind  = a["ind"]
    rec  = a["rec"]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=[
            f"{sym} — Candlestick + EMAs + Bollinger Bands",
            "Volume",
            "RSI (14)",
            "MACD (12,26,9)",
        ],
    )

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    colors = {"ema9":"#ffeb3b","ema21":"#29b6f6","ema50":"#ff9800","ema200":"#ef5350"}
    for e, c in colors.items():
        fig.add_trace(go.Scatter(x=df.index, y=df[e], name=e.upper(),
            line=dict(color=c, width=1.5), opacity=0.9), row=1, col=1)

    # BB fill
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_up"], name="BB Upper",
        line=dict(color="#7e57c2", width=1, dash="dot"), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lo"], name="BB Lower",
        line=dict(color="#7e57c2", width=1, dash="dot"), opacity=0.7,
        fill="tonexty", fillcolor="rgba(126,87,194,0.06)"), row=1, col=1)

    # Trade level lines
    fig.add_hline(y=ltp,         line=dict(color="#ffffff", width=1, dash="dot"),
                  annotation_text=f"LTP {ltp}", row=1, col=1)
    fig.add_hline(y=lvl["sl"],   line=dict(color="#f44336", width=1, dash="dash"),
                  annotation_text=f"SL {lvl['sl']}", row=1, col=1)
    fig.add_hline(y=lvl["target1"], line=dict(color="#69f0ae", width=1, dash="dash"),
                  annotation_text=f"T1 {lvl['target1']}", row=1, col=1)
    fig.add_hline(y=lvl["target2"], line=dict(color="#00e676", width=1, dash="dash"),
                  annotation_text=f"T2 {lvl['target2']}", row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
        marker_color=vol_colors, opacity=0.8), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["vol_ma20"], name="Vol MA20",
        line=dict(color="#ffeb3b", width=1.5)), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_col = "#f44336" if ind["rsi"] > 70 else ("#26a69a" if ind["rsi"] < 30 else "#29b6f6")
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
        line=dict(color=rsi_col, width=2)), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#f44336", width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=50, line=dict(color="#888888", width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#26a69a", width=1, dash="dot"), row=3, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────────
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist",
        marker_color=hist_colors, opacity=0.8), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
        line=dict(color="#29b6f6", width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_sig"], name="Signal",
        line=dict(color="#ff9800", width=1.5)), row=4, col=1)

    fig.update_layout(
        height=800,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=80, t=30, b=0),
        showlegend=True,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=i, col=1)

    return fig.to_html(full_html=False, include_plotlyjs=False)

# ════════════════════════════════════════════════════════════════════════════════
# HTML REPORT GENERATOR
# ════════════════════════════════════════════════════════════════════════════════

def score_bar(score):
    col = ("#00e676" if score>=75 else "#69f0ae" if score>=60
           else "#ffeb3b" if score>=45 else "#ff9800" if score>=30 else "#f44336")
    return (f'<div style="background:#21262d;border-radius:4px;height:8px;width:100%">'
            f'<div style="background:{col};width:{score}%;height:8px;border-radius:4px"></div></div>'
            f'<small style="color:{col}">{score}/100</small>')

def ind_badge(label, val, good_fn=None):
    col = "#e6edf3"
    if good_fn:
        try:
            col = "#69f0ae" if good_fn(val) else "#f44336"
        except Exception:
            pass
    return f'<span class="badge" style="background:#21262d;color:{col};border:1px solid #30363d;margin:2px;padding:4px 8px">{label}: <b>{val}</b></span>'

def generate_html(results, report_time):
    sorted_r = sorted(results, key=lambda x: x["score"], reverse=True)

    # ── Summary table rows ────────────────────────────────────────────────────
    table_rows = ""
    for i, a in enumerate(sorted_r):
        chg_col = "#69f0ae" if a["chg_pct"] >= 0 else "#f44336"
        chg_str = f"{a['chg_pct']:+.2f}%"
        rec_col = a["rec_col"]
        table_rows += f"""
        <tr onclick="showStock('{a['symbol']}')" style="cursor:pointer">
          <td style="color:#aaa">{i+1}</td>
          <td><b style="color:#e6edf3">{a['symbol']}</b></td>
          <td>₹{a['ltp']:,.2f}</td>
          <td style="color:{chg_col}">{chg_str}</td>
          <td>{score_bar(a['score'])}</td>
          <td><span style="color:{rec_col};font-weight:bold">{a['rec_ico']} {a['rec']}</span></td>
          <td>₹{a['levels']['sl']}</td>
          <td style="color:#69f0ae">₹{a['levels']['target1']} (+{a['levels']['t1_pct']}%)</td>
          <td style="color:#00e676">₹{a['levels']['target2']} (+{a['levels']['t2_pct']}%)</td>
          <td style="color:#aaa">{a['levels']['rr1']}:1</td>
        </tr>"""

    # ── Individual stock sections ─────────────────────────────────────────────
    stock_sections = ""
    for a in sorted_r:
        chart_html = make_chart_html(a) if PLOTLY_AVAILABLE else "<p>Install plotly for charts</p>"
        ind = a["ind"]
        lvl = a["levels"]
        chg_col = "#69f0ae" if a["chg_pct"] >= 0 else "#f44336"

        badges = "".join([
            ind_badge("EMA9",   ind["ema9"],   lambda v: a["ltp"]>v),
            ind_badge("EMA21",  ind["ema21"],  lambda v: a["ltp"]>v),
            ind_badge("EMA50",  ind["ema50"],  lambda v: a["ltp"]>v),
            ind_badge("EMA200", ind["ema200"], lambda v: a["ltp"]>v),
            ind_badge("RSI",    ind["rsi"],    lambda v: 30<v<70),
            ind_badge("MACD",   ind["macd"],   lambda v: v>0),
            ind_badge("ATR",    ind["atr"]),
            ind_badge("VolRatio", f"{ind['vr']}x", lambda v: float(v[:-1])>1.2),
            ind_badge("5d",  f"{ind['ret5']:+.1f}%",  lambda v: float(v.replace('%',''))>0),
            ind_badge("20d", f"{ind['ret20']:+.1f}%", lambda v: float(v.replace('%',''))>0),
        ])

        sig_html = " ".join(
            f'<span style="background:#21262d;border-radius:12px;padding:3px 10px;'
            f'font-size:12px;color:#e6edf3;border:1px solid #30363d;display:inline-block;margin:2px">'
            f'{s}</span>' for s in a["signals"]
        )

        stock_sections += f"""
        <div id="section_{a['symbol']}" class="stock-section" style="display:none;margin-top:32px">
          <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px">

            <!-- Header -->
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:20px">
              <div>
                <h2 style="margin:0;color:#e6edf3">{a['symbol']}</h2>
                <span style="color:#aaa;font-size:13px">Fetched: {a['fetched_at']}</span>
              </div>
              <div style="text-align:right">
                <div style="font-size:28px;font-weight:700;color:#e6edf3">₹{a['ltp']:,.2f}</div>
                <div style="color:{chg_col};font-size:16px">{a['chg_pct']:+.2f}% today</div>
              </div>
              <div style="background:{a['rec_col']}22;border:2px solid {a['rec_col']};
                          border-radius:8px;padding:12px 24px;text-align:center">
                <div style="font-size:22px">{a['rec_ico']}</div>
                <div style="color:{a['rec_col']};font-weight:700;font-size:18px">{a['rec']}</div>
                <div style="color:{a['rec_col']};font-size:14px">{a['score']}/100</div>
              </div>
            </div>

            <!-- Trade Setup -->
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                        gap:12px;background:#0d1117;border-radius:8px;padding:16px;margin-bottom:16px">
              <div style="text-align:center">
                <div style="color:#aaa;font-size:11px">ENTRY</div>
                <div style="color:#e6edf3;font-size:18px;font-weight:700">₹{lvl['entry']:,.2f}</div>
              </div>
              <div style="text-align:center">
                <div style="color:#f44336;font-size:11px">STOP LOSS</div>
                <div style="color:#f44336;font-size:18px;font-weight:700">₹{lvl['sl']:,.2f}</div>
                <div style="color:#f44336;font-size:11px">-{lvl['risk_pct']}%</div>
              </div>
              <div style="text-align:center">
                <div style="color:#69f0ae;font-size:11px">TARGET 1</div>
                <div style="color:#69f0ae;font-size:18px;font-weight:700">₹{lvl['target1']:,.2f}</div>
                <div style="color:#69f0ae;font-size:11px">+{lvl['t1_pct']}% | R:R {lvl['rr1']}:1</div>
              </div>
              <div style="text-align:center">
                <div style="color:#00e676;font-size:11px">TARGET 2</div>
                <div style="color:#00e676;font-size:18px;font-weight:700">₹{lvl['target2']:,.2f}</div>
                <div style="color:#00e676;font-size:11px">+{lvl['t2_pct']}% | R:R {lvl['rr2']}:1</div>
              </div>
              <div style="text-align:center">
                <div style="color:#aaa;font-size:11px">ATR(14)</div>
                <div style="color:#e6edf3;font-size:18px;font-weight:700">₹{ind['atr']}</div>
              </div>
            </div>

            <!-- Indicators -->
            <div style="margin-bottom:16px">{badges}</div>

            <!-- Signals -->
            <div style="margin-bottom:20px">
              <div style="color:#aaa;font-size:12px;margin-bottom:8px">SIGNALS DETECTED</div>
              {sig_html}
            </div>

            <!-- Chart -->
            <div style="background:#0d1117;border-radius:8px;overflow:hidden">
              {chart_html}
            </div>

          </div>
        </div>"""

    # ── Full HTML ─────────────────────────────────────────────────────────────
    plotlyjs = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stock Analysis — {report_time}</title>
{plotlyjs}
<style>
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          background:#0d1117; color:#e6edf3; padding:0 16px 40px }}
  h1,h2,h3 {{ font-weight:600 }}
  table {{ width:100%; border-collapse:collapse }}
  th {{ background:#161b22; color:#aaa; font-size:11px; text-transform:uppercase;
        padding:10px 12px; text-align:left; border-bottom:1px solid #30363d }}
  td {{ padding:10px 12px; border-bottom:1px solid #21262d; font-size:13px }}
  tr:hover td {{ background:#161b22 }}
  .badge {{ border-radius:12px; font-size:12px }}
  .tab-btn {{ background:#21262d; border:1px solid #30363d; color:#aaa; padding:6px 16px;
              border-radius:6px; cursor:pointer; font-size:13px; transition:all .2s }}
  .tab-btn.active {{ background:#1f6feb; border-color:#1f6feb; color:#fff }}
  .tab-btn:hover {{ background:#30363d; color:#e6edf3 }}
</style>
</head>
<body>

<!-- HEADER -->
<div style="max-width:1400px;margin:0 auto">
<div style="background:linear-gradient(135deg,#1f6feb22,#161b22);border:1px solid #30363d;
            border-radius:12px;padding:24px 32px;margin:24px 0;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px">
  <div>
    <h1 style="font-size:24px;color:#e6edf3">📊 Stock Analysis Dashboard</h1>
    <p style="color:#aaa;margin-top:4px">Real-time technical analysis powered by Kite API</p>
  </div>
  <div style="text-align:right">
    <div style="color:#aaa;font-size:12px">REPORT GENERATED</div>
    <div style="color:#e6edf3;font-size:16px;font-weight:600">{report_time}</div>
    <div style="color:#aaa;font-size:12px">{len(results)} stocks analyzed</div>
  </div>
</div>

<!-- SUMMARY TABLE -->
<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
            padding:20px;margin-bottom:24px;overflow-x:auto">
  <h3 style="margin-bottom:16px;color:#e6edf3">Rankings (click row to see details)</h3>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Symbol</th><th>LTP</th><th>Change</th>
        <th>Score</th><th>Recommendation</th>
        <th>Stop Loss</th><th>Target 1</th><th>Target 2</th><th>R:R</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<!-- STOCK TAB BUTTONS -->
<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px">
  {''.join(f'<button class="tab-btn" id="btn_{a["symbol"]}" onclick="showStock(\'{a["symbol"]}\')">'
           f'{a["rec_ico"]} {a["symbol"]}</button>' for a in sorted_r)}
</div>

<!-- STOCK DETAIL SECTIONS -->
{stock_sections}

</div>

<script>
var active = null;
function showStock(sym) {{
  document.querySelectorAll('.stock-section').forEach(function(el) {{
    el.style.display = 'none';
  }});
  document.querySelectorAll('.tab-btn').forEach(function(b) {{
    b.classList.remove('active');
  }});
  if (active === sym) {{
    active = null;
    return;
  }}
  var sec = document.getElementById('section_' + sym);
  var btn = document.getElementById('btn_' + sym);
  if (sec) sec.style.display = 'block';
  if (btn) btn.classList.add('active');
  active = sym;
  if (sec) sec.scrollIntoView({{behavior:'smooth', block:'start'}});
}}

// Auto-show first stock
var first = document.querySelector('.tab-btn');
if (first) first.click();
</script>
</body>
</html>"""
    return html

# ════════════════════════════════════════════════════════════════════════════════
# CONSOLE PRINT
# ════════════════════════════════════════════════════════════════════════════════

def print_results(results):
    sorted_r = sorted(results, key=lambda x: x["score"], reverse=True)
    print("\n" + "="*90)
    print(f"  ANALYSIS RESULTS — {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
    print("="*90)
    print(f"{'#':<3} {'Symbol':<12} {'LTP':>10} {'Chg%':>7} {'Score':>6} {'Rec':<14} {'SL':>10} {'T1':>10} {'T2':>10}")
    print("-"*90)
    for i, a in enumerate(sorted_r):
        chg_str = f"{a['chg_pct']:+.1f}%"
        print(f"{i+1:<3} {a['symbol']:<12} {a['ltp']:>10,.2f} {chg_str:>7} "
              f"{a['score']:>6}  {a['rec']:<14} "
              f"{a['levels']['sl']:>10,.2f} "
              f"{a['levels']['target1']:>10,.2f} "
              f"{a['levels']['target2']:>10,.2f}")
    print("="*90)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stock Analysis Tool")
    parser.add_argument("symbols",  nargs="*", help="NSE stock symbols")
    parser.add_argument("--setup",  action="store_true", help="Configure API credentials")
    parser.add_argument("--portfolio", action="store_true", help="Analyze your holdings")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--top",    type=int, default=0, help="Show only top N")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  STOCK ANALYSIS TOOL — Kite API v3")
    print("="*50)

    # Setup mode
    if args.setup:
        setup_credentials()
        return

    # Connect
    kite = get_kite()

    # Get symbols
    symbols = [s.upper() for s in args.symbols]

    if args.portfolio:
        print("\nFetching portfolio holdings...")
        try:
            holdings = kite.holdings()
            symbols += [h["tradingsymbol"] for h in holdings if h["quantity"] > 0]
            print(f"Found {len(holdings)} holdings")
        except Exception as e:
            print(f"Could not fetch holdings: {e}")

    if not symbols:
        print("\nEnter stock symbols (comma or space separated):")
        print("Example: RELIANCE INFY TCS TEJASNET")
        raw = input("Symbols: ").strip()
        symbols = [s.strip().upper() for s in raw.replace(",", " ").split() if s.strip()]

    if not symbols:
        print("No symbols provided. Exiting.")
        return

    # Fetch & analyze
    print(f"\nFetching data for {len(symbols)} stocks...")
    results = []
    for sym in symbols:
        raw = fetch_stock(kite, sym)
        if raw:
            try:
                analyzed = full_analysis(raw)
                results.append(analyzed)
            except Exception as e:
                print(f"  ✗ Analysis failed for {sym}: {e}")

    if not results:
        print("No data fetched. Check your symbols and token.")
        return

    if args.top:
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:args.top]

    # Console output
    print_results(results)

    # HTML report
    if PLOTLY_AVAILABLE:
        report_time = datetime.now().strftime("%d %b %Y %H:%M:%S")
        html = generate_html(results, report_time)
        REPORT_FILE.write_text(html, encoding="utf-8")
        print(f"\n✓ Report saved: {REPORT_FILE}")
        if not args.no_browser:
            webbrowser.open(f"file:///{REPORT_FILE.as_posix()}")
            print("✓ Opened in browser")
    else:
        print("\nInstall plotly for HTML report:  pip install plotly")


if __name__ == "__main__":
    main()
