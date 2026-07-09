"""
Interactive Kite Stock Dashboard
=================================
Generates interactive Plotly charts with:
- Candlestick + volume
- EMA 9/21/50/200
- Bollinger Bands
- RSI panel
- MACD panel
- Trade levels (entry / target / stop-loss)

Usage:
    python dashboard.py INFY
    python dashboard.py --all    # chart all recommendations from recommendations.json
"""

import argparse
import json
import sys
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

import pandas as pd


def show_chart(
    symbol: str,
    df: pd.DataFrame,
    rec: dict = None,
    save_html: bool = True,
    open_browser: bool = True,
) -> str:
    """
    Plot interactive candlestick chart with indicators.
    Returns path to saved HTML file.
    """
    if not PLOTLY_OK:
        print("plotly not installed. Run: pip install plotly")
        return ""

    # --- Layout ---
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        vertical_spacing=0.03,
        subplot_titles=[f"{symbol} — Daily Chart", "Volume", "RSI (14)", "MACD (12,26,9)"],
    )

    dates = df.index

    # ─── Row 1: Candlestick ───────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # EMAs
    ema_colors = {"ema9": "#ff9800", "ema21": "#2196f3", "ema50": "#9c27b0", "ema200": "#f44336"}
    for col_name, color in ema_colors.items():
        if col_name in df.columns:
            period = col_name.replace("ema", "")
            fig.add_trace(go.Scatter(
                x=dates, y=df[col_name],
                name=f"EMA {period}",
                line=dict(color=color, width=1.2),
            ), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["bb_upper"],
            name="BB Upper", line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df["bb_lower"],
            name="BB Lower", line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,255,0.05)",
        ), row=1, col=1)

    # Trade levels
    if rec:
        entry = rec.get("entry")
        target = rec.get("target")
        sl = rec.get("stop_loss")
        last_date = dates[-1]
        future_dates = [last_date, last_date + pd.Timedelta(days=21)]

        if entry:
            fig.add_hline(y=entry,  line_dash="dash", line_color="#ffd600",
                          annotation_text=f"Entry: ₹{entry:.2f}", row=1, col=1)
        if target:
            fig.add_hline(y=target, line_dash="dash", line_color="#00e676",
                          annotation_text=f"Target: ₹{target:.2f} (+{rec.get('expected_return_pct', 0):.1f}%)", row=1, col=1)
        if sl:
            fig.add_hline(y=sl,     line_dash="dash", line_color="#ff1744",
                          annotation_text=f"Stop: ₹{sl:.2f}", row=1, col=1)

    # ─── Row 2: Volume ────────────────────────────────────────────────────
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=dates, y=df["volume"], name="Volume",
        marker_color=colors, showlegend=False,
    ), row=2, col=1)

    if "vol_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["vol_ma20"],
            name="Vol MA20", line=dict(color="#ff9800", width=1),
        ), row=2, col=1)

    # ─── Row 3: RSI ───────────────────────────────────────────────────────
    if "rsi" in df.columns:
        rsi_colors = [
            "#ef5350" if v > 70 else ("#26a69a" if v < 30 else "#90caf9")
            for v in df["rsi"]
        ]
        fig.add_trace(go.Scatter(
            x=dates, y=df["rsi"],
            name="RSI", line=dict(color="#90caf9", width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.5)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.5)", row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(144,202,249,0.05)", row=3, col=1)

    # ─── Row 4: MACD ──────────────────────────────────────────────────────
    if "macd" in df.columns:
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
        fig.add_trace(go.Bar(
            x=dates, y=df["macd_hist"],
            name="MACD Histogram", marker_color=hist_colors, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df["macd"],
            name="MACD", line=dict(color="#2196f3", width=1.5),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=df["macd_signal"],
            name="Signal", line=dict(color="#ff9800", width=1.5),
        ), row=4, col=1)

    # ─── Layout styling ───────────────────────────────────────────────────
    title_text = f"{symbol}"
    if rec:
        title_text += (
            f"  |  Score: {rec.get('score', 0):.0f}/100"
            f"  |  Pattern: {rec.get('pattern', '')}"
            f"  |  Horizon: {rec.get('horizon', '')}"
        )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color="white")),
        template="plotly_dark",
        height=900,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="white"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.07)",
        showspikes=True, spikemode="across",
        spikesnap="cursor", spikecolor="rgba(255,255,255,0.3)",
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")

    # Save
    path = f"chart_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    fig.write_html(path, auto_open=open_browser)
    print(f"Chart saved: {path}")
    return path


def chart_all_recommendations(recs_path: str = "recommendations.json"):
    """Generate charts for all recommendations in the JSON file."""
    try:
        with open(recs_path) as f:
            recs = json.load(f)
    except FileNotFoundError:
        print(f"No {recs_path} found. Run kite_analyzer.py --export first.")
        return

    # Import here to avoid circular
    from kite_analyzer import KiteFetcher, add_indicators

    fetcher = KiteFetcher()
    for i, rec in enumerate(recs):
        symbol = rec["symbol"]
        print(f"Charting {symbol} ({i+1}/{len(recs)})...")
        df = fetcher.fetch_history(symbol, days=120)
        if df is not None and len(df) >= 30:
            df = add_indicators(df)
            show_chart(symbol, df, rec=rec, open_browser=(i == 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kite Stock Chart Dashboard")
    parser.add_argument("symbol", nargs="?", help="Symbol to chart (e.g. INFY)")
    parser.add_argument("--all", action="store_true", help="Chart all recommendations")
    args = parser.parse_args()

    if args.all:
        chart_all_recommendations()
    elif args.symbol:
        from kite_analyzer import KiteFetcher, add_indicators
        fetcher = KiteFetcher()
        print(f"Fetching data for {args.symbol}...")
        df = fetcher.fetch_history(args.symbol, days=120)
        if df is not None:
            df = add_indicators(df)
            show_chart(args.symbol, df, open_browser=True)
        else:
            print(f"Could not fetch data for {args.symbol}")
    else:
        parser.print_help()
