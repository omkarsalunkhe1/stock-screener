"""
backtest.py — walk-forward backtest of the swing screener.

Replays the EXACT production signal logic (compute_indicators, score_stock,
detect_price_action from screener.py) over cached Kite historical data, then
simulates every signal with the production exit rules:

    entry      next session's OPEN after the signal date
    target     +3x ATR (as scored)            -> exit TARGET
    stop       -1.5x ATR                      -> exit SL
    time stop  --max-hold sessions (def. 10)  -> exit TIME at close
    data ends  before resolution              -> exit EOD at last close

Daily-bar fills are approximated conservatively: a gap through a level fills
at the open; if one bar's low breaches the SL and its high reaches the
target, the SL is assumed to fill FIRST. Charges use the paper broker's
Zerodha delivery model on a notional position.

Look-ahead safety: at each rebalance date t, indicators see only candles
dated <= t (the same completed-candle convention as the live scanner) and
the entry uses the t+1 open.

Known limitations (documented, not modelled):
  - earnings / ASM-GSM gates: no historical calendars exist for either, so
    live results should be slightly BETTER than backtest around results days
  - survivorship: universe lists are today's constituents
  - weekly trend filter is off (matches the server default)

Usage:
    python backtest.py                              # nifty50, 2y, weekly scans
    python backtest.py --universe nifty100 --years 3 --every 5
    python backtest.py --min-score 75               # strong-buy only
    python backtest.py --refresh                    # refetch cached history
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from screener import (
    SwingScreener, UNIVERSES, compute_indicators, score_stock,
    detect_price_action, calc_momentum,
)
from swing_paper_broker import _calc_charges

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR    = Path(__file__).parent / "data" / "backtest_cache"
NOTIONAL     = 100_000          # rupees per simulated position (for charges)
WARMUP       = 265              # sessions of history a signal needs (matches live fetch)
INDEX_SYMBOL = "NIFTY 50"

DEFAULT_FILTERS = {
    "rsi_min": 45, "rsi_max": 68, "min_vol_surge": 1.5,
    "min_score": 60, "min_target_pct": 5.0,
}


# ---------------------------------------------------------------------------
# Data layer — fetch once, cache to disk
# ---------------------------------------------------------------------------

def load_history(sc: SwingScreener, symbols: list, years: float,
                 refresh: bool = False) -> dict:
    """Return {symbol: daily OHLCV DataFrame} covering `years` + warmup."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sessions = int(years * 252) + WARMUP + 20
    hist = {}
    for i, sym in enumerate(symbols):
        cache = CACHE_DIR / f"{sym.replace('&', '_')}.csv"
        if cache.exists() and not refresh:
            df = pd.read_csv(cache, parse_dates=["date"])
        else:
            log.info(f"  [{i+1}/{len(symbols)}] fetching {sym}...")
            df = sc._fetch(sym, "day", sessions)
            time.sleep(0.35)
            if df is None or len(df) < WARMUP + 15:
                log.warning(f"  {sym}: insufficient history — skipped")
                continue
            df.to_csv(cache, index=False)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        hist[sym] = df.reset_index(drop=True)
    return hist


# ---------------------------------------------------------------------------
# Signal generation — mirrors analyse_stock's gate chain (minus event gates)
# ---------------------------------------------------------------------------

def signal_asof(df: pd.DataFrame, n_candles: int, index_close: pd.Series,
                filters: dict) -> "dict | None":
    """Score a symbol as of its n_candles-th session. Returns signal or None.

    Gate order is identical to SwingScreener.analyse_stock: RSI window ->
    volume surge -> base score -> volatility gate; PA boost applied after.
    """
    window = df.iloc[:n_candles].tail(WARMUP)
    if len(window) < 30:
        return None
    ind = compute_indicators(window)

    # Relative strength vs the index, as of the same date
    if index_close is not None and len(index_close) > 21:
        ind["rs_nifty_5d"]  = round(ind["momentum_5d"]  - calc_momentum(index_close, 5),  2)
        ind["rs_nifty_20d"] = round(ind["momentum_20d"] - calc_momentum(index_close, 20), 2)

    if not (filters["rsi_min"] <= ind["rsi"] <= filters["rsi_max"]):
        return None
    if ind["volume_ratio"] < filters["min_vol_surge"]:
        return None
    scoring = score_stock(ind, filters)
    if scoring["score"] < filters["min_score"]:
        return None
    if scoring["target_pct"] < filters["min_target_pct"]:
        return None

    pa = detect_price_action(window)
    return {
        "score":      min(130, scoring["score"] + pa["pa_score"]),
        "base_score": scoring["score"],
        "target_pct": scoring["target_pct"],
        "sl_pct":     scoring["sl_pct"],
        "rsi":        ind["rsi"],
        "vol_ratio":  ind["volume_ratio"],
        "rs20":       ind.get("rs_nifty_20d"),
    }


# ---------------------------------------------------------------------------
# Trade simulation — production exit rules on daily bars
# ---------------------------------------------------------------------------

def simulate(df: pd.DataFrame, entry_i: int, target_pct: float, sl_pct: float,
             max_hold: int) -> "dict | None":
    """Enter at bar entry_i's open; walk forward applying TARGET/SL/TIME."""
    if entry_i >= len(df):
        return None
    o = df["open"].values;  h = df["high"].values
    l = df["low"].values;   c = df["close"].values
    entry = float(o[entry_i])
    if not np.isfinite(entry) or entry <= 0:
        return None
    tgt = entry * (1 + target_pct / 100)
    sl  = entry * (1 - sl_pct / 100)

    max_fav = 0.0            # best excursion, for the mandate metric
    last_i  = min(entry_i + max_hold - 1, len(df) - 1)
    exit_px, reason, exit_i = float(c[last_i]), "TIME", last_i

    for i in range(entry_i, last_i + 1):
        max_fav = max(max_fav, (h[i] / entry - 1) * 100)
        if i > entry_i and o[i] <= sl:            # gap through stop
            exit_px, reason, exit_i = float(o[i]), "SL", i; break
        if i > entry_i and o[i] >= tgt:           # gap through target
            exit_px, reason, exit_i = float(o[i]), "TARGET", i; break
        if l[i] <= sl:                            # conservative: SL first
            exit_px, reason, exit_i = sl, "SL", i; break
        if h[i] >= tgt:
            exit_px, reason, exit_i = tgt, "TARGET", i; break

    if reason == "TIME" and last_i == len(df) - 1 and last_i < entry_i + max_hold - 1:
        reason = "EOD"       # history ended before the trade could resolve

    qty        = max(1, int(NOTIONAL / entry))
    gross_pct  = (exit_px / entry - 1) * 100
    charges    = _calc_charges(entry, exit_px, qty)
    net_pct    = gross_pct - charges / (entry * qty) * 100
    return {
        "entry_date": df["date"].iloc[entry_i].date(), "entry": round(entry, 2),
        "exit_date":  df["date"].iloc[exit_i].date(),  "exit":  round(exit_px, 2),
        "held":       exit_i - entry_i + 1,
        "reason":     reason,
        "gross_pct":  round(gross_pct, 2),
        "net_pct":    round(net_pct, 2),
        "max_fav":    round(max_fav, 2),
    }


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------

def run_backtest(hist: dict, index_df: pd.DataFrame, filters: dict,
                 every: int = 5, max_hold: int = 10, top: int = 0) -> list:
    """Scan every `every` sessions on the index calendar; simulate signals."""
    idx_dates = index_df["date"].values
    idx_close = index_df["close"]
    # candle-count position of each symbol's dates on a common axis
    sym_dates = {s: d["date"].values for s, d in hist.items()}

    trades = []
    scan_points = range(WARMUP, len(idx_dates) - 1, every)
    for k in scan_points:
        t = idx_dates[k]
        idx_slice = idx_close.iloc[:k + 1].tail(WARMUP)
        cohort = []
        for sym, df in hist.items():
            n = int(np.searchsorted(sym_dates[sym], t, side="right"))
            if n < WARMUP // 2 or n >= len(df):     # need history AND a next bar
                continue
            sig = signal_asof(df, n, idx_slice, filters)
            if sig:
                sig.update(symbol=sym, n=n)
                cohort.append(sig)
        cohort.sort(key=lambda s: s["score"], reverse=True)
        if top:
            cohort = cohort[:top]
        for sig in cohort:
            tr = simulate(hist[sig["symbol"]], sig["n"], sig["target_pct"],
                          sig["sl_pct"], max_hold)
            if tr:
                tr.update(symbol=sig["symbol"], score=sig["score"],
                          base_score=sig["base_score"], rs20=sig["rs20"],
                          target_pct=sig["target_pct"], sl_pct=sig["sl_pct"],
                          rsi=sig["rsi"], vol_ratio=sig["vol_ratio"])
                trades.append(tr)
    return trades


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _bucket_stats(rows: list) -> str:
    n = len(rows)
    if not n:
        return f"{'—':>5}"
    tgt  = sum(1 for r in rows if r["reason"] == "TARGET") / n * 100
    sl   = sum(1 for r in rows if r["reason"] == "SL") / n * 100
    tm   = sum(1 for r in rows if r["reason"] in ("TIME", "EOD")) / n * 100
    net  = sum(r["net_pct"] for r in rows) / n
    m5   = sum(1 for r in rows if r["max_fav"] >= 5.0) / n * 100
    hold = sum(r["held"] for r in rows) / n
    return (f"{n:>5} {tgt:>8.0f}% {sl:>6.0f}% {tm:>6.0f}% "
            f"{net:>+9.2f}% {m5:>9.0f}% {hold:>7.1f}")


def report(trades: list, label: str):
    print()
    print("=" * 78)
    print(f"BACKTEST REPORT — {label}")
    print("=" * 78)
    if not trades:
        print("No trades generated. Loosen filters or extend the window.")
        return
    d0 = min(t["entry_date"] for t in trades)
    d1 = max(t["exit_date"] for t in trades)
    print(f"{len(trades)} trades, {d0} -> {d1}")
    hdr = (f"{'bucket':<14}{'n':>5} {'TARGET':>8} {'SL':>7} {'TIME':>7} "
           f"{'avg net':>10} {'hit +5%':>10} {'hold d':>8}")
    print(); print(hdr); print("-" * len(hdr))
    print(f"{'ALL':<14}" + _bucket_stats(trades))
    print()
    for lo, hi in [(60, 69), (70, 79), (80, 89), (90, 131)]:
        rows = [t for t in trades if lo <= t["score"] <= hi]
        tag = f"score {lo}-{hi}" if hi < 131 else "score 90+"
        print(f"{tag:<14}" + _bucket_stats(rows))
    print()
    for tag, cond in [("RS20 > +3", lambda r: (r["rs20"] or 0) > 3),
                      ("RS20 0..+3", lambda r: r["rs20"] is not None and 0 <= r["rs20"] <= 3),
                      ("RS20 < 0",  lambda r: (r["rs20"] or 0) < 0)]:
        print(f"{tag:<14}" + _bucket_stats([t for t in trades if cond(t)]))
    print()
    wins = [t["net_pct"] for t in trades if t["net_pct"] > 0]
    losses = [t["net_pct"] for t in trades if t["net_pct"] <= 0]
    print(f"win rate {len(wins)/len(trades)*100:.0f}%   "
          f"avg win {np.mean(wins) if wins else 0:+.2f}%   "
          f"avg loss {np.mean(losses) if losses else 0:+.2f}%   "
          f"expectancy {np.mean([t['net_pct'] for t in trades]):+.2f}%/trade")
    print()
    print("columns: TARGET/SL/TIME = exit-reason share | avg net = mean net P&L")
    print("         hit +5% = trades touching +5% within the hold (mandate metric)")
    print("caveats: no earnings/surveillance gates in backtest; today's universe")
    print("         constituents (survivorship); SL-first fill assumption.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Walk-forward screener backtest")
    ap.add_argument("--universe", default="nifty50", choices=sorted(UNIVERSES))
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--every", type=int, default=5, help="rebalance cadence in sessions")
    ap.add_argument("--max-hold", type=int, default=10, help="time stop in sessions")
    ap.add_argument("--top", type=int, default=0, help="cap signals per scan (0 = all)")
    ap.add_argument("--min-score", type=int, default=DEFAULT_FILTERS["min_score"])
    ap.add_argument("--min-vol", type=float, default=DEFAULT_FILTERS["min_vol_surge"])
    ap.add_argument("--refresh", action="store_true", help="refetch cached history")
    ap.add_argument("--out", default="", help="write trades CSV to this path")
    args = ap.parse_args()

    filters = dict(DEFAULT_FILTERS, min_score=args.min_score, min_vol_surge=args.min_vol)

    cfg = json.load(open(Path(__file__).parent / "kite_config.json"))
    sc = SwingScreener(api_key=cfg["api_key"], access_token=cfg["access_token"])
    sc.load_instruments()

    symbols = UNIVERSES[args.universe]
    log.info(f"Loading history: {len(symbols)} symbols + index, {args.years:g}y "
             f"(cache: {CACHE_DIR})")
    hist = load_history(sc, symbols, args.years, refresh=args.refresh)
    index_hist = load_history(sc, [INDEX_SYMBOL], args.years, refresh=args.refresh)
    index_df = index_hist.get(INDEX_SYMBOL)
    if index_df is None:
        raise SystemExit("Could not load index history")

    log.info(f"Replaying {len(hist)} symbols, scan every {args.every} sessions...")
    trades = run_backtest(hist, index_df, filters,
                          every=args.every, max_hold=args.max_hold, top=args.top)

    label = (f"{args.universe}, {args.years:g}y, every {args.every} sessions, "
             f"min_score {args.min_score}, vol>={args.min_vol}x, hold<={args.max_hold}")
    report(trades, label)

    if args.out:
        pd.DataFrame(trades).to_csv(args.out, index=False)
        log.info(f"Trades written to {args.out}")


if __name__ == "__main__":
    main()
