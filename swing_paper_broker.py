"""
swing_paper_broker.py — paper trading for the swing screener.

Tracks equity swing trades (1-3 week horizon) in a local JSON file.
Each trade stores entry price, ATR-based target/SL, live LTP, and P&L.

Charges reflect Zerodha equity delivery:
  • Brokerage : ₹0 (Zerodha delivery is free)
  • STT        : 0.10% of sell turnover
  • Exchange   : 0.00345% of total turnover (NSE)
  • SEBI       : 0.0001% of total turnover
  • Stamp      : 0.015% of buy turnover
  • GST        : 18% on (exchange + SEBI)
  Total round-trip on ₹10,000: approx ₹13-15
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

TRADES_FILE = Path(__file__).parent / "swing_paper_trades.json"
_IST = timezone(timedelta(hours=5, minutes=30))

# Time stop: the mandate is 5-10% in 1-2 weeks. A position that has hit
# neither target nor SL after this many sessions is a failed thesis tying
# up capital — exit at market and recycle into a fresh signal.
TIME_STOP_SESSIONS = 10

# Position sizing: risk a fixed slice of the notional account per trade,
# so one stopped-out position can never dominate the book the way an
# arbitrary share count can. Mirrored in the UI (swing_trade_screener.html).
ACCOUNT_CAPITAL = 500_000     # notional paper account
RISK_PCT_PER_TRADE = 1.0      # % of account lost if the stop is hit


def suggest_qty(entry_price: float, sl_pct: float) -> int:
    """Share count such that hitting the SL loses RISK_PCT_PER_TRADE of
    ACCOUNT_CAPITAL. Returns 0 when the inputs can't size a position."""
    per_share_risk = float(entry_price) * float(sl_pct) / 100
    if per_share_risk <= 0:
        return 0
    return int((ACCOUNT_CAPITAL * RISK_PCT_PER_TRADE / 100) // per_share_risk)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _now_ist() -> str:
    return datetime.now(_IST).isoformat(timespec="seconds")


def _load() -> dict:
    if TRADES_FILE.exists():
        try:
            return json.loads(TRADES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"open": [], "closed": []}


def _save(state: dict):
    TRADES_FILE.write_text(
        json.dumps(state, indent=2, default=str),
        encoding="utf-8",
    )


def _calc_charges(entry: float, exit_price: float, qty: int) -> float:
    """Estimate Zerodha equity delivery round-trip charges."""
    buy_turn   = entry      * qty
    sell_turn  = exit_price * qty
    total_turn = buy_turn   + sell_turn

    stt      = round(sell_turn  * 0.001,       2)   # 0.10% sell only
    exchange = round(total_turn * 0.0000345,   2)   # 0.00345% both sides
    sebi     = round(total_turn * 0.000001,    2)   # 0.0001%
    stamp    = round(buy_turn   * 0.00015,     2)   # 0.015% buy only
    gst      = round((exchange + sebi) * 0.18, 2)   # 18% on fees
    return round(stt + exchange + sebi + stamp + gst, 2)


def _sessions_held(entry_ts: str) -> int:
    """Completed weekday sessions since entry (NSE holidays not excluded —
    close enough for a time stop; err on the side of holding a day longer)."""
    try:
        entry_day = datetime.fromisoformat(entry_ts).date()
    except (ValueError, TypeError):
        return 0
    today = datetime.now(_IST).date()
    d, n = entry_day, 0
    while d < today:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _do_close(trade: dict, exit_price: float, reason: str):
    """Mutate trade dict in-place, filling exit fields + P&L."""
    trade["status"]      = "closed"
    trade["exit_price"]  = round(float(exit_price), 2)
    trade["exit_ts"]     = _now_ist()
    trade["exit_reason"] = reason.upper()
    gross = round((trade["exit_price"] - trade["entry_price"]) * trade["qty"], 2)
    charges = _calc_charges(trade["entry_price"], trade["exit_price"], trade["qty"])
    trade["gross_pnl"] = gross
    trade["charges"]   = charges
    trade["net_pnl"]   = round(gross - charges, 2)


# ─── Public API ─────────────────────────────────────────────────────────────

def buy(
    symbol:     str,
    qty:        int,
    entry_price: float,
    target_pct: float,
    sl_pct:     float,
    signal:     str = "",
    score:      int = 0,
) -> dict:
    """Open a paper position, averaging into an existing open symbol if present."""
    state        = _load()
    symbol       = symbol.upper()
    qty          = int(qty)
    entry_price  = round(float(entry_price), 2)
    target_price = round(entry_price * (1 + target_pct / 100), 2)
    sl_price     = round(entry_price * (1 - sl_pct    / 100), 2)
    now_ts       = _now_ist()

    matches = [t for t in state["open"] if t.get("symbol", "").upper() == symbol]
    if matches:
        trade = matches[0]
        old_qty = sum(int(t.get("qty") or 0) for t in matches)
        old_capital = sum(
            float(t.get("entry_price") or 0) * int(t.get("qty") or 0)
            for t in matches
        )
        old_entry = round(old_capital / old_qty, 2) if old_qty > 0 else entry_price
        total_qty = old_qty + qty
        if total_qty <= 0:
            return trade

        avg_entry = round((old_capital + (entry_price * qty)) / total_qty, 2)
        trade["qty"]          = total_qty
        trade["entry_price"]  = avg_entry
        trade["target_pct"]   = round(float(target_pct), 2)
        trade["sl_pct"]       = round(float(sl_pct), 2)
        trade["target_price"] = round(avg_entry * (1 + target_pct / 100), 2)
        trade["sl_price"]     = round(avg_entry * (1 - sl_pct / 100), 2)
        trade["signal"]       = signal
        trade["score"]        = int(score)
        trade["ltp"]          = entry_price
        trade["ltp_ts"]       = now_ts
        trade["last_add_ts"]  = now_ts
        trade["last_add_qty"] = qty
        trade["last_add_price"] = entry_price
        trade["avg_count"]    = int(trade.get("avg_count") or 1) + 1
        trade["averaged"]     = True
        state["open"] = [
            t for t in state["open"]
            if t is trade or t.get("symbol", "").upper() != symbol
        ]
        _save(state)
        response = dict(trade)
        response["previous_qty"] = old_qty
        response["previous_entry_price"] = old_entry
        return response

    trade = {
        "id":           str(uuid.uuid4())[:8],
        "symbol":       symbol,
        "qty":          qty,
        "entry_price":  entry_price,
        "entry_ts":     now_ts,
        "target_price": target_price,
        "sl_price":     sl_price,
        "target_pct":   round(float(target_pct), 2),
        "sl_pct":       round(float(sl_pct),     2),
        "signal":       signal,
        "score":        int(score),
        "status":       "open",
        "ltp":          entry_price,
        "ltp_ts":       now_ts,
        "exit_price":   None,
        "exit_ts":      None,
        "exit_reason":  None,
        "gross_pnl":    None,
        "charges":      None,
        "net_pnl":      None,
    }
    state["open"].append(trade)
    _save(state)
    return trade


def close_position(trade_id: str, exit_price: float,
                   reason: str = "MANUAL") -> Optional[dict]:
    """Manually close an open position. Returns the closed trade or None."""
    state = _load()
    for i, t in enumerate(state["open"]):
        if t["id"] == trade_id:
            _do_close(t, exit_price, reason)
            state["closed"].insert(0, state["open"].pop(i))
            _save(state)
            return t
    return None


def list_positions() -> dict:
    """Return {open: [...], closed: [...]} from disk."""
    return _load()


def reset():
    """Wipe all paper trades."""
    _save({"open": [], "closed": []})


def refresh_ltps(kite_client) -> dict:
    """
    Fetch latest LTPs for all open positions via Kite API.
    Auto-closes positions that hit their target or SL, plus a time stop:
    positions still open after TIME_STOP_SESSIONS weekday sessions exit
    at LTP with reason "TIME" (target/SL take precedence on the same tick).

    Returns:
        {
          "updated":      [list of open trades with refreshed LTP],
          "auto_closed":  [list of trades that were closed],
        }
    """
    state = _load()
    if not state["open"]:
        return {"updated": [], "auto_closed": []}

    symbols = list({f"NSE:{t['symbol']}" for t in state["open"]})
    try:
        ltp_data = kite_client.ltp(symbols)
    except Exception:
        return {"updated": state["open"], "auto_closed": []}

    now_ts    = _now_ist()
    auto_closed: list[dict] = []
    still_open: list[dict]  = []

    for t in state["open"]:
        key   = f"NSE:{t['symbol']}"
        price = (ltp_data.get(key) or {}).get("last_price", 0)
        if price and price > 0:
            t["ltp"]    = round(float(price), 2)
            t["ltp_ts"] = now_ts

        ltp = t["ltp"]
        if ltp >= t["target_price"]:
            _do_close(t, ltp, "TARGET")
            state["closed"].insert(0, t)
            auto_closed.append(t)
        elif ltp <= t["sl_price"]:
            _do_close(t, ltp, "SL")
            state["closed"].insert(0, t)
            auto_closed.append(t)
        elif _sessions_held(t.get("entry_ts", "")) >= TIME_STOP_SESSIONS:
            _do_close(t, ltp, "TIME")
            state["closed"].insert(0, t)
            auto_closed.append(t)
        else:
            still_open.append(t)

    state["open"] = still_open
    _save(state)
    return {"updated": still_open, "auto_closed": auto_closed}


def summary(state: Optional[dict] = None) -> dict:
    """Compute summary stats across open + closed trades."""
    if state is None:
        state = _load()

    open_trades   = state.get("open",   [])
    closed_trades = state.get("closed", [])

    # Capital deployed in open positions
    capital_open = round(sum(t["entry_price"] * t["qty"] for t in open_trades), 2)
    capital_closed = round(sum(t["entry_price"] * t["qty"] for t in closed_trades), 2)
    capital_total = round(capital_open + capital_closed, 2)

    # Unrealized P&L on open positions (gross)
    unrealized = round(
        sum((t["ltp"] - t["entry_price"]) * t["qty"] for t in open_trades), 2
    )

    # Realized net P&L (closed trades)
    realized = round(sum((t["net_pnl"] or 0) for t in closed_trades), 2)
    charges  = round(sum((t["charges"] or 0) for t in closed_trades), 2)

    wins   = sum(1 for t in closed_trades if (t["net_pnl"] or 0) > 0)
    losses = sum(1 for t in closed_trades if (t["net_pnl"] or 0) < 0)
    total_closed = wins + losses
    win_rate = round(wins / total_closed * 100) if total_closed else 0

    return {
        "open_count":    len(open_trades),
        "closed_count":  len(closed_trades),
        "capital_open":  capital_open,
        "capital_closed": capital_closed,
        "capital_total": capital_total,
        "unrealized":    unrealized,
        "realized":      realized,
        "charges_paid":  charges,
        "total_pnl":     round(realized + unrealized, 2),
        "wins":          wins,
        "losses":        losses,
        "win_rate":      win_rate,
    }
