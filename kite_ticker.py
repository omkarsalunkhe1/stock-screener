"""
kite_ticker.py — real-time Kite WebSocket price feed.

Maintains a single KiteTicker connection and a live last-price cache.
Other modules call get_ltp() / get_ltp_batch() instead of kite.ltp(),
eliminating repeated HTTP REST calls.

Supports any Kite instrument (NSE equities, NFO options, indices).
Subscribes in MODE_LTP — only last_price is needed for P&L and SL checks.

Lifecycle
─────────
  start(api_key, access_token, kite_client)   ← call when user authenticates
  subscribe_symbols(["RELIANCE", "TCS"])       ← call when positions open
  subscribe_tokens([738561, 260105])           ← call with raw tokens (NFO)
  get_ltp("NSE:RELIANCE")  → float | None     ← read from live cache
  stop()                                       ← clean shutdown (optional)

Reconnect: KiteTicker handles reconnection automatically (up to 50 attempts).
After each reconnect, on_connect re-subscribes all previously known tokens.

Thread safety: CPython's GIL makes single-key dict reads/writes atomic.
No explicit lock is needed for the price cache (read-heavy, write-simple).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# ─── Module-level state ───────────────────────────────────────────────────────

_ticker                         = None          # KiteTicker instance
_price_cache:  dict[int, float] = {}            # {instrument_token → last_price}
_tok_to_sym:   dict[int, str]   = {}            # {token → "NSE:SYMBOL"}
_sym_to_tok:   dict[str, int]   = {}            # {"NSE:SYMBOL" → token}
_subscribed:   set[int]         = set()         # all tokens ever subscribed
_connected                      = False
_last_tick_ts: Optional[str]    = None
_tick_count                     = 0
_api_key                        = ""
_access_token                   = ""
_kite                           = None          # KiteConnect client

_sub_lock = threading.Lock()                    # guards _subscribed mutations


# ─── Public API ───────────────────────────────────────────────────────────────

def start(api_key: str, access_token: str, kite_client) -> bool:
    """
    Start the WebSocket ticker with the given credentials.

    If already connected with the SAME access_token, this is a no-op.
    If the token changed (daily rotation), tears down the old connection
    and reconnects with the new one.

    Returns True if the connection attempt was launched (async — the
    WebSocket handshake happens in a background thread).
    """
    global _api_key, _access_token, _kite, _connected, _ticker

    if _connected and _access_token == access_token:
        return True   # already running with this token

    _api_key      = api_key
    _access_token = access_token
    _kite         = kite_client

    _connect()
    return True


def stop():
    """Gracefully disconnect the ticker."""
    global _ticker, _connected
    if _ticker:
        try:
            _ticker.close()
        except Exception:
            pass
    _ticker    = None
    _connected = False
    log.info("[ticker] Stopped")


def is_connected() -> bool:
    return _connected


def subscribe_symbols(symbols: list[str]) -> int:
    """
    Subscribe to NSE equity / index symbols by name.
    Accepts both "RELIANCE" and "NSE:RELIANCE".
    Returns the number of NEW tokens subscribed.
    """
    if not _kite:
        return 0
    tokens = []
    for raw in symbols:
        sym = raw.upper().strip()
        if not sym.startswith("NSE:"):
            sym = "NSE:" + sym
        if sym in _sym_to_tok:
            tokens.append(_sym_to_tok[sym])
        else:
            tok = _resolve_nse_token(sym)
            if tok:
                tokens.append(tok)
    return _do_subscribe(tokens)


def subscribe_tokens(tokens: list[int]) -> int:
    """
    Subscribe to instruments by their raw Kite instrument_token integers.
    Use this for NFO options / futures where you already have the token.
    Returns the number of NEW tokens subscribed.
    """
    return _do_subscribe(tokens)


def get_ltp(symbol: str) -> Optional[float]:
    """
    Return live last-traded price for an NSE symbol.
    symbol: "RELIANCE" or "NSE:RELIANCE"
    Returns None if not in cache (not yet subscribed or no tick received).
    """
    sym = symbol.upper().strip()
    if not sym.startswith("NSE:"):
        sym = "NSE:" + sym
    tok = _sym_to_tok.get(sym)
    if tok is None:
        return None
    return _price_cache.get(tok)


def get_ltp_by_token(token: int) -> Optional[float]:
    """Return live price for any instrument by its integer token."""
    return _price_cache.get(token)


def get_ltp_batch(symbols: list[str]) -> dict[str, float]:
    """
    Return {symbol: price} for each symbol found in the live cache.
    Mirrors the shape of kite.ltp() so callers can swap in easily.
    e.g. {"NSE:RELIANCE": {"last_price": 2847.5}, ...}
    """
    result = {}
    for raw in symbols:
        sym = raw.upper().strip()
        if not sym.startswith("NSE:"):
            sym = "NSE:" + sym
        tok   = _sym_to_tok.get(sym)
        price = _price_cache.get(tok) if tok else None
        if price is not None:
            result[sym] = {"last_price": price}
    return result


def status() -> dict:
    """Snapshot for /ticker/status endpoint."""
    return {
        "connected":        _connected,
        "subscribed_count": len(_subscribed),
        "cache_size":       len(_price_cache),
        "last_tick":        _last_tick_ts,
        "tick_count":       _tick_count,
        "token_set":        bool(_access_token),
    }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _do_subscribe(tokens: list[int]) -> int:
    """Subscribe new tokens to the running ticker. Returns new-token count."""
    with _sub_lock:
        new_tokens = [t for t in tokens if t not in _subscribed]
        if not new_tokens:
            return 0
        _subscribed.update(new_tokens)

    if _ticker and _connected:
        try:
            _ticker.subscribe(new_tokens)
            _ticker.set_mode(_ticker.MODE_LTP, new_tokens)
            log.info(f"[ticker] Subscribed {len(new_tokens)} token(s) — "
                     f"total: {len(_subscribed)}")
        except Exception as e:
            log.warning(f"[ticker] Subscribe failed: {e}")
    return len(new_tokens)


def _resolve_nse_token(symbol: str) -> Optional[int]:
    """
    Look up the Kite instrument_token for an NSE equity/index symbol.
    Uses the live instruments list from the authenticated kite client.
    Caches the result in _sym_to_tok / _tok_to_sym for future calls.
    """
    if not _kite:
        return None
    bare = symbol.replace("NSE:", "")
    try:
        instruments = _kite.instruments("NSE")
        for inst in instruments:
            if inst["tradingsymbol"] == bare:
                # Prefer EQ type but accept any match
                tok = int(inst["instrument_token"])
                _sym_to_tok[symbol] = tok
                _tok_to_sym[tok]    = symbol
                return tok
    except Exception as e:
        log.warning(f"[ticker] Token lookup failed for {symbol}: {e}")
    return None


# ─── KiteTicker callbacks ─────────────────────────────────────────────────────

def _on_ticks(ws, ticks):
    global _last_tick_ts, _tick_count
    for tick in ticks:
        tok   = tick.get("instrument_token")
        price = tick.get("last_price", 0)
        if tok and price and price > 0:
            _price_cache[tok] = float(price)
    _tick_count  += len(ticks)
    _last_tick_ts = datetime.now(_IST).strftime("%H:%M:%S")


def _on_connect(ws, response):
    global _connected
    _connected = True
    log.info("[ticker] WebSocket connected to Kite")
    # Re-subscribe all tokens (needed after a reconnect)
    with _sub_lock:
        tokens = list(_subscribed)
    if tokens:
        try:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_LTP, tokens)
            log.info(f"[ticker] Re-subscribed {len(tokens)} token(s) after connect")
        except Exception as e:
            log.warning(f"[ticker] Re-subscribe failed: {e}")


def _on_close(ws, code, reason):
    global _connected
    _connected = False
    log.warning(f"[ticker] WebSocket closed: {code} — {reason}")


def _on_error(ws, code, reason):
    global _connected
    _connected = False
    log.warning(f"[ticker] WebSocket error: {code} — {reason}")


def _on_reconnect(ws, attempts):
    log.info(f"[ticker] Reconnecting… attempt {attempts}")


def _on_noreconnect(ws):
    global _connected
    _connected = False
    log.error("[ticker] Max reconnect attempts reached — ticker offline")


# ─── Connection bootstrap ─────────────────────────────────────────────────────

def _connect():
    """Tear down any existing ticker and start a fresh WebSocket connection."""
    global _ticker, _connected

    # Tear down existing connection cleanly
    if _ticker:
        try:
            _ticker.close()
        except Exception:
            pass
        _ticker    = None
        _connected = False

    try:
        from kiteconnect import KiteTicker
    except ImportError:
        log.error("[ticker] kiteconnect package not found — WebSocket unavailable")
        return

    _ticker                 = KiteTicker(_api_key, _access_token)
    _ticker.on_ticks        = _on_ticks
    _ticker.on_connect      = _on_connect
    _ticker.on_close        = _on_close
    _ticker.on_error        = _on_error
    _ticker.on_reconnect    = _on_reconnect
    _ticker.on_noreconnect  = _on_noreconnect

    log.info("[ticker] Launching KiteTicker WebSocket (threaded)…")
    # threaded=True runs the WebSocket event loop in its own daemon thread
    _ticker.connect(threaded=True)
