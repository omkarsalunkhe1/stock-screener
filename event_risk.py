"""
event_risk.py — event-risk screens for the swing screener.

Two checks, both fail-open (missing data never blocks a scan, it only
disables the screen for that cycle):

1. NSE surveillance lists (ASM / GSM), cached 12h.
   Stocks under Additional or Graded Surveillance carry 100% margin and
   tight circuit limits — exits get trapped, which is fatal for a
   1-2 week swing mandate.

2. Next earnings date via yfinance, cached 24h per symbol.
   A hold that straddles a results day is a coin flip on gap risk,
   not a technical trade.
"""

import time
import logging
from datetime import date, datetime

log = logging.getLogger(__name__)

# ── NSE surveillance (ASM/GSM) ─────────────────────────────────────────────

_surv_cache = {"ts": 0.0, "sets": None}
_SURV_TTL_OK   = 12 * 3600
_SURV_TTL_FAIL = 900          # retry sooner after a failed fetch

_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def _collect_symbols(node, out: set):
    """Recursively harvest every "symbol" value from an NSE JSON payload —
    the report shapes change occasionally, so don't trust exact paths."""
    if isinstance(node, dict):
        sym = node.get("symbol")
        if isinstance(sym, str):
            out.add(sym.strip().upper())
        for v in node.values():
            _collect_symbols(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_symbols(v, out)


def _fetch_surveillance():
    try:
        import requests
    except ImportError:
        log.warning("requests not installed — surveillance screen disabled")
        return None
    try:
        s = requests.Session()
        s.headers.update(_NSE_HEADERS)
        s.get("https://www.nseindia.com", timeout=10)   # cookie bootstrap
        asm = s.get("https://www.nseindia.com/api/reportASM?json=true",
                    timeout=10).json()
        gsm = s.get("https://www.nseindia.com/api/reportGSM?json=true",
                    timeout=10).json()
        asm_lt, asm_st, gsm_set = set(), set(), set()
        _collect_symbols((asm or {}).get("longterm"),  asm_lt)
        _collect_symbols((asm or {}).get("shortterm"), asm_st)
        if not asm_lt and not asm_st:   # unknown payload shape — lump together
            _collect_symbols(asm, asm_lt)
        _collect_symbols(gsm, gsm_set)
        log.info(f"NSE surveillance lists loaded: {len(asm_lt)} ASM-LT, "
                 f"{len(asm_st)} ASM-ST, {len(gsm_set)} GSM")
        return {"asm_lt": asm_lt, "asm_st": asm_st, "gsm": gsm_set}
    except Exception as e:
        log.warning(f"NSE surveillance fetch failed (screen off this cycle): {e}")
        return None


def surveillance_label(symbol: str):
    """Return "GSM" / "ASM (long-term)" / "ASM (short-term)" or None."""
    now = time.time()
    ttl = _SURV_TTL_OK if _surv_cache["sets"] is not None else _SURV_TTL_FAIL
    if now - _surv_cache["ts"] >= ttl:
        _surv_cache["sets"] = _fetch_surveillance()
        _surv_cache["ts"] = now
    sets = _surv_cache["sets"]
    if not sets:
        return None
    sym = symbol.strip().upper()
    if sym in sets["gsm"]:
        return "GSM"
    if sym in sets["asm_lt"]:
        return "ASM (long-term)"
    if sym in sets["asm_st"]:
        return "ASM (short-term)"
    return None


# ── Earnings dates ─────────────────────────────────────────────────────────

_earn_cache: dict = {}        # symbol -> {"ts": float, "date": iso str | None}
_EARN_TTL = 24 * 3600


def _next_earnings_date(symbol: str):
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        cal = yf.Ticker(symbol + ".NS").calendar
    except Exception:
        return None

    raw = []
    if isinstance(cal, dict):
        raw = cal.get("Earnings Date") or []
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
    elif cal is not None and getattr(cal, "empty", True) is False:
        try:
            raw = list(cal.loc["Earnings Date"])   # legacy DataFrame shape
        except Exception:
            raw = []

    today = date.today()
    future = []
    for d in raw:
        if isinstance(d, datetime):
            d = d.date()
        elif isinstance(d, str):
            try:
                d = date.fromisoformat(d[:10])
            except ValueError:
                continue
        if isinstance(d, date) and d >= today:
            future.append(d)
    return min(future).isoformat() if future else None


def earnings_info(symbol: str) -> dict:
    """{"date": "YYYY-MM-DD" | None, "days_to": int | None} — cached 24h.

    days_to is calendar days from today; None when no upcoming date is
    published (common for NSE names on Yahoo — treat as unknown, not safe).
    """
    now = time.time()
    hit = _earn_cache.get(symbol)
    if hit is None or now - hit["ts"] >= _EARN_TTL:
        hit = {"ts": now, "date": _next_earnings_date(symbol)}
        _earn_cache[symbol] = hit
    days_to = None
    if hit["date"]:
        days_to = (date.fromisoformat(hit["date"]) - date.today()).days
    return {"date": hit["date"], "days_to": days_to}
