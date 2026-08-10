"""
FastAPI server — exposes screener as REST API
Run:  uvicorn server:app --reload --port 8000
"""

import os
import hashlib
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import threading
import uuid
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from screener import (
    SwingScreener, UNIVERSES, to_json,
    calc_rsi, calc_macd, calc_bollinger, calc_adx, calc_volume_ratio,
    calc_momentum, calc_support_resistance, calc_atr, calc_weekly_trend,
    detect_price_action, score_stock,
    calc_52w, calc_golden_cross, calc_weekly_vol_ratio,
)
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from datetime import time as dtime
from pathlib import Path
import fundamentals as _fundamentals
import news_sentiment as _news_sentiment
import swing_paper_broker as _spb
import kite_ticker as _kt
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

import sys as _sys
import subprocess as _subprocess
import httpx as _httpx
import rag_engine as _rag

SYMBOLS_CACHE_FILE = Path(__file__).parent / "symbols_cache.json"
SYMBOLS_CACHE_TTL  = 86400  # 24 hours
KITE_CONFIG_FILE   = Path(__file__).parent / "kite_config.json"
IST                = timezone(timedelta(hours=5, minutes=30))

SECTOR_INDEX_MAP: dict = {
    "nifty_bank":              "NSE:NIFTY BANK",
    "nifty_it":                "NSE:NIFTY IT",
    "nifty_auto":              "NSE:NIFTY AUTO",
    "nifty_pharma":            "NSE:NIFTY PHARMA",
    "nifty_fmcg":              "NSE:NIFTY FMCG",
    "nifty_financial":         "NSE:NIFTY FIN SERVICE",
    "nifty_metal":             "NSE:NIFTY METAL",
    "nifty_realty":            "NSE:NIFTY REALTY",
    "nifty_psu_bank":          "NSE:NIFTY PSU BANK",
    "nifty_energy":            "NSE:NIFTY ENERGY",
    "nifty_infra":             "NSE:NIFTY INFRA",
    "nifty_media":             "NSE:NIFTY MEDIA",
    "nifty_healthcare":        "NSE:NIFTY HEALTHCARE",
    "nifty_consumer_durables": "NSE:NIFTY CONSR DURBL",
    "nifty_chemicals":         "NSE:NIFTY CHEMICALS",
}
_sector_perf_cache: dict = {}
SECTOR_PERF_TTL = 300  # 5 minutes

log = logging.getLogger(__name__)

app = FastAPI(title="Swing Screener API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY        = os.environ.get("KITE_API_KEY", "")
API_SECRET     = os.environ.get("KITE_API_SECRET", "")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_URL     = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "phi4-mini")

_screener_cache: dict = {}

# ---------------------------------------------------------------------------
# Background scan job state
# ---------------------------------------------------------------------------

@dataclass
class ScanJob:
    job_id:         str
    status:         str = "running"   # running | paused | done | stopped | error
    total:          int = 0
    liquid:         int = 0
    processed:      int = 0
    qualified:      int = 0
    current_symbol: str = ""
    results:        list = field(default_factory=list)
    error:          str = ""
    _stop_event:    threading.Event = field(default_factory=threading.Event)
    _pause_event:   threading.Event = field(default_factory=threading.Event)

_scan_jobs: dict[str, ScanJob] = {}


def get_screener(access_token: str) -> SwingScreener:
    if access_token not in _screener_cache:
        sc = SwingScreener(api_key=API_KEY, access_token=access_token)
        sc.load_instruments()
        _screener_cache[access_token] = sc
        # Start WebSocket ticker with the freshly authenticated client
        _kt.start(API_KEY, access_token, sc.kite)
        log.info(f"[ticker] Started for new session (token …{access_token[-6:]})")
    return _screener_cache[access_token]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _save_access_token(access_token: str, user_name: str = "") -> None:
    cfg = {}
    if KITE_CONFIG_FILE.exists():
        try:
            cfg = json.loads(KITE_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg["api_key"] = cfg.get("api_key") or API_KEY
    cfg["api_secret"] = cfg.get("api_secret") or API_SECRET
    cfg["access_token"] = access_token
    if user_name:
        cfg["user_name"] = user_name
    cfg["access_token_updated_at"] = datetime.now(IST).isoformat(timespec="seconds")
    KITE_CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _load_kite_config() -> dict:
    if not KITE_CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(KITE_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _kite_api_key() -> str:
    cfg = _load_kite_config()
    return (API_KEY or str(cfg.get("api_key") or "")).strip()


def _kite_client_for_token(access_token: str):
    token = (access_token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Kite access token is required")
    api_key = _kite_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="Kite api_key is not configured")
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token)
    return kite


@app.get("/auth/saved-token")
def saved_token(request: Request):
    """Return the saved Kite access token for the local browser UI."""
    client_host = (request.client.host if request.client else "") or ""
    local_hosts = {"127.0.0.1", "::1", "localhost"}
    if client_host not in local_hosts and not client_host.startswith("127."):
        raise HTTPException(status_code=403, detail="saved token can only be read from localhost")

    cfg = _load_kite_config()
    token = str(cfg.get("access_token") or "").strip()
    return {
        "has_token": bool(token),
        "access_token": token,
        "user_name": cfg.get("user_name", ""),
        "updated_at": cfg.get("access_token_updated_at", ""),
    }


SCREENER_HTML_FILE = Path(__file__).parent / "swing_trade_screener.html"


@app.get("/")
def kite_callback(request_token: str = "", action: str = "", type: str = "", status: str = ""):
    if not request_token:
        # No OAuth redirect in progress — serve the screener UI.
        if SCREENER_HTML_FILE.exists():
            return HTMLResponse(SCREENER_HTML_FILE.read_text(encoding="utf-8"))
        raise HTTPException(404, "Not Found")
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=API_KEY)
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        user_name = data.get("user_name", "")
        _save_access_token(access_token, user_name)
        _screener_cache.clear()
        return HTMLResponse(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Kite login successful</title>"
            "<style>body{background:#101010;color:#f3f3f3;font-family:Arial,sans-serif;padding:32px}"
            ".ok{color:#39ce7a}.muted{color:#aaa}</style></head><body>"
            "<h2 class='ok'>Kite login successful</h2>"
            "<p>Access token saved for the Kite and Options apps.</p>"
            "<p class='muted'>Opening Options app...</p>"
            "<script>setTimeout(()=>{ location.replace('http://127.0.0.1:8010/') }, 900);</script>"
            "</body></html>"
        )
    except Exception as e:
        return HTMLResponse(
            "<h2>Kite login failed</h2>"
            f"<p>{str(e)}</p>"
            "<p>Request tokens are one-time and expire quickly. Please login again from the Options app.</p>",
            status_code=400,
        )

@app.get("/auth/login-url")
def login_url():
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=API_KEY)
    return {"url": kite.login_url()}


class TokenRequest(BaseModel):
    request_token: str


@app.post("/auth/token")
def exchange_token(req: TokenRequest):
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=API_KEY)
    try:
        data = kite.generate_session(req.request_token, api_secret=API_SECRET)
        _save_access_token(data["access_token"], data.get("user_name", ""))
        return {
            "access_token": data["access_token"],
            "user_name": data.get("user_name", ""),
            "email": data.get("email", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

@app.get("/screen")
def screen(
    access_token: str  = Query(...),
    universe: str      = Query("nifty50"),
    custom_symbols: str = Query("", description="Comma-separated symbols when universe=custom"),
    rsi_min: float     = Query(45,  ge=20, le=60),
    rsi_max: float     = Query(68,  ge=55, le=85),
    min_vol_surge: float = Query(1.5, ge=1.0, le=5.0),
    min_score: int     = Query(60,  ge=0,  le=100),
    min_target_pct: float = Query(5.0, ge=1.0, le=20.0),
    weekly_filter: bool  = Query(False, description="Only show stocks with bullish weekly trend"),
    weekly_required: str = Query("bullish", description="bullish | not_bearish"),
    exclude_earnings_days: int = Query(10, ge=0, le=30, description="Reject stocks reporting earnings within N days (0 = off)"),
    exclude_surveillance: bool = Query(True, description="Reject stocks on NSE ASM/GSM surveillance lists"),
    headroom_check: bool = Query(True, description="Cap targets at the nearest resistance (realistic 2-3 week move) instead of raw 3x ATR"),
    min_turnover_cr: float = Query(5.0, ge=0, le=100, description="Min 20-day avg daily turnover in ₹ crore (0 = off)"),
    rs_min: float = Query(0.0, ge=-100, le=20, description="Min 20-day relative strength vs Nifty in pp (-100 = off)"),
    top_n: int         = Query(20,  ge=1,  le=50),
):
    if universe == "custom":
        symbols = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
        if not symbols:
            raise HTTPException(status_code=400, detail="custom_symbols is required when universe=custom")
    elif universe not in UNIVERSES:
        raise HTTPException(status_code=400, detail=f"Unknown universe: {universe}")
    else:
        symbols = None

    filters = {
        "rsi_min":              rsi_min,
        "rsi_max":              rsi_max,
        "min_vol_surge":        min_vol_surge,
        "min_score":            min_score,
        "min_target_pct":       min_target_pct,
        "weekly_trend_filter":  weekly_filter,
        "weekly_trend_required": weekly_required,
        "exclude_earnings_days": exclude_earnings_days,
        "exclude_surveillance":  exclude_surveillance,
        "headroom_check":        headroom_check,
        "min_turnover_cr":       min_turnover_cr,
        "rs_min":                rs_min if rs_min > -100 else None,
    }

    try:
        sc = get_screener(access_token)
        if symbols is not None:
            # Custom watchlist — show ALL selected stocks; bypass every filter so
            # user-chosen stocks are never silently dropped (defaults in analyse_stock
            # would still apply RSI/score/vol cuts if keys are simply absent).
            custom_filters = {
                "rsi_min":             0,
                "rsi_max":             100,
                "min_vol_surge":       0.0,
                "min_score":           0,
                "min_target_pct":      0.0,   # volatility gate off — never drop user picks
                "weekly_trend_filter": False,
                "exclude_earnings_days": 0,    # event screens flag, never drop user picks
                "exclude_surveillance":  False,
                "headroom_check":        True,   # capping only shapes the target —
                "min_turnover_cr":       0.0,    # with min_target_pct 0 nothing is dropped
                "rs_min":                None,   # never drop user picks on RS
            }
            results = []
            for sym in symbols:
                result = sc.analyse_stock(sym, filters=custom_filters)
                if result:
                    results.append(result)
                time.sleep(0.35)
            results.sort(key=lambda x: x["scoring"]["score"], reverse=True)
        else:
            results = sc.run(universe=universe, filters=filters)
        top = results[:top_n]
        return JSONResponse(content=json.loads(to_json(top)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/universes")
def list_universes():
    return {k: len(v) for k, v in UNIVERSES.items()}


@app.get("/analyse")
def analyse_symbol(
    access_token: str = Query(...),
    symbol: str       = Query(..., description="NSE trading symbol e.g. RELIANCE"),
):
    symbol = symbol.strip().upper()
    sc = get_screener(access_token)

    # Use _analyse_holding which runs full analysis without any filters
    try:
        analysis = _analyse_holding(sc, symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if analysis is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}. Check the symbol or try again.")

    # Get current price from indicators
    price = analysis["indicators"]["current_price"]
    return JSONResponse(content=json.loads(json.dumps({
        "symbol":       symbol,
        "sector":       "—",
        "price":        price,
        "indicators":   analysis["indicators"],
        "scoring":      analysis["scoring"],
        "weekly":       analysis["weekly"],
        "price_action": analysis["price_action"],
    }, default=float)))


def _analyse_holding(sc: SwingScreener, symbol: str) -> dict | None:
    """Run full technical analysis on a holding — no filters applied."""
    df = sc.fetch_ohlcv(symbol)
    if df is None or len(df) < 30:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    current_price = float(close.iloc[-1])   # fallback: last historical close

    # Override with live LTP so the displayed price matches the Kite app
    try:
        ltp_data = sc.kite.ltp([f"NSE:{symbol}"])
        live_px  = ltp_data.get(f"NSE:{symbol}", {}).get("last_price", 0)
        if live_px and live_px > 0:
            current_price = float(live_px)
    except Exception:
        pass   # market closed or API error — historical close is fine

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

    # Relative strength vs Nifty 50 — same convention as the scan path
    idx = sc._index_momentum()
    indicators["nifty_mom_5d"]  = idx["mom5"]
    indicators["nifty_mom_20d"] = idx["mom20"]
    indicators["rs_nifty_5d"] = (
        round(indicators["momentum_5d"] - idx["mom5"], 2)
        if idx["mom5"] is not None else None)
    indicators["rs_nifty_20d"] = (
        round(indicators["momentum_20d"] - idx["mom20"], 2)
        if idx["mom20"] is not None else None)

    scoring = score_stock(indicators, {})
    pa = detect_price_action(df)

    final_score = min(130, scoring["score"] + pa["pa_score"])
    scoring["base_score"] = scoring["score"]
    scoring["pa_score"] = pa["pa_score"]
    scoring["final_score"] = final_score
    scoring["score"] = final_score

    # Weekly trend
    time.sleep(0.25)
    weekly_df = sc.fetch_weekly_ohlcv(symbol)
    weekly_trend = calc_weekly_trend(weekly_df)

    return {
        "indicators": indicators,
        "scoring": scoring,
        "weekly": weekly_trend,
        "price_action": pa,
    }


@app.get("/portfolio")
def get_portfolio(access_token: str = Query(...)):
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(access_token)
    try:
        holdings = kite.holdings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sc = get_screener(access_token)

    processed = []
    total_invested = 0
    total_current = 0
    total_day_change = 0

    for i, h in enumerate(holdings):
        symbol = h.get("tradingsymbol", "")
        qty = h.get("quantity", 0)
        avg = h.get("average_price", 0)
        ltp = h.get("last_price", 0)
        invested = qty * avg
        current = qty * ltp
        pnl = current - invested
        pnl_pct = (pnl / invested * 100) if invested else 0
        day_chg = h.get("day_change", 0)
        day_chg_pct = h.get("day_change_percentage", 0)

        total_invested += invested
        total_current += current
        total_day_change += day_chg * qty if day_chg else 0

        holding_data = {
            "tradingsymbol": symbol,
            "exchange": h.get("exchange", ""),
            "quantity": qty,
            "average_price": round(avg, 2),
            "last_price": round(ltp, 2),
            "invested": round(invested, 2),
            "current": round(current, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "day_change": round(day_chg, 2),
            "day_change_pct": round(day_chg_pct, 2),
        }

        # Run full technical analysis
        log.info(f"  [{i+1}/{len(holdings)}] Analysing {symbol}...")
        try:
            analysis = _analyse_holding(sc, symbol)
            if analysis:
                holding_data["analysis"] = analysis
        except Exception as e:
            log.warning(f"  Analysis failed for {symbol}: {e}")

        time.sleep(0.35)
        processed.append(holding_data)

    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0

    return JSONResponse(content=json.loads(json.dumps({
        "holdings": processed,
        "summary": {
            "total_invested": round(total_invested, 2),
            "total_current": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "total_day_change": round(total_day_change, 2),
            "count": len(processed),
        }
    }, default=float)))


def _num_or_zero(value) -> float:
    try:
        return round(float(value or 0), 2)
    except Exception:
        return 0.0


def _normalise_margin_segment(segment: dict) -> dict:
    segment = segment or {}
    available = segment.get("available") or {}
    utilised = segment.get("utilised") or {}
    available_total = _num_or_zero(
        available.get("live_balance")
        or available.get("cash")
        or segment.get("net")
    )
    debits = _num_or_zero(utilised.get("debits"))
    used_total = debits if debits else round(sum(
        max(_num_or_zero(utilised.get(k)), 0)
        for k in ("span", "exposure", "option_premium", "delivery", "turnover")
    ), 2)
    return {
        "enabled": bool(segment),
        "net": _num_or_zero(segment.get("net")),
        "available_total": available_total,
        "used_total": used_total,
        "available": {k: _num_or_zero(v) for k, v in available.items()},
        "utilised": {k: _num_or_zero(v) for k, v in utilised.items()},
    }


@app.get("/funds")
def get_funds(access_token: str = Query(...)):
    """Return Kite funds/margins for the authenticated account."""
    kite = _kite_client_for_token(access_token)
    try:
        margins = kite.margins()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    equity = _normalise_margin_segment(margins.get("equity", {}))
    commodity = _normalise_margin_segment(margins.get("commodity", {}))
    return {
        "updated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "equity": equity,
        "commodity": commodity,
        "raw": margins,
    }


@app.get("/symbols")
def get_symbols():
    """Return all NSE EQ symbols with names. Cached locally for 24 hours."""
    # Serve from local cache if fresh
    if SYMBOLS_CACHE_FILE.exists():
        try:
            cached = json.loads(SYMBOLS_CACHE_FILE.read_text())
            if time.time() - cached.get("ts", 0) < SYMBOLS_CACHE_TTL:
                return {"symbols": cached["symbols"], "count": len(cached["symbols"]), "cached": True}
        except Exception:
            pass

    # Use an existing authenticated session's kite instance if available
    instruments = None
    if _screener_cache:
        try:
            sc = next(iter(_screener_cache.values()))
            instruments = sc.kite.instruments("NSE")
        except Exception:
            pass

    # Fall back to API-key-only fetch (instruments endpoint doesn't need access token)
    if instruments is None:
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=API_KEY)
            instruments = kite.instruments("NSE")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Cannot fetch symbols — connect to Kite first. ({e})")

    seen: set = set()
    symbols = []
    for inst in instruments:
        if inst.get("instrument_type") == "EQ":
            sym = inst["tradingsymbol"]
            if sym not in seen:
                seen.add(sym)
                symbols.append({"s": sym, "n": inst.get("name", "")})
    symbols.sort(key=lambda x: x["s"])

    SYMBOLS_CACHE_FILE.write_text(json.dumps({"ts": time.time(), "symbols": symbols}))
    log.info(f"Symbols cache updated: {len(symbols)} NSE EQ instruments")
    return {"symbols": symbols, "count": len(symbols), "cached": False}


@app.get("/sector-perf")
def sector_perf(access_token: str = Query(...)):
    """Return today's 1-day % change for each tracked sector index. Cached 5 min."""
    now = time.time()
    if _sector_perf_cache.get("ts", 0) + SECTOR_PERF_TTL > now:
        return JSONResponse(content=_sector_perf_cache["data"])

    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(access_token)

    symbols = list(SECTOR_INDEX_MAP.values())
    try:
        quote_data = kite.quote(symbols)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kite quote failed: {e}")

    result = {}
    for sector_key, instrument_key in SECTOR_INDEX_MAP.items():
        q          = quote_data.get(instrument_key, {})
        last_price = q.get("last_price", 0.0)
        prev_close = (q.get("ohlc") or {}).get("close", 0.0)
        chg_1d     = round((last_price / prev_close - 1) * 100, 2) if prev_close else 0.0
        result[sector_key] = {"chg_1d": chg_1d, "last_price": round(last_price, 2)}

    _sector_perf_cache["data"] = result
    _sector_perf_cache["ts"]   = now
    return JSONResponse(content=result)


@app.get("/fundamentals")
def get_fundamentals(
    symbol: str       = Query(..., description="NSE trading symbol e.g. RELIANCE"),
    access_token: str = Query("",  description="Access token (not required — yfinance does not use Kite)"),
):
    """
    Fetch fundamental data for a symbol via yfinance.
    Results are cached in-process for 24 hours.
    Returns P/E, P/B, EPS, ROE, D/E, FCF, Piotroski F-Score, Magic Formula, quarterly results.
    """
    symbol = symbol.strip().upper()
    try:
        data = _fundamentals.fetch(symbol)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news-sentiment")
def get_news_sentiment(
    symbol: str       = Query(..., description="NSE trading symbol e.g. RELIANCE"),
    access_token: str = Query("",  description="Not required — news APIs do not use Kite"),
):
    """Fetch latest news + sentiment for a symbol. Cached 1 hour."""
    symbol = symbol.strip().upper()
    try:
        data = _news_sentiment.fetch(symbol)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag-status")
def get_rag_status():
    """Check whether the RAG database has been built and is loaded."""
    ready = _rag.is_ready()
    db    = Path(__file__).parent / "rag_db"
    tb_count = 0
    if ready:
        try:
            import chromadb
            client   = chromadb.PersistentClient(path=str(db))
            tb_count = client.get_collection("trading_docs").count()
        except Exception:
            pass
    ar = _rag.ar_stats()
    return {
        "ready":            ready,
        "textbook_chunks":  tb_count,
        "ar_chunks":        ar["count"],
        "ar_companies":     ar["companies"],
        "db_path":          str(db),
        "hint":             "" if ready else "Run: python rag_ingest.py  to build the database",
    }


# ── Annual report ingest job state ───────────────────────────────────────────
_ingest_job: dict = {"status": "idle", "message": "", "ts": 0}


@app.post("/ingest-reports")
def ingest_reports():
    """
    Trigger a background rebuild of the annual_reports RAG collection
    from all PDFs in the reports/ folder.
    """
    if _ingest_job["status"] == "running":
        return JSONResponse(content={"status": "running", "message": "Ingest already in progress — please wait"})

    def _run():
        _ingest_job["status"]  = "running"
        _ingest_job["message"] = "Ingesting PDFs from reports/ …"
        _ingest_job["ts"]      = time.time()
        try:
            ingest_script = Path(__file__).parent / "rag_ingest.py"
            result = _subprocess.run(
                [_sys.executable, str(ingest_script), "--reports-only"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                _rag.reload_ar()   # hot-reload collection without server restart
                ar = _rag.ar_stats()
                companies = ", ".join(ar["companies"]) if ar["companies"] else "none"
                _ingest_job["status"]  = "done"
                _ingest_job["message"] = f"{ar['count']} chunks — companies: {companies}"
            else:
                _ingest_job["status"]  = "error"
                _ingest_job["message"] = (result.stderr or result.stdout or "Unknown error")[:400]
        except Exception as e:
            _ingest_job["status"]  = "error"
            _ingest_job["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse(content={"status": "started"})


@app.get("/ingest-reports/status")
def ingest_reports_status():
    """Poll ingest job progress."""
    return JSONResponse(content=_ingest_job)


@app.get("/local-models")
def get_local_models():
    """List models available in the local Ollama instance."""
    try:
        resp = _httpx.get(f"{OLLAMA_URL}/api/tags", timeout=4.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"available": True, "models": models, "default": OLLAMA_MODEL, "url": OLLAMA_URL}
    except Exception as e:
        return {"available": False, "models": [], "default": OLLAMA_MODEL,
                "url": OLLAMA_URL, "error": str(e)}


@app.get("/ai-analysis")
def get_ai_analysis(
    symbol:          str   = Query(...,  description="NSE trading symbol"),
    price:           float = Query(0),
    score:           int   = Query(0),
    signal:          str   = Query(""),
    rsi:             float = Query(0),
    volume_ratio:    float = Query(1),
    macd_bull:       bool  = Query(False),
    adx:             float = Query(0),
    momentum_5d:     float = Query(0),
    momentum_20d:    float = Query(0),
    ma20_dist_pct:   float = Query(0),
    weekly_trend:    str   = Query(""),
    price_action:    str   = Query(""),
    bearish_patterns:str   = Query(""),
    target_pct:      float = Query(0),
    sl_pct:          float = Query(0),
    provider:        str   = Query("claude"),      # "claude" | "local"
    local_model:     str   = Query(""),            # override OLLAMA_MODEL
    claude_model:    str   = Query(""),            # override default Claude model
):
    """
    Generate a swing-trade analysis using either Claude API or a local Ollama model.
    provider="claude"  → Anthropic Claude claude-opus-4-6 (requires ANTHROPIC_API_KEY)
    provider="local"   → Local Ollama model (requires Ollama running on localhost:11434)
    """
    if provider == "local":
        # ── guard: is Ollama reachable? ─────────────────────────────────────
        pass   # handled below in the try block
    else:
        if not _ANTHROPIC_AVAILABLE:
            raise HTTPException(status_code=503, detail="anthropic package not installed — run: pip install anthropic")
        if not ANTHROPIC_KEY:
            raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set in .env")

    symbol = symbol.strip().upper()
    sig_labels = {"strong_buy": "Strong Buy", "moderate": "Moderate", "watch": "Watch"}
    sig_label  = sig_labels.get(signal, signal or "—")

    # Derived contextual notes fed into the prompt
    rr_ratio   = round(target_pct / sl_pct, 1) if sl_pct > 0 else 0
    rsi_note   = ("oversold — potential reversal zone" if rsi < 32
                  else "overbought — momentum may stall" if rsi > 70
                  else "neutral range")
    adx_note   = ("strong trend" if adx > 30 else
                  "developing trend" if adx > 20 else
                  "trendless / ranging — patterns less reliable")
    vol_note   = ("high — pattern confirmation strong" if volume_ratio >= 1.5
                  else "below average — breakout/reversal NOT yet volume-confirmed" if volume_ratio < 0.9
                  else "average")
    ma_note    = ("extended above MA20 — late entry risk" if ma20_dist_pct > 8
                  else "near MA20 support" if -3 <= ma20_dist_pct <= 2
                  else "below MA20 — still in recovery" if ma20_dist_pct < -3
                  else "")
    nr4_active = "NR4 Squeeze" in (price_action or "")
    harami_active = "Bullish Harami" in (price_action or "")

    system_prompt = (
        "You are a disciplined swing trade analyst specialising in NSE-listed Indian equities. "
        "Your analysis is grounded in classical technical analysis: Dow Theory trend alignment, "
        "candlestick patterns that REQUIRE volume confirmation to be valid, "
        "and strict risk-first position sizing (minimum 1:2 risk-reward). "
        "Key rules you follow:\n"
        "• A candlestick pattern is only actionable after breakout/close confirmation WITH above-average volume.\n"
        "• NR4 Squeeze signals an imminent volatility expansion — direction unconfirmed until breakout.\n"
        "• Bullish Harami is a potential reversal but can break either direction — wait for next-candle confirmation.\n"
        "• Support/resistance are ZONES, not exact price levels.\n"
        "• RSI <30 = oversold reversal opportunity; RSI >70 = momentum risk.\n"
        "• ADX <20 = trendless market — continuation patterns have lower reliability.\n"
        "• If bearish patterns co-exist with bullish ones, they override — flag the conflict.\n"
    )

    data_lines = [
        f"Symbol:           {symbol}",
        f"Price:            ₹{price:,.2f}",
        f"Score:            {score}/100  ({sig_label})",
        f"RSI ({rsi:.1f}):       {rsi_note}",
        f"Volume ({volume_ratio:.1f}×):    {vol_note}",
        f"MACD:             {'Bullish crossover' if macd_bull else 'Bearish / no cross'}",
        f"ADX ({adx:.0f}):         {adx_note}",
        f"5D momentum:      {momentum_5d:+.1f}%",
        f"20D momentum:     {momentum_20d:+.1f}%",
        f"vs MA20:          {ma20_dist_pct:+.1f}%{('  ← ' + ma_note) if ma_note else ''}",
        f"R:R ratio:        1:{rr_ratio} (target +{target_pct:.1f}% / SL -{sl_pct:.1f}%)",
    ]
    if weekly_trend:
        data_lines.append(f"Weekly trend:     {weekly_trend}")
    if price_action:
        data_lines.append(f"Bullish patterns: {price_action}")
    if bearish_patterns:
        data_lines.append(f"⚠ Bearish signals: {bearish_patterns}")
    if nr4_active:
        data_lines.append("NR4 context:      Volatility squeeze — watch for directional breakout with volume surge")
    if harami_active:
        data_lines.append("Harami context:   Reversal candidate — requires next candle confirmation before entry")

    user_prompt = (
        "Based strictly on the data above, write exactly 3 sentences (plain text, no markdown):\n"
        "  1. Setup: primary trend context + the strongest confirmed signal (mention if volume-confirmed or not).\n"
        "  2. Trade logic: why this qualifies (or almost qualifies) as a swing entry, "
        f"     referencing the {rr_ratio}:1 R:R and key levels.\n"
        "  3. Key risk: the single most important caution — bearish pattern conflict, "
        "     unconfirmed volume, ADX weakness, RSI extreme, or MA extension.\n"
        "Be specific with numbers. Do not add disclaimers."
    )

    data_block = "\n".join(data_lines)

    # ── RAG: inject relevant textbook passages into the system prompt ────────
    rag_query   = _rag.build_rag_query(
        price_action     = price_action,
        bearish_patterns = bearish_patterns,
        rsi              = rsi,
        volume_ratio     = volume_ratio,
        macd_bull        = macd_bull,
        adx              = adx,
        weekly_trend     = weekly_trend,
        momentum_20d     = momentum_20d,
        ma20_dist_pct    = ma20_dist_pct,
    )
    rag_context = _rag.get_rag_context(rag_query, n=3, symbol=symbol)
    if rag_context:
        system_prompt += (
            "\n\nRelevant passages from your trading reference library "
            "(NSE TA Module, Fidelity Chart Patterns, Fundamentals of Investments). "
            "Use these to ground your analysis in the exact textbook definitions:\n\n"
            + rag_context
        )

    prompt = f"{user_prompt}\n\n{data_block}"

    # ── Route to the chosen provider ────────────────────────────────────────
    if provider == "local":
        model_name = (local_model.strip() or OLLAMA_MODEL)
        try:
            resp = _httpx.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model":    model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500,
                        "num_ctx":     4096,   # cap context window — phi4-mini default is 128k (needs 16 GB RAM)
                        "stop": ["\n\n\n"],    # prevent rambling
                    },
                },
                timeout=120.0,   # local inference can be slow on CPU
            )
            resp.raise_for_status()
            data = resp.json()
            analysis_text = data["message"]["content"].strip()
            return JSONResponse(content={
                "symbol":   symbol,
                "analysis": analysis_text,
                "model":    f"local/{model_name}",
                "error":    None,
            })
        except _httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama not running at {OLLAMA_URL} — start it with: ollama serve"
            )
        except _httpx.HTTPStatusError as e:
            detail = e.response.text[:400] if e.response else str(e)
            dl = detail.lower()
            if "model" in dl and "not found" in dl:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_name}' not pulled. Run: ollama pull {model_name}"
                )
            if "more system memory" in dl or "not enough memory" in dl or "out of memory" in dl:
                raise HTTPException(
                    status_code=507,
                    detail=(
                        f"Model '{model_name}' needs more RAM than your system has available. "
                        f"Switch to a smaller model — try: phi4-mini, llama3.2:3b, or qwen2.5:3b "
                        f"(run: ollama pull phi4-mini)"
                    )
                )
            raise HTTPException(status_code=500, detail=detail)
        except Exception as e:
            err = str(e)
            if "more system memory" in err.lower() or "out of memory" in err.lower():
                raise HTTPException(
                    status_code=507,
                    detail=(
                        f"Model '{model_name}' needs more RAM than your system has available. "
                        f"Switch to a smaller model — try: phi4-mini, llama3.2:3b, or qwen2.5:3b"
                    )
                )
            log.warning(f"[ai/local] Ollama analysis failed for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=err)

    else:  # provider == "claude"
        _CLAUDE_MODELS = {
            "opus":   "claude-opus-4-6",
            "sonnet": "claude-sonnet-4-5",
            "haiku":  "claude-haiku-3-5",
        }
        # Accept short alias (opus/sonnet/haiku) or full model name
        _req = claude_model.strip().lower()
        chosen_claude = (
            _CLAUDE_MODELS.get(_req)
            or (_req if _req.startswith("claude-") else None)
            or "claude-opus-4-6"
        )
        try:
            client  = _anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            message = client.messages.create(
                model      = chosen_claude,
                max_tokens = 420,
                system     = system_prompt,
                messages   = [{"role": "user", "content": prompt}],
            )
            analysis_text = message.content[0].text.strip()
            return JSONResponse(content={
                "symbol":   symbol,
                "analysis": analysis_text,
                "model":    chosen_claude,
                "error":    None,
            })
        except Exception as e:
            log.warning(f"[ai] Claude analysis failed for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate-token")
def validate_token(access_token: str = Query(...)):
    """
    Verify an access token by calling kite.profile().
    Returns 200 + user info on success, 401 on invalid/expired token.
    """
    from kiteconnect import KiteConnect
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(access_token)
        profile = kite.profile()
        return {
            "valid":     True,
            "user_name": profile.get("user_name", ""),
            "email":     profile.get("email", ""),
            "user_id":   profile.get("user_id", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Paper trading — swing screener
# ---------------------------------------------------------------------------

class PaperBuyRequest(BaseModel):
    symbol:     str
    qty:        int
    price:      float
    target_pct: float
    sl_pct:     float
    signal:     str = ""
    score:      int = 0
    mode:       str = "demo"   # "live" if taken against live Kite data, else "demo"


SWING_MODERATE_CAPITAL_CAP = 15_000
SWING_LOW_SCORE_CAPITAL_CAP = 10_000


def _paper_buy_cap(req: PaperBuyRequest) -> tuple[int | None, str]:
    """Return max paper capital for weaker swing setups."""
    signal = (req.signal or "").lower()
    score = int(req.score or 0)
    if score < 70:
        return SWING_LOW_SCORE_CAPITAL_CAP, "score below 70"
    if signal != "strong_buy" or score < 85:
        return SWING_MODERATE_CAPITAL_CAP, "moderate/non-strong setup"
    return None, ""


@app.get("/paper/positions")
def paper_positions():
    """Return all open + closed paper trades, plus summary stats."""
    state = _spb.list_positions()
    stats = _spb.summary(state)
    return JSONResponse(content={**state, "summary": stats})


@app.post("/paper/buy")
def paper_buy(req: PaperBuyRequest):
    """Open a new paper position for an equity swing trade."""
    if req.qty <= 0:
        raise HTTPException(status_code=400, detail="qty must be > 0")
    if req.price <= 0:
        raise HTTPException(status_code=400, detail="price must be > 0")
    cap, reason = _paper_buy_cap(req)
    state = _spb.list_positions()
    existing_trades = [
        t for t in state.get("open", [])
        if t.get("symbol", "").upper() == req.symbol.upper()
    ]
    existing_capital = sum(
        float(t.get("entry_price") or 0) * int(t.get("qty") or 0)
        for t in existing_trades
    )
    capital = req.qty * req.price
    total_capital = existing_capital + capital
    if cap is not None and total_capital > cap:
        max_qty = int(max(0, cap - existing_capital) // req.price)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Position size capped for {reason}: max paper capital ₹{cap:,}. "
                f"Existing ₹{existing_capital:,.0f}, new ₹{capital:,.0f}, "
                f"total ₹{total_capital:,.0f}. "
                f"Reduce quantity to {max_qty} share(s) or wait for a stronger setup."
            ),
        )
    trade = _spb.buy(
        symbol      = req.symbol,
        qty         = req.qty,
        entry_price = req.price,
        target_pct  = req.target_pct,
        sl_pct      = req.sl_pct,
        signal      = req.signal,
        score       = req.score,
        mode        = req.mode,
    )
    # Subscribe symbol to the live ticker so SL/target checks get real-time prices
    if _kt.is_connected():
        _kt.subscribe_symbols([req.symbol])
    return JSONResponse(content=trade)


@app.post("/paper/close/{trade_id}")
def paper_close(
    trade_id:     str,
    exit_price:   float = Query(0),
    access_token: str   = Query(""),
):
    """
    Manually close an open paper position.
    If exit_price is 0, attempts to fetch live LTP via Kite using access_token.
    Falls back to last known LTP stored in the trade.
    """
    state = _spb.list_positions()
    trade = next((t for t in state["open"] if t["id"] == trade_id), None)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found in open positions")

    # Determine exit price
    price = float(exit_price) if exit_price > 0 else 0.0
    if price == 0 and access_token:
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=API_KEY)
            kite.set_access_token(access_token)
            ltp_data = kite.ltp([f"NSE:{trade['symbol']}"])
            price = float((ltp_data.get(f"NSE:{trade['symbol']}") or {}).get("last_price", 0))
        except Exception:
            pass
    if price == 0:
        price = float(trade.get("ltp") or trade["entry_price"])

    closed = _spb.close_position(trade_id, price, "MANUAL")
    if not closed:
        raise HTTPException(status_code=404, detail="Trade not found")
    return JSONResponse(content=closed)


@app.post("/paper/refresh")
def paper_refresh(access_token: str = Query(...)):
    """
    Refresh LTPs for all open paper positions via Kite.
    Auto-closes any position that hit its target or SL.
    """
    from kiteconnect import KiteConnect
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(access_token)
        result = _spb.refresh_ltps(kite)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LTP refresh failed: {e}")
    state = _spb.list_positions()
    stats = _spb.summary(state)
    return JSONResponse(content={
        "auto_closed": result["auto_closed"],
        "updated":     result["updated"],
        "summary":     stats,
        **state,
    })


@app.post("/paper/reset")
def paper_reset():
    """Wipe all paper trades (irreversible)."""
    _spb.reset()
    return {"message": "Paper trades cleared"}


@app.get("/paper/poll-status")
def paper_poll_status():
    """Return the current state of the server-side background LTP watcher."""
    return JSONResponse(content={
        "running":           _paper_poll["running"],
        "last_check":        _paper_poll["last_check"],
        "last_auto_close":   _paper_poll["last_auto_close"],
        "auto_closed_today": _paper_poll["auto_closed_today"][-10:],
        "market_open":       _is_market_open(),
        "ticker":            _kt.status(),
    })


@app.get("/ticker/status")
def ticker_status():
    """Return Kite WebSocket ticker connection status."""
    return JSONResponse(content=_kt.status())


# ---------------------------------------------------------------------------
# Server-side background paper-position watcher
# Runs every 60 s during market hours (Mon–Fri 09:15–15:30 IST).
# Borrows the first authenticated Kite client from the screener cache.
# ---------------------------------------------------------------------------

_IST_TZ = IST

_paper_poll: dict = {
    "running":         False,
    "last_check":      None,   # ISO timestamp of last LTP fetch
    "last_auto_close": None,   # ISO timestamp of most recent auto-exit
    "auto_closed_today": [],   # accumulates within a calendar day
}


def _is_market_open() -> bool:
    """True if NSE is currently open (Mon–Fri 09:15–15:30 IST)."""
    now = datetime.now(_IST_TZ)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(9, 15) <= t <= dtime(15, 30)


def _paper_poll_worker():
    """
    Daemon thread — checks paper position SL/targets during market hours.

    Strategy:
      • Ticker connected  → read live cache every 5 s  (near real-time)
      • Ticker offline    → fall back to kite.ltp() REST every 60 s
    This means SL/target hits are caught within ~5 s when the WebSocket
    is live, or within ~60 s when it has to fall back to polling.
    """
    _paper_poll["running"] = True
    log.info("[paper-poll] Background SL/target watcher started")

    while True:
        # Sleep 5 s when ticker is live, 60 s when falling back to REST
        sleep_secs = 5 if _kt.is_connected() else 60
        time.sleep(sleep_secs)

        if not _is_market_open():
            continue

        # Skip if no open positions
        state = _spb.list_positions()
        if not state.get("open"):
            continue

        try:
            if _kt.is_connected():
                # ── Fast path: read from WebSocket cache ─────────────────
                result = _check_positions_from_ticker(state["open"])
            else:
                # ── Slow path: REST fallback ──────────────────────────────
                kite = None
                for sc in list(_screener_cache.values()):
                    try:
                        kite = sc.kite
                        break
                    except Exception:
                        pass
                if kite is None:
                    continue
                result = _spb.refresh_ltps(kite)

            now_ts = datetime.now(_IST_TZ).isoformat(timespec="seconds")
            _paper_poll["last_check"] = now_ts

            # Reset today's auto-close list at midnight
            today = datetime.now(_IST_TZ).date().isoformat()
            if _paper_poll["auto_closed_today"]:
                last_ts = (_paper_poll["auto_closed_today"][-1].get("exit_ts") or "")[:10]
                if last_ts != today:
                    _paper_poll["auto_closed_today"] = []

            if result["auto_closed"]:
                _paper_poll["last_auto_close"] = now_ts
                _paper_poll["auto_closed_today"].extend(result["auto_closed"])
                for t in result["auto_closed"]:
                    log.info(
                        f"[paper-poll] AUTO-EXIT {t['symbol']} × {t['qty']} "
                        f"@ ₹{t['exit_price']} ({t['exit_reason']}) "
                        f"— net ₹{t['net_pnl']} "
                        f"[{'ticker' if _kt.is_connected() else 'REST'}]"
                    )
        except Exception as e:
            log.warning(f"[paper-poll] Check failed: {e}")


def _check_positions_from_ticker(open_positions: list[dict]) -> dict:
    """
    Read LTPs from the WebSocket cache and run SL/target logic.
    Returns same shape as swing_paper_broker.refresh_ltps():
      {"auto_closed": [...], "updated": [...]}

    Also subscribes any position symbols not yet in the ticker.
    """
    # Ensure all open position symbols are subscribed to the ticker
    symbols = [p["symbol"] for p in open_positions]
    _kt.subscribe_symbols(symbols)

    from datetime import datetime as _dt
    now_ts      = _dt.now(_IST_TZ).isoformat(timespec="seconds")
    auto_closed: list[dict] = []
    still_open:  list[dict] = []

    # Read the full JSON state so we can write it back in one shot
    import swing_paper_broker as _spb_inner
    state = _spb_inner._load()

    for t in state["open"]:
        price = _kt.get_ltp(t["symbol"])
        if price and price > 0:
            t["ltp"]    = round(float(price), 2)
            t["ltp_ts"] = now_ts

        ltp = t["ltp"]
        if ltp >= t["target_price"]:
            _spb_inner._do_close(t, ltp, "TARGET")
            state["closed"].insert(0, t)
            auto_closed.append(t)
        elif ltp <= t["sl_price"]:
            _spb_inner._do_close(t, ltp, "SL")
            state["closed"].insert(0, t)
            auto_closed.append(t)
        else:
            still_open.append(t)

    state["open"] = still_open
    _spb_inner._save(state)
    return {"auto_closed": auto_closed, "updated": still_open}


# Start the watcher as a daemon thread when the server loads
threading.Thread(target=_paper_poll_worker, daemon=True, name="paper-poll").start()


# ---------------------------------------------------------------------------
# Full NSE background scan
# ---------------------------------------------------------------------------

def _prefilter_liquid_symbols(sc: SwingScreener, symbols: list[str], min_ltp: float = 20.0) -> list[str]:
    """Batch LTP fetch (500 per call) and return symbols with LTP >= min_ltp."""
    liquid = []
    batch_size = 500
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        instruments = [f"NSE:{s}" for s in batch]
        try:
            ltp_data = sc.kite.ltp(instruments)
            for sym in batch:
                key = f"NSE:{sym}"
                price = ltp_data.get(key, {}).get("last_price", 0)
                if price >= min_ltp:
                    liquid.append(sym)
        except Exception as e:
            log.warning(f"LTP batch failed for offset {i}: {e}")
            # On error keep the whole batch to avoid missing stocks
            liquid.extend(batch)
        time.sleep(0.3)
    return liquid


def _run_full_scan(job: ScanJob, sc: SwingScreener, symbols: list[str], filters: dict):
    """Background thread: pre-filter then analyse each liquid stock."""
    try:
        log.info(f"[scan:{job.job_id[:8]}] Pre-filtering {len(symbols)} symbols...")
        liquid = _prefilter_liquid_symbols(sc, symbols)
        job.liquid = len(liquid)
        job.total = len(liquid)
        log.info(f"[scan:{job.job_id[:8]}] {len(liquid)} liquid stocks — starting analysis")

        for sym in liquid:
            if job._stop_event.is_set():
                job.status = "stopped"
                log.info(f"[scan:{job.job_id[:8]}] Stopped at {job.processed}/{job.total}")
                return

            # Pause — block here until resumed or stopped
            if job._pause_event.is_set():
                job.status = "paused"
                log.info(f"[scan:{job.job_id[:8]}] Paused at {job.processed}/{job.total}")
                while job._pause_event.is_set():
                    if job._stop_event.is_set():
                        job.status = "stopped"
                        return
                    time.sleep(0.4)
                job.status = "running"
                log.info(f"[scan:{job.job_id[:8]}] Resumed at {job.processed}/{job.total}")

            job.current_symbol = sym
            try:
                result = sc.analyse_stock(sym, filters=filters)
                if result:
                    job.results.append(result)
                    job.qualified += 1
            except Exception as e:
                log.debug(f"[scan:{job.job_id[:8]}] {sym} skipped: {e}")
            job.processed += 1
            time.sleep(0.35)

        job.results.sort(key=lambda x: x["scoring"]["score"], reverse=True)
        job.status = "done"
        log.info(f"[scan:{job.job_id[:8]}] Done — {job.qualified} qualified from {job.processed} analysed")
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        log.exception(f"[scan:{job.job_id[:8]}] Fatal error: {e}")


@app.post("/scan/start")
def scan_start(
    access_token: str   = Query(...),
    rsi_min: float      = Query(45,  ge=20, le=60),
    rsi_max: float      = Query(68,  ge=55, le=85),
    min_vol_surge: float = Query(1.5, ge=1.0, le=5.0),
    min_score: int      = Query(60,  ge=0,  le=100),
    min_target_pct: float = Query(5.0, ge=1.0, le=20.0),
    weekly_filter: bool  = Query(False),
    weekly_required: str = Query("bullish"),
):
    # Load symbols from cache
    if not SYMBOLS_CACHE_FILE.exists():
        raise HTTPException(status_code=503, detail="symbols_cache.json not found — call /symbols first")
    try:
        cached = json.loads(SYMBOLS_CACHE_FILE.read_text())
        symbols = [item["s"] for item in cached.get("symbols", [])]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot read symbols cache: {e}")

    if not symbols:
        raise HTTPException(status_code=503, detail="Symbols cache is empty")

    sc = get_screener(access_token)
    filters = {
        "rsi_min":               rsi_min,
        "rsi_max":               rsi_max,
        "min_vol_surge":         min_vol_surge,
        "min_score":             min_score,
        "min_target_pct":        min_target_pct,
        "weekly_trend_filter":   weekly_filter,
        "weekly_trend_required": weekly_required,
    }

    job = ScanJob(job_id=str(uuid.uuid4()), total=len(symbols))
    _scan_jobs[job.job_id] = job

    t = threading.Thread(target=_run_full_scan, args=(job, sc, symbols, filters), daemon=True)
    t.start()

    return {"job_id": job.job_id, "total_symbols": len(symbols)}


@app.get("/scan/status/{job_id}")
def scan_status(job_id: str):
    job = _scan_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id":         job.job_id,
        "status":         job.status,
        "total":          job.total,
        "liquid":         job.liquid,
        "processed":      job.processed,
        "qualified":      job.qualified,
        "current_symbol": job.current_symbol,
        "error":          job.error,
    }


@app.get("/scan/results/{job_id}")
def scan_results(job_id: str):
    job = _scan_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=json.loads(to_json(job.results)))


@app.post("/scan/stop/{job_id}")
def scan_stop(job_id: str):
    job = _scan_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # If paused, clear pause first so the thread can see the stop signal
    job._pause_event.clear()
    job._stop_event.set()
    return {"job_id": job_id, "message": "Stop signal sent"}


@app.post("/scan/pause/{job_id}")
def scan_pause(job_id: str):
    job = _scan_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "running":
        raise HTTPException(status_code=400, detail=f"Cannot pause — job is '{job.status}'")
    job._pause_event.set()
    return {"job_id": job_id, "message": "Pause signal sent"}


@app.post("/scan/resume/{job_id}")
def scan_resume(job_id: str):
    job = _scan_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "paused":
        raise HTTPException(status_code=400, detail=f"Cannot resume — job is '{job.status}'")
    job._pause_event.clear()
    return {"job_id": job_id, "message": "Resume signal sent"}
