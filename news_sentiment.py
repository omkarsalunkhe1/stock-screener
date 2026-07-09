"""
news_sentiment.py — News headline fetching + sentiment scoring (1-hour TTL cache)

Source: Yahoo Finance via yfinance (appends .NS for NSE stocks).
        No API key required. TextBlob scores headline sentiment.

Usage:
    from news_sentiment import fetch
    data = fetch("RELIANCE")
"""

import time
import logging
from datetime import datetime

log = logging.getLogger(__name__)

# ── In-process 1-hour cache (news is time-sensitive) ──────────────────────
_cache: dict = {}
CACHE_TTL = 3600  # 1 hour


# ── Helpers ───────────────────────────────────────────────────────────────

def _sentiment_label(score: float) -> str:
    """Map normalised score (−1 to +1) to a display label."""
    if score >=  0.15: return "Bullish"
    if score <= -0.15: return "Bearish"
    return "Neutral"


def _news_score(avg: float) -> int:
    """
    Map average sentiment to a 0–15 pt score bonus.
    Only positive sentiment awards points; negative/neutral = 0 (no penalty).
    """
    if avg <= 0:
        return 0
    return min(15, round(avg * 15))


def _relative_time(dt_str: str) -> str:
    """Return a human-readable relative time string (e.g. '2h ago')."""
    try:
        dt    = datetime.fromisoformat(dt_str[:19])
        delta = datetime.utcnow() - dt
        secs  = int(delta.total_seconds())
        if secs < 3600:  return f"{secs // 60}m ago"
        if secs < 86400: return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return dt_str[:10] if dt_str else "—"


# ── yfinance / Yahoo Finance ──────────────────────────────────────────────

def fetch_yfinance(symbol: str) -> list:
    """
    Fetch news from Yahoo Finance via yfinance (appends .NS suffix).
    Uses TextBlob for sentiment. No API key required.
    Handles both old and new yfinance news payload shapes.
    Returns list of article dicts (max 5), same shape as other sources.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol + ".NS")
    news   = ticker.news or []
    if not news:
        return []

    try:
        from textblob import TextBlob
        use_textblob = True
    except ImportError:
        use_textblob = False

    articles = []
    for item in news[:5]:
        # ── field extraction (old shape vs new nested "content" shape) ──
        content   = item.get("content") or {}
        headline  = (item.get("title")
                     or content.get("title", ""))
        url       = (item.get("link")
                     or (item.get("canonicalUrl") or {}).get("url", "")
                     or (content.get("canonicalUrl") or {}).get("url", ""))
        publisher = (item.get("publisher")
                     or (content.get("provider") or {}).get("displayName", "Yahoo Finance"))

        # timestamp: epoch int (old) or ISO string inside content (new)
        epoch = item.get("providerPublishTime", 0)
        if not epoch:
            raw_dt = content.get("pubDate", "")
            try:
                epoch = int(datetime.fromisoformat(raw_dt[:19]).timestamp()) if raw_dt else 0
            except Exception:
                epoch = 0
        try:
            dt_str = datetime.utcfromtimestamp(epoch).isoformat() if epoch else ""
        except Exception:
            dt_str = ""

        if not headline:
            continue

        if use_textblob:
            try:
                polarity = float(TextBlob(headline).sentiment.polarity)
            except Exception:
                polarity = 0.0
        else:
            polarity = 0.0

        articles.append({
            "title":           headline,
            "url":             url,
            "source":          publisher,
            "published_at":    dt_str,
            "relative_time":   _relative_time(dt_str),
            "sentiment_score": round(polarity, 3),
            "sentiment_label": _sentiment_label(polarity),
        })

    return articles


# ── Main entry point ──────────────────────────────────────────────────────

def fetch(symbol: str) -> dict:
    """
    Fetch news + sentiment for an NSE symbol via Yahoo Finance (yfinance).
    Results are cached in-process for 1 hour.

    Returns dict:
        symbol, articles (list), articles_count, overall_sentiment (−1 to +1),
        overall_label, news_score (0–15), source, error
    """
    sym = symbol.strip().upper()
    now = time.time()

    # Serve from cache if fresh
    if sym in _cache and now - _cache[sym].get("_ts", 0) < CACHE_TTL:
        return {k: v for k, v in _cache[sym].items() if k != "_ts"}

    articles: list    = []
    source_used: str  = "yahoo_finance"
    error: str | None = None

    try:
        articles = fetch_yfinance(sym)
        log.info(f"[news] {sym} — {len(articles)} articles via Yahoo Finance")
    except Exception as e:
        log.warning(f"[news] Yahoo Finance news fetch error for {sym}: {e}")
        error = str(e)

    # ── Compute aggregate sentiment ──
    avg_sentiment = (
        sum(a["sentiment_score"] for a in articles) / len(articles)
        if articles else 0.0
    )

    result = {
        "symbol":            sym,
        "articles":          articles[:5],
        "articles_count":    len(articles),
        "overall_sentiment": round(avg_sentiment, 3),
        "overall_label":     _sentiment_label(avg_sentiment),
        "news_score":        _news_score(avg_sentiment),
        "source":            source_used,
        "error":             error,
    }

    _cache[sym] = {**result, "_ts": now}
    return result
