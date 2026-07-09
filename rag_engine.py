"""
rag_engine.py — RAG retrieval module
=====================================
Imported by server.py.  Provides two public functions:

    get_rag_context(query, n=3, symbol="") -> str
        Returns a formatted string of the top-n relevant passages.
        When a symbol is provided, company annual-report chunks are
        retrieved first (up to 2 slots), then textbook chunks fill the rest.

    build_rag_query(indicators) -> str
        Constructs a semantic search query from the stock's live
        technical state (patterns, RSI zone, volume, trend, etc.)
"""

from __future__ import annotations
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "rag_db"

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_client        = None
_collection    = None   # trading_docs   (textbooks)
_ar_collection = None   # annual_reports (company reports)
_ready         = False


def _init():
    """Initialise ChromaDB client + both collections (called once on first use)."""
    global _client, _collection, _ar_collection, _ready
    if _ready:
        return True
    if not DB_PATH.exists():
        log.info("[rag] No database found — run rag_ingest.py to build it")
        return False
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        _client = chromadb.PersistentClient(path=str(DB_PATH))
        ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        # Core textbook collection (always required)
        _collection = _client.get_collection("trading_docs", embedding_function=ef)
        count = _collection.count()
        log.info(f"[rag] trading_docs ready — {count} chunks")

        # Annual reports collection (optional — present after reports ingest)
        try:
            _ar_collection = _client.get_collection("annual_reports", embedding_function=ef)
            ar_count = _ar_collection.count()
            log.info(f"[rag] annual_reports ready — {ar_count} chunks")
        except Exception:
            _ar_collection = None
            log.info("[rag] No annual_reports collection yet — drop PDFs in reports/ and run rag_ingest.py --reports-only")

        _ready = True
        return True
    except Exception as e:
        log.warning(f"[rag] Init failed: {e}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def get_rag_context(query: str, n: int = 3, symbol: str = "") -> str:
    """
    Retrieve the top-n relevant passages and return them as a formatted string.

    When *symbol* is given (e.g. "TECHM"), up to 2 slots are filled from the
    company's annual report first; the remainder come from the trading textbooks.
    Returns "" if the database is not ready or query fails.
    """
    if not _init():
        return ""
    try:
        lines = []

        # ── 1. Company annual report chunks (symbol-specific, up to 2 slots) ──
        if symbol and _ar_collection:
            try:
                ar = _ar_collection.query(
                    query_texts=[query],
                    n_results=min(2, _ar_collection.count()),
                    where={"symbol": symbol.upper()},
                )
                ar_docs  = ar.get("documents", [[]])[0]
                ar_metas = ar.get("metadatas", [[]])[0]
                ar_dists = ar.get("distances", [[]])[0]
                for doc, meta, dist in zip(ar_docs, ar_metas, ar_dists):
                    if dist > 0.78:   # slightly lenient — company context is always relevant
                        continue
                    source = meta.get("source", f"{symbol} Annual Report")
                    page   = meta.get("page", "?")
                    lines.append(f"[{source}, p.{page}]\n{doc.strip()}")
            except Exception as e:
                log.debug(f"[rag] Annual report query skipped for {symbol}: {e}")

        # ── 2. Textbook chunks to fill remaining slots ────────────────────────
        remaining = max(2, n - len(lines))
        tb = _collection.query(query_texts=[query], n_results=remaining + 2)
        tb_docs  = tb.get("documents", [[]])[0]
        tb_metas = tb.get("metadatas", [[]])[0]
        tb_dists = tb.get("distances", [[]])[0]

        for doc, meta, dist in zip(tb_docs, tb_metas, tb_dists):
            if dist > 0.70:
                continue
            source = meta.get("source", "Trading Reference")
            page   = meta.get("page", "?")
            lines.append(f"[{source}, p.{page}]\n{doc.strip()}")
            if len(lines) >= n:
                break

        return "\n\n".join(lines)
    except Exception as e:
        log.warning(f"[rag] Query failed: {e}")
        return ""


def build_rag_query(
    price_action:     str   = "",
    bearish_patterns: str   = "",
    rsi:              float = 50.0,
    volume_ratio:     float = 1.0,
    macd_bull:        bool  = False,
    adx:              float = 20.0,
    weekly_trend:     str   = "",
    momentum_20d:     float = 0.0,
    ma20_dist_pct:    float = 0.0,
) -> str:
    """
    Build a semantic search query from the stock's current technical state.
    """
    terms = []

    if price_action:
        for p in price_action.split(","):
            p = p.strip()
            if p:
                terms.append(p)

    if bearish_patterns:
        for p in bearish_patterns.split(","):
            p = p.strip()
            if p:
                terms.append(p)

    if rsi < 32:
        terms.append("RSI oversold reversal buy signal")
    elif rsi > 70:
        terms.append("RSI overbought momentum exhaustion")
    elif 45 <= rsi <= 60:
        terms.append("RSI neutral momentum confirmation")

    if volume_ratio >= 2.0:
        terms.append("high volume breakout confirmation candlestick")
    elif volume_ratio < 0.9:
        terms.append("low volume unconfirmed pattern reversal")

    if macd_bull:
        terms.append("MACD bullish crossover momentum signal")

    if adx > 30:
        terms.append("ADX strong trend continuation")
    elif adx < 20:
        terms.append("ADX weak trendless ranging market pattern reliability")

    wt = weekly_trend.lower()
    if "pullback" in wt:
        terms.append("secondary correction pullback in uptrend Dow Theory buy zone")
    elif "bullish" in wt:
        terms.append("primary uptrend continuation signal")
    elif "bearish" in wt:
        terms.append("downtrend bear market rally dead cat bounce risk")

    if 5 <= momentum_20d <= 25:
        terms.append("multi-week recovery momentum sustained trend")
    elif momentum_20d < 0:
        terms.append("negative momentum downtrend continuation")

    if ma20_dist_pct > 8:
        terms.append("price extended overbought moving average reversion risk")
    elif -3 <= ma20_dist_pct <= 2:
        terms.append("price near moving average support resistance zone")

    if not terms:
        terms.append("swing trade technical analysis entry signal candlestick")

    return " ".join(terms)


def ar_stats() -> dict:
    """Return counts and company list for the annual_reports collection."""
    if not _init() or not _ar_collection:
        return {"count": 0, "companies": []}
    try:
        results = _ar_collection.get(include=["metadatas"])
        symbols = sorted({m.get("symbol", "") for m in results["metadatas"] if m.get("symbol")})
        return {"count": _ar_collection.count(), "companies": symbols}
    except Exception:
        return {"count": 0, "companies": []}


def reload_ar():
    """Force-reload the annual_reports collection (called after ingest)."""
    global _ar_collection, _ready
    _ar_collection = None
    _ready = False
    _init()


def is_ready() -> bool:
    """Return True if the RAG database is built and loaded."""
    return _init()
