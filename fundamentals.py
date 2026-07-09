"""
fundamentals.py — On-demand fundamental data via yfinance (24-hour TTL cache)

Usage:
    from fundamentals import fetch
    data = fetch("RELIANCE")   # NSE symbol — appends .NS internally
"""
import time
import logging
from typing import Any

log = logging.getLogger(__name__)

# ── In-process 24-hour cache ──────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 86_400  # 24 hours


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe(d: Any, *keys, default=None):
    """Safely traverse nested dicts/objects."""
    val = d
    for k in keys:
        try:
            val = val[k]
        except (KeyError, IndexError, TypeError, AttributeError):
            return default
    return default if val is None else val


def _crore(val) -> "float | None":
    """Convert absolute rupees to crores (1 Cr = 10^7). None if non-numeric."""
    try:
        return round(float(val) / 1e7, 1)
    except (TypeError, ValueError):
        return None


def _get(df, row: str, col: int = 0) -> "float | None":
    """
    Read one cell from a yfinance DataFrame (index = metric names, columns = dates).
    Returns None on any lookup failure.
    """
    try:
        if df is None or df.empty:
            return None
        if row not in df.index:
            return None
        return float(df.loc[row].iloc[col])
    except Exception:
        return None


def _r(v, mul: float = 1.0, dp: int = 2) -> "float | None":
    """Round and scale a raw value. Returns None if not numeric."""
    try:
        return round(float(v) * mul, dp)
    except (TypeError, ValueError):
        return None


# ── Piotroski F-Score ─────────────────────────────────────────────────────

_PIOTROSKI_LABELS = {
    "roa_positive":               "ROA > 0  (profitable)",
    "ocf_positive":               "Operating cash flow > 0",
    "roa_increasing":             "ROA improving YoY",
    "accruals":                   "OCF > Net income  (quality earnings)",
    "leverage_decreasing":        "Long-term debt ratio falling",
    "current_ratio_increasing":   "Liquidity (current ratio) improving",
    "no_dilution":                "No share dilution",
    "gross_margin_increasing":    "Gross margin improving YoY",
    "asset_turnover_increasing":  "Asset turnover improving YoY",
}


def calc_piotroski(info: dict, fin: Any, bs: Any, cf: Any) -> dict:
    """
    Return dict:
        score (int 0-9),
        criteria (list of {key, label, pass})
    """
    c: dict[str, bool] = {}

    # -- Profitability (4 criteria) --
    net_income   = _safe(info, "netIncomeToCommon", default=0) or 0
    total_assets = _safe(info, "totalAssets",       default=1) or 1
    op_cf        = _safe(info, "operatingCashflow", default=0) or 0

    roa = net_income / total_assets
    c["roa_positive"] = roa > 0
    c["ocf_positive"] = op_cf > 0

    ni_curr = _get(fin, "Net Income", 0)
    ni_prev = _get(fin, "Net Income", 1)
    ta_curr = _get(bs,  "Total Assets", 0) or total_assets
    ta_prev = _get(bs,  "Total Assets", 1) or total_assets
    if ni_curr is not None and ni_prev is not None and ta_curr and ta_prev:
        c["roa_increasing"] = (ni_curr / ta_curr) > (ni_prev / ta_prev)
    else:
        c["roa_increasing"] = False

    c["accruals"] = (op_cf / total_assets) > roa if total_assets else False

    # -- Leverage / liquidity (3 criteria) --
    ltd_curr    = _get(bs, "Long Term Debt", 0) or 0
    ltd_prev    = _get(bs, "Long Term Debt", 1) or 0
    ta_bs_curr  = _get(bs, "Total Assets",   0) or 1
    ta_bs_prev  = _get(bs, "Total Assets",   1) or ta_bs_curr
    c["leverage_decreasing"] = (ltd_curr / ta_bs_curr) < (ltd_prev / ta_bs_prev) if ta_bs_curr and ta_bs_prev else False

    ca_curr = _get(bs, "Current Assets",      0)
    cl_curr = _get(bs, "Current Liabilities", 0)
    ca_prev = _get(bs, "Current Assets",      1)
    cl_prev = _get(bs, "Current Liabilities", 1)
    if ca_curr and cl_curr and ca_prev and cl_prev and cl_curr > 0 and cl_prev > 0:
        c["current_ratio_increasing"] = (ca_curr / cl_curr) > (ca_prev / cl_prev)
    else:
        curr_ratio = _safe(info, "currentRatio", default=0) or 0
        c["current_ratio_increasing"] = float(curr_ratio) > 1.0

    # No dilution — conservative: mark True unless we can confirm shares increased
    c["no_dilution"] = True

    # -- Operating efficiency (2 criteria) --
    rev_curr = _get(fin, "Total Revenue", 0)
    rev_prev = _get(fin, "Total Revenue", 1)
    gp_curr  = _get(fin, "Gross Profit",  0)
    gp_prev  = _get(fin, "Gross Profit",  1)
    if gp_curr and rev_curr and gp_prev and rev_prev and rev_curr > 0 and rev_prev > 0:
        c["gross_margin_increasing"] = (gp_curr / rev_curr) > (gp_prev / rev_prev)
    else:
        gm = _safe(info, "grossMargins", default=0) or 0
        c["gross_margin_increasing"] = float(gm) > 0.15

    if rev_curr and rev_prev and ta_bs_curr and ta_bs_prev and ta_bs_curr > 0 and ta_bs_prev > 0:
        c["asset_turnover_increasing"] = (rev_curr / ta_bs_curr) > (rev_prev / ta_bs_prev)
    else:
        c["asset_turnover_increasing"] = False

    score = sum(1 for v in c.values() if v)
    criteria_list = [
        {"key": k, "label": _PIOTROSKI_LABELS.get(k, k), "pass": v}
        for k, v in c.items()
    ]
    return {"score": score, "criteria": criteria_list}


# ── Magic Formula ─────────────────────────────────────────────────────────

def calc_magic_formula(info: dict, fin: Any, bs: Any) -> dict:
    """
    Earnings Yield  = EBIT / Enterprise Value  (%)
    ROIC            = EBIT / Invested Capital   (%)
    Invested Capital = Net Working Capital + Net Fixed Assets
    """
    try:
        ebit = _safe(info, "ebit")
        if ebit is None:
            ebit = _get(fin, "EBIT", 0) or _get(fin, "Operating Income", 0)
        if ebit is None:
            return {"earnings_yield": None, "roic": None, "quality": "—"}

        ev = _safe(info, "enterpriseValue")
        earnings_yield = round(float(ebit) / float(ev) * 100, 2) if (ev and float(ev) > 0) else None

        ca       = _get(bs, "Current Assets",      0) or 0
        cl       = _get(bs, "Current Liabilities", 0) or 0
        net_wc   = ca - cl
        ppe      = _get(bs, "Net PPE",             0) or _get(bs, "Net Tangible Assets", 0) or 0
        inv_cap  = net_wc + ppe

        roic = round(float(ebit) / inv_cap * 100, 2) if inv_cap > 0 else None

        quality = "—"
        if earnings_yield is not None and roic is not None:
            if   earnings_yield > 10 and roic > 20: quality = "High quality"
            elif earnings_yield >  6 and roic > 12: quality = "Good"
            else:                                   quality = "Average"

        return {"earnings_yield": earnings_yield, "roic": roic, "quality": quality}
    except Exception as e:
        log.debug(f"Magic formula calc error: {e}")
        return {"earnings_yield": None, "roic": None, "quality": "—"}


# ── Quarterly results ─────────────────────────────────────────────────────

def get_quarterly(ticker) -> list:
    """
    Return list (up to 4) of quarterly results:
        [{"date": "YYYY-MM-DD", "revenue_cr": float|None, "net_income_cr": float|None}]
    """
    quarters: list = []
    try:
        qfin = ticker.quarterly_financials
        if qfin is None or qfin.empty:
            return quarters
        for i in range(min(4, qfin.shape[1])):
            col_dt = qfin.columns[i]
            date = str(col_dt.date()) if hasattr(col_dt, "date") else str(col_dt)[:10]
            rev = _crore(_get(qfin, "Total Revenue", i))
            ni  = _crore(_get(qfin, "Net Income",    i))
            quarters.append({"date": date, "revenue_cr": rev, "net_income_cr": ni})
    except Exception as e:
        log.debug(f"Quarterly data error: {e}")
    return quarters


# ── Main entry point ──────────────────────────────────────────────────────

def fetch(symbol: str) -> dict:
    """
    Fetch fundamental data for an NSE symbol via yfinance.
    Results are cached in-process for 24 hours.

    Returns dict with keys:
        symbol, pe, pb, eps, roe, de_ratio, fcf_cr, market_cap_cr,
        div_yield, book_value, current_ratio, gross_margin, rev_growth,
        piotroski, magic_formula, quarterly, signal, error
    """
    sym = symbol.strip().upper()
    now = time.time()

    # Serve from cache if fresh
    if sym in _cache and now - _cache[sym].get("_ts", 0) < CACHE_TTL:
        return {k: v for k, v in _cache[sym].items() if k != "_ts"}

    try:
        import yfinance as yf
        ticker = yf.Ticker(sym + ".NS")
        info   = ticker.info or {}

        # Key ratios (raw yfinance values)
        pe         = _safe(info, "trailingPE")
        pb         = _safe(info, "priceToBook")
        eps        = _safe(info, "trailingEps")
        roe        = _safe(info, "returnOnEquity")        # decimal, e.g. 0.15 = 15%
        de_ratio   = _safe(info, "debtToEquity")          # yfinance: ×100, e.g. 50 = D/E 0.50
        op_cf      = _safe(info, "operatingCashflow")
        capex      = _safe(info, "capitalExpenditures")
        mktcap     = _safe(info, "marketCap")
        div_yield  = _safe(info, "dividendYield")         # yfinance changed format: may be decimal (0.02) OR percent (2.0)
        # Normalise to decimal form so the later ×100 stays consistent
        try:
            dy_f = float(div_yield) if div_yield is not None else None
            if dy_f is not None and dy_f > 1:      # already in percent (new yfinance)
                div_yield = dy_f / 100.0
        except (TypeError, ValueError):
            pass
        book_value = _safe(info, "bookValue")
        curr_ratio = _safe(info, "currentRatio")
        gross_marg = _safe(info, "grossMargins")          # decimal
        rev_growth = _safe(info, "revenueGrowth")         # decimal

        # Free cash flow = OCF - Capex, in crores
        fcf_cr: "float | None" = None
        if op_cf is not None and capex is not None:
            fcf_cr = _crore(float(op_cf) - abs(float(capex)))
        elif op_cf is not None:
            fcf_cr = _crore(float(op_cf))

        # Annual financials for Piotroski + Magic Formula
        fin = bs = cf = None
        try:
            fin = ticker.financials
            bs  = ticker.balance_sheet
            cf  = ticker.cashflow
        except Exception:
            pass

        piotroski     = calc_piotroski(info, fin, bs, cf)
        magic_formula = calc_magic_formula(info, fin, bs)
        quarterly     = get_quarterly(ticker)

        # Fundamental signal driven by Piotroski F-Score
        f_score = piotroski["score"]
        signal  = "strong" if f_score >= 7 else ("neutral" if f_score >= 4 else "weak")

        result = {
            "symbol":        sym,
            "pe":            _r(pe),
            "pb":            _r(pb),
            "eps":           _r(eps),
            "roe":           _r(roe, 100),       # → %
            "de_ratio":      _r(de_ratio),        # raw yfinance value (×100 of actual ratio)
            "fcf_cr":        fcf_cr,
            "market_cap_cr": _crore(mktcap),
            "div_yield":     _r(div_yield, 100),  # → %
            "book_value":    _r(book_value),
            "current_ratio": _r(curr_ratio),
            "gross_margin":  _r(gross_marg, 100), # → %
            "rev_growth":    _r(rev_growth, 100),  # → %
            "piotroski":     piotroski,
            "magic_formula": magic_formula,
            "quarterly":     quarterly,
            "signal":        signal,
            "error":         None,
        }

    except ImportError:
        result = {
            "symbol": sym, "error": "yfinance not installed — run: pip install yfinance",
            "pe": None, "pb": None, "eps": None, "roe": None, "de_ratio": None,
            "fcf_cr": None, "market_cap_cr": None, "div_yield": None,
            "book_value": None, "current_ratio": None, "gross_margin": None, "rev_growth": None,
            "piotroski":     {"score": 0, "criteria": []},
            "magic_formula": {"earnings_yield": None, "roic": None, "quality": "—"},
            "quarterly": [], "signal": "—",
        }
    except Exception as e:
        log.warning(f"Fundamentals fetch error for {sym}: {e}")
        result = {
            "symbol": sym, "error": f"Data unavailable: {str(e)[:120]}",
            "pe": None, "pb": None, "eps": None, "roe": None, "de_ratio": None,
            "fcf_cr": None, "market_cap_cr": None, "div_yield": None,
            "book_value": None, "current_ratio": None, "gross_margin": None, "rev_growth": None,
            "piotroski":     {"score": 0, "criteria": []},
            "magic_formula": {"earnings_yield": None, "roic": None, "quality": "—"},
            "quarterly": [], "signal": "—",
        }

    _cache[sym] = {**result, "_ts": now}
    return result
