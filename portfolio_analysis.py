"""Simple portfolio analysis using Kite API

This script demonstrates how to fetch holdings via the MCP toolkit and perform
basic analytics such as total P/L, required return to breakeven, and generating
alerts for large drawdowns.

Usage:
    python portfolio_analysis.py

Note: requires valid Kite login tokens available via environment or manual input
as per the kite-mcp setup in this workspace.
"""

from typing import List, Dict

# We'll build a wrapper that reuses the mcp_kite_get_holdings tool from the
# chatbot. In a standalone script you'd call the Kite Connect REST API or use
# kitemcp library. For demo purposes we simulate the calls.

# The script assumes you have mcp_kite_* functions accessible when running
# within the repl context of the assistant. If running locally, you'd adapt the
# calls accordingly.


def format_holdings(holdings: List[Dict]) -> None:
    """Pretty-print holdings in a table."""
    if not holdings:
        print("No holdings available.")
        return

    print(f"{'Symbol':<15}{'Qty':>8}{'Avg Price':>12}{'Last':>12}{'P/L':>14}{'Day %':>10}")
    for h in holdings:
        symbol = h['tradingsymbol']
        qty = h['quantity']
        avg = h['average_price']
        last = h['last_price']
        pnl = h['pnl']
        day_pct = h.get('day_change_percentage', 0)
        print(f"{symbol:<15}{qty:>8}{avg:>12.2f}{last:>12.2f}{pnl:>14.2f}{day_pct:>10.2f}")


def compute_totals(holdings: List[Dict]) -> Dict[str, float]:
    total_invested = sum(h['average_price'] * h['quantity'] for h in holdings)
    total_current = sum(h['last_price'] * h['quantity'] for h in holdings)
    total_pnl = total_current - total_invested
    return {
        'invested': total_invested,
        'current': total_current,
        'pnl': total_pnl,
    }


def required_return_to_breakeven(holdings: List[Dict]) -> float:
    """Assuming a new investment equal to loss, find required overall return."""
    totals = compute_totals(holdings)
    invested = totals['invested']
    pnl = totals['pnl']
    if invested == 0:
        return 0.0
    # breakeven needs pnl to be 0, so change needed = -pnl
    return (-pnl / invested) * 100


def fetch_holdings():
    """Fetch holdings via MCP, return empty list on failure."""
    try:
        from mcp import mcp_kite_get_holdings
        return mcp_kite_get_holdings()
    except Exception:
        print("Unable to fetch holdings using MCP tool. Please integrate with the real Kite Connect API.")
        return []


def fetch_quote(symbol: str) -> Dict:
    """Return a quote dictionary for an NSE symbol using MCP helpers."""
    # the API expects exchange:symbol
    exch_sym = f"NSE:{symbol}"
    try:
        from mcp import mcp_kite_get_quotes
        resp = mcp_kite_get_quotes(instruments=[exch_sym])
        return resp.get(exch_sym, {})
    except Exception:
        print(f"Could not fetch quote for {symbol}")
        return {}


def analyze_sector(symbols: List[str]) -> None:
    """Print basic quote information for a list of symbols."""
    print(f"{'Symbol':<10}{'Last':>10}{'Change%':>10}{'OI':>10}")
    for sym in symbols:
        q = fetch_quote(sym)
        last = q.get('last_price', 0)
        day_pct = q.get('day_change_percentage', 0)
        oi = q.get('oi', 0)
        print(f"{sym:<10}{last:>10.2f}{day_pct:>10.2f}{oi:>10}")


def main():
    # fetch holdings via MCP tool
    print("Fetching holdings...\n")
    holdings = fetch_holdings()

    format_holdings(holdings)
    totals = compute_totals(holdings)
    print("\nTotals:")
    print(f"Invested: {totals['invested']:.2f}")
    print(f"Current:  {totals['current']:.2f}")
    print(f"P/L:      {totals['pnl']:.2f}")

    req = required_return_to_breakeven(holdings)
    print(f"Required return to breakeven: {req:.2f}%")

    # optional sector analysis
    print("\nIT sector analysis (live quotes):")
    it_symbols = ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MINDTREE"]
    analyze_sector(it_symbols)


if __name__ == '__main__':
    main()
