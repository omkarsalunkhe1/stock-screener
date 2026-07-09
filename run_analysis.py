"""
run_analysis.py — Claude MCP entrypoint
========================================
This script is designed to be run from within the Claude Code MCP session
where kite MCP tools are already injected as functions.

It monkeypatches the MCP tool calls so kite_analyzer can use them directly.
"""

import sys
import types

# ---------------------------------------------------------------------------
# Inject MCP tool shims into a fake "mcp_tools" module
# so kite_analyzer.py can import them with:
#   from mcp_tools import mcp_kite_get_historical_data, ...
# ---------------------------------------------------------------------------
# These functions will be replaced by the actual MCP calls from Claude's context
# when this script is executed by the assistant.

def _not_available(*args, **kwargs):
    raise RuntimeError("MCP tool not bound. Run this inside Claude Code session.")

mcp_mod = types.ModuleType("mcp_tools")

# These get replaced when the assistant calls run_in_session()
mcp_mod.mcp_kite_get_historical_data  = _not_available
mcp_mod.mcp_kite_get_quotes           = _not_available
mcp_mod.mcp_kite_search_instruments   = _not_available
mcp_mod.mcp_kite_get_holdings         = _not_available
mcp_mod.mcp_kite_get_ltp              = _not_available

sys.modules["mcp_tools"] = mcp_mod


def bind_mcp_tools(
    get_historical_data_fn,
    get_quotes_fn,
    search_instruments_fn,
    get_holdings_fn=None,
    get_ltp_fn=None,
):
    """Bind actual MCP tool functions after import."""
    import mcp_tools
    mcp_tools.mcp_kite_get_historical_data = get_historical_data_fn
    mcp_tools.mcp_kite_get_quotes          = get_quotes_fn
    mcp_tools.mcp_kite_search_instruments  = search_instruments_fn
    if get_holdings_fn:
        mcp_tools.mcp_kite_get_holdings = get_holdings_fn
    if get_ltp_fn:
        mcp_tools.mcp_kite_get_ltp = get_ltp_fn


if __name__ == "__main__":
    # When run as a script, forward to kite_analyzer main
    from kite_analyzer import main
    main()
