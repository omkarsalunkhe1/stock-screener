# NSE Swing Trade Screener

Screens NSE equities for 1–3 week swing setups, scores them against a weighted
technical model, and validates the model with a walk-forward backtest harness.
Built on the Kite Connect API (Zerodha), with a FastAPI service, a CLI, and a
scheduler for daily runs.

> Educational project. Not financial advice — see the disclaimer at the end.

---

## Why this exists

Most retail screeners return a list of stocks matching a handful of hard filters,
which produces either too many results or none at all, and gives no way to compare
one candidate against another. This project takes a different approach: every
candidate is scored on a weighted model across seven indicators, so results are
ranked rather than merely filtered, and the model itself can be backtested and
tuned instead of guessed at.

---

## The scoring model

For each stock in the selected universe (Nifty 50 / 100 / Midcap), the screener
pulls 60 days of daily OHLCV candles and scores it out of 100:

| Indicator      | What it checks                            | Weight |
| -------------- | ----------------------------------------- | ------ |
| RSI (14)       | Momentum building, not overbought (45–68) | 22     |
| Volume surge   | Today's volume vs 20-day average (≥1.5×)  | 18     |
| MACD           | Bullish crossover / positive histogram    | 15     |
| Price vs 20MA  | Above, and within 3% of the average       | 15     |
| ADX (14)       | Trend strength ≥ 25                       | 15     |
| 5D momentum    | Positive short-term price momentum        | 10     |
| Bollinger band | Middle-third position (setup zone)        | 5      |

**Signal bands:** Strong Buy ≥ 75, Moderate ≥ 60, Watch below 60.

**Levels:** target = 3 × ATR above current price, stop loss = 1.5 × ATR below.
ATR-based levels rather than fixed percentages, so targets adapt to each stock's
own volatility instead of assuming every name moves the same way.

The weights reflect a bias toward entries where momentum is building but not yet
extended — RSI carries the most weight and is capped at 68 rather than the usual
70, and the price-vs-20MA band deliberately rewards proximity to the average
rather than distance above it, since the strategy targets pullback entries rather
than breakouts.

---

## Architecture

```
├── screener.py             Core engine — indicators, scoring, universe handling
├── server.py               FastAPI REST service
├── run.py                  CLI runner
├── scheduler.py            Daily auto-run at 9:30 AM IST
├── backtest.py             Walk-forward backtest harness with experiment knobs
├── kite_ticker.py          Kite WebSocket client for live tick streaming
├── get_token.py            Interactive Kite OAuth helper
├── app.py                  Config bootstrap and session handling
│
├── fundamentals.py         Fundamental data enrichment
├── news_sentiment.py       News-based sentiment signal
├── event_risk.py           Event-risk checks (earnings and similar)
├── portfolio_analysis.py   Holdings and position analysis
├── swing_paper_broker.py   Paper-trading broker for tracking hypothetical trades
├── kite_analyzer.py        Kite data analysis helpers
│
├── rag_ingest.py           Document ingestion into the vector store
├── rag_engine.py           Retrieval layer for document-grounded queries
│
├── dashboard.py            Local dashboard
├── swing_trade_screener.html   Standalone frontend
└── Stock-Screener.sh       Convenience launcher
```

### Kite Connect integration

Authentication is Kite's two-step OAuth: a login redirect returns a
`request_token`, which is exchanged for an `access_token` using a SHA-256
checksum of the API key, request token and secret. **Access tokens expire daily
at 6 AM IST**, so `get_token.py` exists to make the morning refresh a single
command rather than a manual dance.

Market data comes from two places: the REST API for historical candles, and a
WebSocket tick stream (`kite_ticker.py`) for live prices. The ticker maintains a
single connection and no-ops on reconnect if the access token hasn't changed,
since Kite limits concurrent connections.

---

## Backtesting

The screening model is only as good as its evidence, so `backtest.py` runs it
walk-forward over historical data with configurable parameters — entry variant
(including a pullback-to-MA20 entry), holding period, universe and scoring
thresholds — so changes to the model can be measured rather than assumed.

```bash
python backtest.py --years 3 --universe nifty100
```

---

## Setup

**Requirements:** Python 3.11+, an active Zerodha account, and a Kite Connect
subscription (the historical data add-on is needed for candle data).

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root — it is git-ignored and must stay that way:

```
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_ACCESS_TOKEN=          # filled by get_token.py
```

Then fetch a token (repeat each morning after 6 AM):

```bash
python get_token.py
```

It prints a login URL, takes the `request_token` from the redirect, exchanges it,
and writes the result to your local config.

---

## Usage

**CLI:**

```bash
python run.py                                          # Nifty 50, default filters
python run.py --universe midcap --target 7 --min-score 70
python run.py --output today.json
python run.py --help
```

**REST API:**

```bash
uvicorn server:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

**Scheduler:**

```bash
python scheduler.py           # runs every trading day at 9:30 AM IST
python scheduler.py --once    # single run
```

Or via cron:

```
30 9 * * 1-5 cd /path/to/screener && python scheduler.py --once >> screener.log 2>&1
```

---

## Known limitations

- Access tokens are currently passed as query parameters on some API endpoints.
  Moving these to an `Authorization` header is the correct fix and is on the list —
  query strings end up in access logs and browser history.
- Kite's historical data API is rate-limited, so screening a large universe is
  sequential and takes time; there is no concurrency layer yet.
- The scoring weights are hand-tuned against backtest results, not optimised
  systematically. They are a starting point, not a validated edge.
- Universe definitions are static rather than pulled from an index-constituent
  feed, so they drift as indices are rebalanced.

---

## Disclaimer

For educational purposes only. This is not financial advice, and past technical
patterns do not predict future returns. Do your own research and consult a
SEBI-registered investment adviser before trading.
