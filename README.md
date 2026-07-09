# NSE Swing Trade Screener

Screens NSE stocks for 1–3 week swing trade setups targeting ~5% return.
Powered by Kite Connect API (Zerodha).

---

## What it does

For each stock in the selected universe (Nifty 50 / 100 / Midcap), it fetches
60 days of daily OHLCV candles and computes:

| Indicator        | What it checks                              | Max score |
|------------------|---------------------------------------------|-----------|
| RSI (14)         | Momentum building, not overbought (45–68)   | 22        |
| Volume surge     | Today's vol vs 20-day avg (1.5x+)           | 18        |
| MACD             | Bullish crossover / positive histogram      | 15        |
| Price vs 20MA    | Above and within 3% of moving average       | 15        |
| ADX (14)         | Trend strength ≥ 25                         | 15        |
| 5D momentum      | Positive short-term price momentum          | 10        |
| Bollinger band   | Middle-third position (setup zone)          | 5         |

**Target price** = 3× ATR above current price  
**Stop loss** = 1.5× ATR below current price  
**Signal**: Strong Buy (≥75), Moderate (≥60), Watch (<60)

---

## Setup

### 1. Prerequisites

- Python 3.11+
- Active Zerodha account
- Kite Connect subscription (₹500/month + ₹2000/month for historical data)
- Kite Connect developer account at https://developers.kite.trade

### 2. Install

```bash
cd swing_screener
pip install -r requirements.txt
```

### 3. Configure credentials

Create a `.env` file (never commit this):

```bash
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=   # fill this after login (step 4)
```

### 4. Get your access token (do this every morning — expires at 6 AM)

**Option A — Manual (one-time setup):**

```python
from kiteconnect import KiteConnect
import hashlib

API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

kite = KiteConnect(api_key=API_KEY)
print("Login URL:", kite.login_url())

# After logging in, you'll be redirected to your registered URL with ?request_token=xxx
request_token = input("Paste your request_token: ")
data = kite.generate_session(request_token, api_secret=API_SECRET)
print("Access token:", data["access_token"])
```

**Option B — Via the REST API server:**

```bash
# Start server
uvicorn server:app --port 8000

# Get login URL
curl http://localhost:8000/auth/login-url

# Exchange token (after logging in via the URL above)
curl -X POST http://localhost:8000/auth/token \
     -H "Content-Type: application/json" \
     -d '{"request_token": "xxx"}'
```

---

## Usage

### CLI (simplest)

```bash
export KITE_API_KEY=xxx
export KITE_ACCESS_TOKEN=yyy

# Screen Nifty 50 for 5%+ setups
python run.py

# Screen Midcap for 7%+ setups with tighter filters
python run.py --universe midcap --target 7 --min-score 70

# Save full results to JSON
python run.py --output today.json

# Full options
python run.py --help
```

### REST API server

```bash
uvicorn server:app --reload --port 8000

# Screen Nifty 100
curl "http://localhost:8000/screen?access_token=yyy&universe=nifty100&min_target_pct=5"

# With all filters
curl "http://localhost:8000/screen?access_token=yyy&universe=nifty50&rsi_min=45&rsi_max=68&min_vol_surge=1.5&min_score=65&min_target_pct=5&top_n=15"
```

Interactive docs: http://localhost:8000/docs

### Daily scheduler (auto-run at 9:30 AM IST)

```bash
# Keep running in background — screens every trading day at 9:30 AM
python scheduler.py

# Or run once immediately
python scheduler.py --once

# Or use cron (add to crontab -e)
30 9 * * 1-5 cd /path/to/screener && python scheduler.py --once >> screener.log 2>&1
```

Results are saved to `results/screen_YYYY-MM-DD.json`.

---

## Connect to the frontend widget

The React screener widget in Claude expects a backend at `http://localhost:8000`.
When you have the server running with a valid access_token, update the widget's
`BACKEND_URL` constant and switch `USE_LIVE_DATA = true`.

---

## Project structure

```
swing_screener/
├── screener.py       ← Core engine: indicators + scoring
├── server.py         ← FastAPI REST server
├── run.py            ← CLI runner
├── scheduler.py      ← Daily auto-runner
├── requirements.txt
├── .env              ← Your credentials (git-ignored)
└── results/          ← Daily JSON outputs
```

---

## Disclaimer

This tool is for **educational purposes only**.  
It does not constitute financial advice.  
Past technical patterns do not guarantee future returns.  
Always do your own research and consult a SEBI-registered investment advisor.
