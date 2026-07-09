"""
Daily scheduler — runs the screener automatically every morning at 9:30 AM IST
(just after market open), saves results, and optionally sends a summary.

Run once and keep alive:  python scheduler.py

Or add to crontab:
  30 9 * * 1-5  /usr/bin/python3 /path/to/scheduler.py --once
"""

import argparse
import json
import os
import time
import logging
from datetime import datetime
import pytz

from screener import SwingScreener, UNIVERSES, print_report, to_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("screener.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

API_KEY = os.environ.get("KITE_API_KEY", "")
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

DEFAULT_FILTERS = {
    "rsi_min": 45,
    "rsi_max": 68,
    "min_vol_surge": 1.5,
    "min_score": 60,
    "min_target_pct": 5.0,
}


def run_and_save():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sc = SwingScreener(api_key=API_KEY, access_token=ACCESS_TOKEN)
    sc.load_instruments()

    all_results = {}
    for universe in ["nifty50", "nifty100", "midcap"]:
        log.info(f"Screening {universe}...")
        results = sc.run(universe=universe, filters=DEFAULT_FILTERS)
        all_results[universe] = results

    today = datetime.now(IST).strftime("%Y-%m-%d")
    out_path = os.path.join(RESULTS_DIR, f"screen_{today}.json")

    with open(out_path, "w") as f:
        json.dump(
            {k: json.loads(to_json(v)) for k, v in all_results.items()},
            f, indent=2
        )
    log.info(f"Results saved to {out_path}")

    # Print top picks
    combined = []
    for v in all_results.values():
        combined.extend(v)
    combined.sort(key=lambda x: x["scoring"]["score"], reverse=True)
    # Deduplicate symbols
    seen = set()
    deduped = []
    for r in combined:
        if r["symbol"] not in seen:
            deduped.append(r)
            seen.add(r["symbol"])
    print_report(deduped, top_n=10)


def should_run_today() -> bool:
    now = datetime.now(IST)
    return now.weekday() < 5  # Monday=0 ... Friday=4


def wait_until_930():
    while True:
        now = datetime.now(IST)
        target = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now >= target:
            return
        wait = (target - now).total_seconds()
        log.info(f"Waiting {int(wait/60)} minutes until 9:30 AM IST...")
        time.sleep(min(wait, 300))  # wake up every 5 min to recheck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="Run once immediately and exit")
    args = parser.parse_args()

    if not API_KEY or not ACCESS_TOKEN:
        log.error("Set KITE_API_KEY and KITE_ACCESS_TOKEN environment variables")
        return

    if args.once:
        run_and_save()
        return

    log.info("Scheduler started. Will run every trading day at 9:30 AM IST.")
    last_run_date = None

    while True:
        now = datetime.now(IST)
        today = now.date()

        if should_run_today() and last_run_date != today:
            wait_until_930()
            log.info("Running screener...")
            try:
                run_and_save()
                last_run_date = today
            except Exception as e:
                log.error(f"Screener failed: {e}", exc_info=True)

        time.sleep(60)


if __name__ == "__main__":
    main()
