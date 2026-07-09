"""
CLI runner — run the screener directly without the web server.

Usage:
  python run.py --access-token YOUR_TOKEN
  python run.py --access-token YOUR_TOKEN --universe midcap --target 7
  python run.py --access-token YOUR_TOKEN --universe nifty100 --min-score 70 --output results.json
"""

import argparse
import json
import os
from screener import SwingScreener, UNIVERSES, print_report, to_json


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trade Screener")
    parser.add_argument("--api-key", default=os.environ.get("KITE_API_KEY", ""),
                        help="Kite API key (or set KITE_API_KEY env var)")
    parser.add_argument("--access-token", default=os.environ.get("KITE_ACCESS_TOKEN", ""),
                        help="Kite access token (or set KITE_ACCESS_TOKEN env var)")
    parser.add_argument("--universe", default="nifty50",
                        choices=list(UNIVERSES.keys()),
                        help="Stock universe to screen")
    parser.add_argument("--rsi-min", type=float, default=45)
    parser.add_argument("--rsi-max", type=float, default=68)
    parser.add_argument("--min-volume", type=float, default=1.5,
                        help="Minimum volume surge vs 20-day average")
    parser.add_argument("--min-score", type=int, default=60,
                        help="Minimum composite score (0-100)")
    parser.add_argument("--target", type=float, default=5.0,
                        help="Minimum target return %%")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top results to show")
    parser.add_argument("--output", default=None,
                        help="Save full results to JSON file")

    args = parser.parse_args()

    if not args.api_key or not args.access_token:
        print("\nError: --api-key and --access-token are required.")
        print("Set them as env vars: export KITE_API_KEY=xxx  KITE_ACCESS_TOKEN=yyy\n")
        parser.print_help()
        return

    filters = {
        "rsi_min": args.rsi_min,
        "rsi_max": args.rsi_max,
        "min_vol_surge": args.min_volume,
        "min_score": args.min_score,
        "min_target_pct": args.target,
    }

    sc = SwingScreener(api_key=args.api_key, access_token=args.access_token)
    sc.load_instruments()
    results = sc.run(universe=args.universe, filters=filters)

    print_report(results, top_n=args.top)

    if args.output:
        with open(args.output, "w") as f:
            f.write(to_json(results))
        print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
