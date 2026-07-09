"""
get_token.py — Kite Access Token Generator
============================================
Run this script once per day to refresh your Kite access token.

Usage:
    python get_token.py

Steps:
  1. Script opens the Kite login page in your browser
  2. Log in with Zerodha credentials (User ID + Password + TOTP/PIN)
  3. After login you are redirected to a URL containing ?request_token=XXXX
  4. Copy that request_token and paste it here
  5. Script exchanges it for an access_token and saves to kite_config.json
"""

import json
import sys
import webbrowser
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "kite_config.json"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        print(f"\n[!] kite_config.json not found. Creating template...")
        cfg = {
            "api_key": "",
            "api_secret": "",
            "access_token": ""
        }
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        print(f"[✓] Created {CONFIG_FILE}")
        return cfg
    return json.loads(CONFIG_FILE.read_text())


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    print(f"[✓] Saved to {CONFIG_FILE}")


def main():
    print("\n" + "=" * 50)
    print("  Kite Access Token Generator")
    print("=" * 50)

    # ── Load config ──────────────────────────────────────────────────────────
    config = load_config()

    api_key = config.get("api_key", "").strip()
    api_secret = config.get("api_secret", "").strip()

    # ── Get API key if missing ────────────────────────────────────────────────
    if not api_key or api_key.startswith("YOUR_"):
        print("\nYour API Key is not configured.")
        print("Find it at: https://developers.kite.trade/apps")
        api_key = input("Enter API Key: ").strip()
        config["api_key"] = api_key

    if not api_secret or api_secret.startswith("YOUR_"):
        print("\nYour API Secret is not configured.")
        print("Find it at: https://developers.kite.trade/apps")
        api_secret = input("Enter API Secret: ").strip()
        config["api_secret"] = api_secret

    save_config(config)

    # ── Open login URL ────────────────────────────────────────────────────────
    login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
    print(f"\n[~] Opening Kite login page...")
    print(f"    URL: {login_url}")

    try:
        webbrowser.open(login_url)
        print("[✓] Browser opened")
    except Exception:
        print("[!] Could not open browser. Open this URL manually:")
        print(f"    {login_url}")

    # ── Get request token ─────────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("STEPS:")
    print("  1. Log in with your Zerodha credentials")
    print("  2. After login, the browser redirects to a URL like:")
    print("     https://...?request_token=XXXXXXXXXXXXXXXXXX&action=login")
    print("  3. Copy the request_token value from that URL")
    print("-" * 50)

    request_token = input("\nPaste request_token here: ").strip()
    if not request_token:
        print("[✗] No token entered. Exiting.")
        sys.exit(1)

    # Clean up if user pasted full URL
    if "request_token=" in request_token:
        request_token = request_token.split("request_token=")[1].split("&")[0]
        print(f"[~] Extracted token: {request_token[:8]}...")

    # ── Exchange for access token ─────────────────────────────────────────────
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("\n[✗] kiteconnect not installed.")
        print("    Run: pip install kiteconnect")
        sys.exit(1)

    print("\n[~] Generating access token...")
    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        user_name    = data.get("user_name", "Unknown")

        print(f"[✓] Access token generated for: {user_name}")
        print(f"    Token (first 8 chars): {access_token[:8]}...")

        # Save
        config["access_token"] = access_token
        save_config(config)

        print("\n" + "=" * 50)
        print(f"  SUCCESS! Token saved to kite_config.json")
        print(f"  User    : {user_name}")
        print(f"  Valid   : Until midnight tonight")
        print("=" * 50)
        print("\n  Now run: python app.py")
        print("  Open   : http://localhost:8000\n")

    except Exception as e:
        print(f"\n[✗] Failed to generate access token: {e}")
        print("\nCommon causes:")
        print("  - Wrong API secret")
        print("  - request_token already used (each token is one-time use)")
        print("  - Token expired (login again)")
        sys.exit(1)


if __name__ == "__main__":
    main()
