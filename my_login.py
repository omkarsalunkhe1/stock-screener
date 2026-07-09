from kiteconnect import KiteConnect

kite = KiteConnect(api_key="REDACTED")
print("Login here:", kite.login_url())