# Swing Screener Change Log

This file tracks date/time-wise changes made to the swing screener app. Times are recorded in IST (UTC+05:30).

## 2026-05-13 09:19:16 IST

### Kite Funds Visibility
- Added a read-only `/funds` API endpoint that fetches Kite equity and commodity margins using the saved/live Kite access token.
- Added a `Funds` button and compact funds chip in the screener connection bar.
- Added a portfolio-tab funds summary showing available funds, opening balance, used margin, collateral, and last update time.
- Funds are fetched only after live connection or manual refresh, avoiding continuous polling.

## 2026-05-11 21:18:29 IST

### Paper Trade Position Sizing Guard
- Added backend paper-buy capital caps for weaker swing setups: INR 10,000 when score is below 70 and INR 15,000 for moderate/non-strong setups.
- Added a paper-buy modal warning that shows the applicable cap and disables confirmation when the requested quantity exceeds it.
- Strong setups with score 85+ and `strong_buy` signal remain uncapped by this new paper-sizing guard.
