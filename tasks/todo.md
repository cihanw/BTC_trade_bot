# BTC Demo Trade Bot

## Todo
- [x] Rebuild the training-time scaler and live inference pipeline from `final.csv`.
- [x] Fetch live 30m candles and funding data, then merge daily context features.
- [x] Implement Binance demo order execution, position handling, and TP/SL management.
- [x] Apply the requested `risk_high` and `risk_low` trade rules.
- [x] Ship a minimal Tkinter UI with only risk selection and a start button.
- [x] Run a local smoke test and document the result.

## Review
- Added `live_trading_bot.py` for end-to-end live inference, UI, and Binance demo execution.
- Added `bot_settings.py` so API keys and runtime defaults can be edited in one place.
- Updated sizing so stop loss targets `3%` of account equity with `SL = threshold * 0.75` and `TP = threshold`.
- Smoke test passed on 2026-03-23 with the latest closed bar at `2026-03-23 08:00:00`.
