"""
coinbase_feed.py — Coinbase WebSocket BTC price feed

Responsibilities:
- Connect to Coinbase Advanced Trade WebSocket API
- Subscribe to BTC-USD ticker channel
- Maintain live price, volume, and OHLCV data
- Reconnect automatically on disconnect
- Provide async get_price() and get_candles() interfaces
"""

# TODO: implement CoinbaseFeed class with asyncio WebSocket
# TODO: implement subscribe() and reconnect loop
# TODO: implement OHLCV candle aggregation (15m, 1h)
# TODO: implement get_price(), get_candles(), get_volume()
# TODO: expose asyncio.Event for price update signals
