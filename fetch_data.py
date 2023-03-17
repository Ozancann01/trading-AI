import ccxt
import pandas as pd

def initialize_exchange():
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })
    return exchange


def fetch_historical_data(exchange, symbol, timeframe, start_date, end_date):
    data = []
    since = exchange.parse8601(start_date)
    end_timestamp = exchange.parse8601(end_date)

    while since < end_timestamp:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since)
        if not candles:
            break

        data.extend(candles)
        since = candles[-1][0] + 1

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df
