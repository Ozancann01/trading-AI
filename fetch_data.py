import ccxt
import pandas as pd
import talib
from talib import abstract


def initialize_exchange():
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })
    return exchange

def head_and_shoulders(data, window=100, percentage_diff=3):
    """
    Detect head and shoulders pattern.
    """
    head_and_shoulders = []

    for i in range(window, len(data) - window):
        local_max = data['high'][i - window:i + window].max()
        local_min = data['low'][i - window:i + window].min()

        max_indices = data['high'][i - window:i + window][data['high'][i - window:i + window] == local_max].index.tolist()
        min_indices = data['low'][i - window:i + window][data['low'][i - window:i + window] == local_min].index.tolist()

        if len(max_indices) >= 3 and len(min_indices) >= 2:
            mid_idx = max_indices[len(max_indices) // 2]
            if max_indices[-1] > max_indices[0] and min_indices[-1] > min_indices[0] and mid_idx in max_indices:
                head_and_shoulders.append(1)
            else:
                head_and_shoulders.append(0)
        else:
            head_and_shoulders.append(0)

    # Pad the beginning and end of the pattern list
    padding = [0] * window
    head_and_shoulders = padding + head_and_shoulders + padding

    return head_and_shoulders


def double_top_bottom(data, window=100, percentage_diff=3):
    """
    Detect double top and double bottom patterns.
    """
    double_top = []
    double_bottom = []
    
    for i in range(window, len(data) - window):
        local_max = data['high'][i-window:i+window].max()
        local_min = data['low'][i-window:i+window].min()
        
        if abs(data['high'][i] - local_max) / local_max * 100 <= percentage_diff:
            double_top.append(1)
        else:
            double_top.append(0)
        
        if abs(data['low'][i] - local_min) / local_min * 100 <= percentage_diff:
            double_bottom.append(1)
        else:
            double_bottom.append(0)
    
    # Pad the beginning and end of the pattern lists
    padding = [0] * window
    double_top = padding + double_top + padding
    double_bottom = padding + double_bottom + padding
    
    return double_top, double_bottom


def add_indicators(data):
    # RSI
    data['RSI'] = talib.RSI(data['close'].values, timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_signal'] = macd_signal
    data['MACD_hist'] = macd_hist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(data['close'].values, timeperiod=20)
    data['BB_upper'] = upper
    data['BB_middle'] = middle
    data['BB_lower'] = lower

    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, data['close'].values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    data['Stoch_slowk'] = slowk
    data['Stoch_slowd'] = slowd
  # Trend: Moving Averages
    data['SMA'] = talib.SMA(data['close'].values, timeperiod=30)
    data['EMA'] = talib.EMA(data['close'].values, timeperiod=30)

    # Parabolic SAR
    data['SAR'] = talib.SAR(data['high'].values, data['low'].values, acceleration=0.02, maximum=0.2)

    # Average Directional Movement Index (ADX)
    data['ADX'] = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)

    # Candlestick Patterns
    data['Hammer'] = abstract.CDLHAMMER(data).values
    data['Doji'] = abstract.CDLDOJI(data).values
    data['Engulfing'] = abstract.CDLENGULFING(data).values
    data['Harami'] = abstract.CDLHARAMI(data).values
    data['PiercingLine'] = abstract.CDLPIERCING(data).values
    data['DarkCloudCover'] = abstract.CDLDARKCLOUDCOVER(data).values
    
       # Chart Patterns
    data['DoubleTop'], data['DoubleBottom'] = double_top_bottom(data)
        # Chart Patterns
    data['DoubleTop'], data['DoubleBottom'] = double_top_bottom(data)
    data['HeadAndShoulders'] = head_and_shoulders(data)

    return data

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
    
    df = add_indicators(df)  # Call the add_indicators function here

    return df
