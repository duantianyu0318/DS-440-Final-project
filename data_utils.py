import hashlib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def _seed_from_ticker(ticker: str) -> int:
    return int(hashlib.sha256(ticker.encode('utf-8')).hexdigest()[:8], 16)


def generate_demo_data(ticker: str, period: str = '2y') -> pd.DataFrame:
    period_map = {'6mo': 126, '1y': 252, '2y': 504, '5y': 1260}
    n = period_map.get(period, 504)
    rng = np.random.default_rng(_seed_from_ticker(ticker))
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq='B')
    drift = rng.uniform(0.0002, 0.0010)
    volatility = rng.uniform(0.01, 0.025)
    start_price = rng.uniform(40, 250)
    returns = rng.normal(drift, volatility, size=n)
    prices = start_price * np.exp(np.cumsum(returns))
    volume = rng.integers(800_000, 5_000_000, size=n)
    df = pd.DataFrame({'Close': prices, 'Volume': volume}, index=dates)
    return df


def get_stock_data(ticker: str, period: str = '2y', demo_mode: bool = False) -> pd.DataFrame | None:
    ticker = ticker.strip().upper()
    if demo_mode:
        return generate_demo_data(ticker, period)
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return generate_demo_data(ticker, period)
        cols = [c for c in ['Close', 'Volume'] if c in data.columns]
        return data[cols].dropna()
    except Exception:
        return generate_demo_data(ticker, period)


def preprocess_data(data: pd.DataFrame, window: int = 10):
    if data is None or data.empty or len(data) <= window + 5:
        raise ValueError('Not enough data to build the model.')

    close_only = data[['Close']].copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_only)

    X, y, timeline = [], [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i].flatten())
        y.append(scaled[i][0])
        timeline.append(close_only.index[i])

    X = np.array(X)
    y = np.array(y)
    split_idx = max(int(len(X) * 0.8), 1)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    train_dates, test_dates = timeline[:split_idx], timeline[split_idx:]

    if len(X_test) == 0:
        X_train, X_test = X[:-1], X[-1:]
        y_train, y_test = y[:-1], y[-1:]
        train_dates, test_dates = timeline[:-1], timeline[-1:]

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_dates': train_dates,
        'test_dates': test_dates,
        'scaler': scaler,
        'raw_close': close_only,
        'last_window': X[-1].reshape(1, -1),
        'last_actual_close': float(close_only.iloc[-1, 0]),
    }
