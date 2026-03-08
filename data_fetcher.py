"""Market data utilities and ML-based volatility estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
TRADING_DAYS = 252


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _synthetic_data(period_days: int = 504) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.012, period_days)
    close = 100 * np.exp(np.cumsum(returns))
    volume = rng.integers(1_000_000, 5_000_000, size=period_days)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=period_days, freq="B")
    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download historical OHLCV market data; fallback to synthetic demo data."""
    try:
        import yfinance as yf

        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna().copy()
        if not data.empty and {"Close", "Volume"}.issubset(data.columns):
            return data
    except Exception:
        pass

    return _synthetic_data()


def estimate_drift_vol(data: pd.DataFrame) -> tuple[float, float, float, pd.Series]:
    """Compute annualized drift and volatility from historical returns."""
    close = data["Close"]
    returns = close.pct_change().dropna()
    s0 = float(close.iloc[-1])
    mu = float(returns.mean() * TRADING_DAYS)
    sigma_hist = float(returns.std() * np.sqrt(TRADING_DAYS))
    return s0, mu, sigma_hist, returns


def ml_volatility_forecast(data: pd.DataFrame) -> float:
    """Train a simple RandomForest model to predict next-day realized volatility."""
    df = data.copy()
    df["Return"] = df["Close"].pct_change()
    df["RollingVol_5"] = df["Return"].rolling(5).std()
    df["RollingVol_21"] = df["Return"].rolling(21).std()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["VolumeChange"] = df["Volume"].pct_change()
    df["RSI_14"] = compute_rsi(df["Close"], 14)

    realized_vol = df["Return"].rolling(21).std().shift(-1)
    features = df[["RollingVol_5", "RollingVol_21", "MA_10", "MA_50", "VolumeChange", "RSI_14"]]

    model_df = pd.concat([features, realized_vol.rename("Target")], axis=1).dropna()
    if len(model_df) < 60:
        return float(df["Return"].std() * np.sqrt(TRADING_DAYS))

    x = model_df.drop(columns=["Target"])
    y = model_df["Target"]

    try:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=3, n_jobs=-1)
        model.fit(x, y)

        latest_features = features.dropna().iloc[[-1]]
        predicted_daily_vol = float(model.predict(latest_features)[0])
        predicted_daily_vol = max(predicted_daily_vol, 1e-6)
        return float(predicted_daily_vol * np.sqrt(TRADING_DAYS))
    except Exception:
        return float(df["Return"].std() * np.sqrt(TRADING_DAYS))
