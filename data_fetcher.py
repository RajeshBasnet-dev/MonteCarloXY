"""Market data utilities and ML feature engineering for MonteCarloXY."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _synthetic_price_data(tickers: list[str], period_days: int = 756) -> pd.DataFrame:
    """Create synthetic adjusted close prices as an offline fallback."""
    rng = np.random.default_rng(7)
    idx = pd.date_range(end=pd.Timestamp.today(), periods=period_days, freq="B")
    prices: dict[str, np.ndarray] = {}

    for i, ticker in enumerate(tickers):
        drift = 0.0002 + (0.0001 * (i % 4))
        vol = 0.01 + (0.002 * (i % 3))
        returns = rng.normal(drift, vol, period_days)
        prices[ticker] = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame(prices, index=idx)


def fetch_stock_data(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Fetch adjusted close data for one or many tickers from yfinance."""
    clean_tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not clean_tickers:
        raise ValueError("Please provide at least one ticker.")

    try:
        import yfinance as yf

        data = yf.download(clean_tickers, period=period, auto_adjust=True, progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            # yfinance usually returns OHLC columns at level 0 and tickers at level 1.
            close = data["Close"].copy()
        elif "Close" in data.columns:
            close = data[["Close"]].rename(columns={"Close": clean_tickers[0]})
        else:
            close = data.copy()

        close = close.dropna(how="all").ffill().dropna(how="any")
        if not close.empty:
            close.columns = [str(c).upper() for c in close.columns]
            return close
    except Exception:
        pass

    return _synthetic_price_data(clean_tickers)


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from adjusted close prices."""
    returns = price_df.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough history to compute returns.")
    return returns


def portfolio_return_series(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Build daily portfolio returns from asset returns and allocation weights."""
    return pd.Series(returns.to_numpy() @ weights, index=returns.index, name="portfolio_return")


def build_ml_features(portfolio_returns: pd.Series, window: int = 21) -> pd.DataFrame:
    """Construct rolling features for return/volatility prediction."""
    df = pd.DataFrame(index=portfolio_returns.index)
    df["ret"] = portfolio_returns
    df["rolling_mean_5"] = portfolio_returns.rolling(5).mean()
    df["rolling_mean_21"] = portfolio_returns.rolling(window).mean()
    df["rolling_vol_5"] = portfolio_returns.rolling(5).std()
    df["rolling_vol_21"] = portfolio_returns.rolling(window).std()
    df["momentum_10"] = (1 + portfolio_returns).rolling(10).apply(np.prod, raw=True) - 1

    df["target_next_return"] = portfolio_returns.shift(-1)
    df["target_next_vol"] = portfolio_returns.rolling(window).std().shift(-1)
    return df.dropna()


def ai_forecast_return_vol(portfolio_returns: pd.Series) -> dict[str, float | str]:
    """Predict next-period return and volatility using Random Forest regressors."""
    base_mu = float(portfolio_returns.mean() * TRADING_DAYS)
    base_sigma = float(portfolio_returns.std() * np.sqrt(TRADING_DAYS))

    ml_df = build_ml_features(portfolio_returns)
    if len(ml_df) < 80:
        return {
            "mu_annual": base_mu,
            "sigma_annual": max(base_sigma, 1e-6),
            "source": "historical_fallback",
        }

    try:
        from sklearn.ensemble import RandomForestRegressor

        features = [
            "rolling_mean_5",
            "rolling_mean_21",
            "rolling_vol_5",
            "rolling_vol_21",
            "momentum_10",
        ]

        x = ml_df[features]

        model_ret = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1)
        model_vol = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1)

        model_ret.fit(x, ml_df["target_next_return"])
        model_vol.fit(x, ml_df["target_next_vol"])

        latest = x.iloc[[-1]]
        pred_ret_daily = float(model_ret.predict(latest)[0])
        pred_vol_daily = max(float(model_vol.predict(latest)[0]), 1e-6)

        return {
            "mu_annual": pred_ret_daily * TRADING_DAYS,
            "sigma_annual": pred_vol_daily * np.sqrt(TRADING_DAYS),
            "source": "ml_random_forest",
        }
    except Exception:
        return {
            "mu_annual": base_mu,
            "sigma_annual": max(base_sigma, 1e-6),
            "source": "historical_fallback",
        }
