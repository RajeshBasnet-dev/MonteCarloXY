import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_fetcher import (
    MIN_HIST_ROWS_FOR_ML,
    ai_forecast_return_vol,
    compute_returns,
    portfolio_return_series,
)
from monte_engine import simulate_portfolio_paths, terminal_returns
from risk_metrics import calculate_risk_metrics


def test_simulation_output_dimensions():
    rng = np.random.default_rng(1)
    asset_returns = rng.normal(0.0005, 0.01, size=(400, 3))
    weights = np.array([0.4, 0.35, 0.25])

    paths, daily, elapsed = simulate_portfolio_paths(
        initial_investment=10000,
        asset_returns=asset_returns,
        weights=weights,
        years=1,
        n_sims=5000,
        seed=1,
    )
    assert paths.shape == (253, 5000)
    assert daily.shape == (252, 5000)
    assert elapsed >= 0


def test_risk_metrics_dictionary_keys():
    rng = np.random.default_rng(2)
    asset_returns = rng.normal(0.0002, 0.012, size=(300, 2))
    paths, daily, _ = simulate_portfolio_paths(10000, asset_returns, np.array([0.6, 0.4]), 1, 3000, seed=2)

    metrics = calculate_risk_metrics(paths, daily)
    required = {
        "VaR_95",
        "CVaR_95",
        "Expected_Return",
        "Annualized_Volatility",
        "Sharpe_Ratio",
        "Max_Drawdown",
    }
    assert required.issubset(metrics.keys())
    assert metrics["CVaR_95"] >= metrics["VaR_95"]


def test_engine_stability_no_nans():
    rng = np.random.default_rng(3)
    asset_returns = rng.normal(0.0003, 0.02, size=(600, 4))
    paths, _, _ = simulate_portfolio_paths(15000, asset_returns, np.array([0.2, 0.3, 0.25, 0.25]), 2, 10000, seed=42)
    rets = terminal_returns(paths)
    assert np.isfinite(paths).all()
    assert np.isfinite(rets).all()


def test_ai_forecast_includes_debug_metadata_and_valid_source():
    rng = np.random.default_rng(4)
    prices = pd.DataFrame(
        {
            "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 260))),
            "BBB": 90 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, 260))),
        }
    )
    returns = compute_returns(prices)
    portfolio_rets = portfolio_return_series(returns, np.array([0.5, 0.5]))

    forecast = ai_forecast_return_vol(portfolio_rets)
    assert np.isfinite(forecast["mu_annual"])
    assert np.isfinite(forecast["sigma_annual"])
    assert forecast["sigma_annual"] > 0
    assert forecast["source"] in {"ml_random_forest", "historical_fallback"}
    assert isinstance(forecast["n_hist_rows"], int)
    assert isinstance(forecast["n_feature_rows"], int)
    assert "mode_label" in forecast


def test_ai_forecast_uses_fallback_when_history_is_too_short():
    rng = np.random.default_rng(5)
    short_returns = pd.Series(rng.normal(0.0005, 0.01, MIN_HIST_ROWS_FOR_ML - 20))
    forecast = ai_forecast_return_vol(short_returns)
    assert forecast["source"] == "historical_fallback"
    assert "need >=" in str(forecast["mode_label"])
