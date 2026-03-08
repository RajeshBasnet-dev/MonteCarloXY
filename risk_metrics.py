"""Financial risk metrics for Monte Carlo portfolio simulations."""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def calculate_risk_metrics(
    paths: np.ndarray,
    daily_returns: np.ndarray,
    confidence: float = 0.95,
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    """Compute professional risk analytics from simulation outputs."""
    if paths.size == 0 or daily_returns.size == 0:
        raise ValueError("paths and daily_returns cannot be empty")

    terminal = (paths[-1] / paths[0]) - 1.0

    alpha = 1.0 - confidence
    var_cutoff = np.percentile(terminal, 100 * alpha)
    tail = terminal[terminal <= var_cutoff]

    expected_terminal_return = float(np.mean(terminal))
    terminal_volatility = float(np.std(terminal))

    flat_daily = daily_returns.reshape(-1)
    mean_daily = float(np.mean(flat_daily))
    std_daily = float(np.std(flat_daily))

    annual_return = mean_daily * TRADING_DAYS
    annual_volatility = std_daily * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0

    running_max = np.maximum.accumulate(paths, axis=0)
    drawdowns = (paths - running_max) / running_max
    max_drawdown = abs(float(np.min(drawdowns)))

    return {
        "VaR_95": -float(var_cutoff),
        "CVaR_95": -float(np.mean(tail)) if tail.size else -float(var_cutoff),
        "Expected_Return": expected_terminal_return,
        "Terminal_Volatility": terminal_volatility,
        "Annualized_Return": annual_return,
        "Annualized_Volatility": annual_volatility,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": max_drawdown,
    }
