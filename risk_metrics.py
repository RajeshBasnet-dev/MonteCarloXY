"""Financial risk metrics for Monte Carlo outputs."""

from __future__ import annotations

import numpy as np


def calculate_risk_metrics(terminal_returns: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    """Return VaR, CVaR, Sharpe, and max drawdown from terminal returns."""
    if terminal_returns.size == 0:
        raise ValueError("terminal_returns cannot be empty")

    alpha = 1 - confidence
    var_level = np.percentile(terminal_returns, 100 * alpha)
    var_95 = -float(var_level)

    tail = terminal_returns[terminal_returns <= var_level]
    cvar_95 = -float(np.mean(tail)) if tail.size > 0 else var_95

    mean_ret = float(np.mean(terminal_returns))
    std_ret = float(np.std(terminal_returns))
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

    equity_curve = np.cumprod(1 + terminal_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = abs(float(np.min(drawdowns)))

    return {
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": max_drawdown,
    }
