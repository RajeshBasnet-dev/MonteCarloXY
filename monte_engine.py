"""Vectorized Monte Carlo simulation engine for portfolio-level simulations."""

from __future__ import annotations

import time

import numpy as np

TRADING_DAYS = 252


def simulate_portfolio_paths(
    initial_investment: float,
    asset_returns: np.ndarray,
    weights: np.ndarray,
    years: float,
    n_sims: int,
    target_mu_annual: float | None = None,
    target_sigma_annual: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate portfolio value paths by bootstrapping historical returns.

    Returns:
        paths: (n_days + 1, n_sims)
        portfolio_daily_returns: (n_days, n_sims)
        elapsed_seconds
    """
    if initial_investment <= 0:
        raise ValueError("initial_investment must be positive")
    if years <= 0 or n_sims <= 0:
        raise ValueError("years and n_sims must be positive")
    if asset_returns.ndim != 2:
        raise ValueError("asset_returns must be a 2D array")

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    n_days = max(int(years * TRADING_DAYS), 1)
    n_hist, n_assets = asset_returns.shape
    if weights.size != n_assets:
        raise ValueError("weights length must match number of assets")

    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, n_hist, size=(n_days, n_sims))

    start = time.perf_counter()
    sampled_asset_returns = asset_returns[sampled_idx]
    portfolio_daily = np.einsum("dsa,a->ds", sampled_asset_returns, weights)

    if target_mu_annual is not None and target_sigma_annual is not None:
        hist_mu = float(np.mean(portfolio_daily))
        hist_sigma = max(float(np.std(portfolio_daily)), 1e-8)
        target_mu_daily = target_mu_annual / TRADING_DAYS
        target_sigma_daily = max(target_sigma_annual / np.sqrt(TRADING_DAYS), 1e-8)

        z = (portfolio_daily - hist_mu) / hist_sigma
        portfolio_daily = z * target_sigma_daily + target_mu_daily

    growth = np.cumprod(1.0 + portfolio_daily, axis=0)
    paths = np.vstack([np.full((1, n_sims), initial_investment), initial_investment * growth])
    elapsed = time.perf_counter() - start

    return paths, portfolio_daily, elapsed


def terminal_returns(paths: np.ndarray) -> np.ndarray:
    """Compute terminal returns from simulated portfolio paths."""
    return (paths[-1] / paths[0]) - 1.0


def percentile_bands(paths: np.ndarray, percentiles: tuple[int, int, int] = (5, 50, 95)) -> dict[int, np.ndarray]:
    """Return percentile bands over time for path confidence visualization."""
    return {p: np.percentile(paths, p, axis=1) for p in percentiles}


def convergence_series(paths: np.ndarray) -> np.ndarray:
    """Running mean of terminal portfolio value to inspect convergence."""
    terminal_values = paths[-1]
    return np.cumsum(terminal_values) / np.arange(1, terminal_values.size + 1)
