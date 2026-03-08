"""Vectorized Monte Carlo simulation engine for MonteCarloXY v2.0."""

from __future__ import annotations

import time

import numpy as np


TRADING_DAYS = 252


def simulate_price_paths(
    s0: float,
    mu: float,
    sigma: float,
    years: float,
    n_steps: int,
    n_sims: int,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Simulate GBM price paths using full NumPy vectorization.

    Returns:
        paths: ndarray with shape (n_steps + 1, n_sims)
        elapsed_seconds: execution time in seconds
    """
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if years <= 0 or n_steps <= 0 or n_sims <= 0:
        raise ValueError("years, n_steps, and n_sims must be positive")

    dt = years / n_steps
    rng = np.random.default_rng(seed)

    start = time.perf_counter()
    shocks = rng.standard_normal((n_steps, n_sims))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros((1, n_sims)), np.cumsum(increments, axis=0)])
    paths = s0 * np.exp(log_paths)
    elapsed = time.perf_counter() - start

    return paths, elapsed


def terminal_returns(paths: np.ndarray) -> np.ndarray:
    """Compute terminal returns from simulated paths."""
    return paths[-1] / paths[0] - 1.0


def convergence_series(paths: np.ndarray) -> np.ndarray:
    """Running mean of terminal prices for convergence visualization."""
    terminal_prices = paths[-1]
    return np.cumsum(terminal_prices) / np.arange(1, terminal_prices.size + 1)
