import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from monte_engine import simulate_price_paths, terminal_returns
from risk_metrics import calculate_risk_metrics


def test_simulation_output_dimensions():
    paths, elapsed = simulate_price_paths(100, 0.12, 0.25, 1, 252, 5000, seed=1)
    assert paths.shape == (253, 5000)
    assert elapsed >= 0


def test_risk_metrics_dictionary_keys():
    fake_returns = np.array([-0.2, -0.1, 0.0, 0.05, 0.1])
    metrics = calculate_risk_metrics(fake_returns)
    assert {"VaR_95", "CVaR_95", "Sharpe_Ratio", "Max_Drawdown"}.issubset(metrics.keys())
    assert metrics["CVaR_95"] >= metrics["VaR_95"]


def test_engine_stability_no_nans():
    paths, _ = simulate_price_paths(150, 0.08, 0.3, 2, 504, 10000, seed=42)
    rets = terminal_returns(paths)
    assert np.isfinite(paths).all()
    assert np.isfinite(rets).all()
