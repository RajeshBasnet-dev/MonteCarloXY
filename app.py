"""Streamlit dashboard for MonteCarloXY v2.0."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_fetcher import estimate_drift_vol, fetch_stock_data, volatility_forecast_with_source
from monte_engine import convergence_series, simulate_price_paths, terminal_returns
from risk_metrics import calculate_risk_metrics

st.set_page_config(page_title="MonteCarloXY", layout="wide")
st.title("MonteCarloXY – Nepal Stock Risk Simulation Engine")
st.caption("Vectorized Monte Carlo + ML volatility forecast + interactive risk analytics")

with st.sidebar:
    st.header("Simulation Controls")
    ticker = st.text_input("Ticker", value="NIBL.NS")
    years = st.slider("Time Horizon (years)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    n_sims = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)
    n_steps = st.slider("Time Steps", min_value=30, max_value=756, value=252, step=21)
    manual_price = st.number_input("Manual Stock Price Override (0 = auto)", min_value=0.0, value=0.0, step=1.0)
    manual_vol = st.slider("Manual Volatility Override (0 = ML/historical)", min_value=0.0, max_value=1.5, value=0.0, step=0.01)

run = st.button("Run Simulation", type="primary")

if run:
    with st.spinner("Fetching market data and simulating paths..."):
        data = fetch_stock_data(ticker)
        s0, mu, sigma_hist, _ = estimate_drift_vol(data)
        sigma_ml, sigma_source = volatility_forecast_with_source(data)

        s0_used = manual_price if manual_price > 0 else s0
        sigma_used = manual_vol if manual_vol > 0 else sigma_ml

        paths, elapsed = simulate_price_paths(
            s0=s0_used,
            mu=mu,
            sigma=sigma_used,
            years=years,
            n_steps=n_steps,
            n_sims=n_sims,
            seed=42,
        )

        rets = terminal_returns(paths)
        metrics = calculate_risk_metrics(rets)
        conv = convergence_series(paths)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Value at Risk (95%)", f"{metrics['VaR_95']:.2%}")
    c2.metric("Conditional VaR", f"{metrics['CVaR_95']:.2%}")
    c3.metric("Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.3f}")
    c4.metric("Maximum Drawdown", f"{metrics['Max_Drawdown']:.2%}")

    st.info(
        f"Fetched {ticker} | S0={s0:.2f}, mu={mu:.2%}, hist sigma={sigma_hist:.2%}, "
        f"forecast sigma={sigma_ml:.2%} ({sigma_source}), used sigma={sigma_used:.2%} | runtime={elapsed:.3f}s"
    )

    if sigma_source != "ml_random_forest":
        st.warning("scikit-learn is unavailable or insufficient data for ML training; using historical volatility fallback.")

    t_axis = np.linspace(0, years, n_steps + 1)
    show_paths = min(300, n_sims)
    idx = np.random.default_rng(42).choice(n_sims, show_paths, replace=False)

    fig_paths = go.Figure()
    for i in idx:
        fig_paths.add_trace(
            go.Scattergl(x=t_axis, y=paths[:, i], mode="lines", line=dict(width=1), opacity=0.2, showlegend=False)
        )
    fig_paths.update_layout(title="Monte Carlo Price Paths", xaxis_title="Years", yaxis_title="Price", template="plotly_white")

    fig_hist = go.Figure(
        data=[go.Histogram(x=rets, nbinsx=60, marker_color="#1f77b4", opacity=0.85)]
    )
    fig_hist.update_layout(
        title="Terminal Return Distribution",
        xaxis_title="Return",
        yaxis_title="Frequency",
        template="plotly_white",
    )

    fig_conv = go.Figure(
        data=[go.Scatter(y=conv, mode="lines", line=dict(color="#e67e22", width=2), name="Running Mean")]
    )
    fig_conv.update_layout(
        title="Convergence Plot (Running Mean Terminal Price)",
        xaxis_title="Number of Simulations",
        yaxis_title="Mean Terminal Price",
        template="plotly_white",
    )

    st.plotly_chart(fig_paths, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        st.plotly_chart(fig_conv, use_container_width=True)
else:
    st.write("Set parameters in the sidebar and click **Run Simulation**.")
