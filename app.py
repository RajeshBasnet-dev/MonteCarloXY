"""Streamlit AI-powered portfolio risk and simulation dashboard."""

from __future__ import annotations

import logging

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_fetcher import ai_forecast_return_vol, compute_returns, fetch_stock_data, portfolio_return_series
from monte_engine import convergence_series, percentile_bands, simulate_portfolio_paths, terminal_returns
from risk_metrics import calculate_risk_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

st.set_page_config(page_title="MonteCarloXY", layout="wide")
st.title("MonteCarloXY — AI-Powered Portfolio Risk & Monte Carlo Dashboard")
st.caption("Real market data + machine learning forecasts + vectorized simulation analytics")

if "monte_engine" not in st.session_state:
    st.session_state["monte_engine"] = None

with st.sidebar:
    st.header("1) Stock Selection")
    tickers_raw = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOGL")
    period = st.selectbox("Historical Lookback", ["1y", "2y", "3y", "5y", "10y"], index=3)

    st.header("2) Simulation Parameters")
    initial_investment = st.number_input("Initial Investment ($)", min_value=100.0, value=10000.0, step=500.0)
    years = st.slider("Investment Horizon (years)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    n_sims = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=20000, step=1000)
    risk_free_rate = st.slider("Risk-Free Rate", min_value=0.0, max_value=0.1, value=0.02, step=0.005)

run = st.button("Run AI-Powered Simulation", type="primary")

if run:
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        st.error("Please provide at least one valid ticker.")
        st.stop()

    with st.spinner("Downloading market data, training ML models, and simulating portfolio paths..."):
        prices = fetch_stock_data(tickers, period=period)
        returns_df = compute_returns(prices)

        st.subheader("3) Portfolio Allocation")
        alloc_cols = st.columns(len(tickers))
        raw_weights = []
        for i, ticker in enumerate(tickers):
            raw_weights.append(alloc_cols[i].number_input(f"{ticker} weight", min_value=0.0, max_value=1.0, value=1.0 / len(tickers), step=0.05))

        weights = np.array(raw_weights, dtype=float)
        if weights.sum() <= 0:
            st.error("Portfolio weights must sum to a positive value.")
            st.stop()
        weights = weights / weights.sum()

        portfolio_hist_returns = portfolio_return_series(returns_df[tickers], weights)
        ai_forecast = ai_forecast_return_vol(portfolio_hist_returns)

        paths, sim_daily_returns, elapsed = simulate_portfolio_paths(
            initial_investment=initial_investment,
            asset_returns=returns_df[tickers].to_numpy(),
            weights=weights,
            years=years,
            n_sims=n_sims,
            target_mu_annual=float(ai_forecast["mu_annual"]),
            target_sigma_annual=float(ai_forecast["sigma_annual"]),
            seed=42,
        )

        metrics = calculate_risk_metrics(paths, sim_daily_returns, confidence=0.95, risk_free_rate=risk_free_rate)
        term_rets = terminal_returns(paths)
        bands = percentile_bands(paths, percentiles=(5, 50, 95))
        conv = convergence_series(paths)

        st.session_state["monte_engine"] = {
            "tickers": tickers,
            "years": years,
            "prices": prices,
            "ai_forecast": ai_forecast,
            "paths": paths,
            "metrics": metrics,
            "term_rets": term_rets,
            "bands": bands,
            "conv": conv,
            "elapsed": elapsed,
        }

monte_state = st.session_state.get("monte_engine")

if monte_state is None:
    st.write("Configure the dashboard from the sidebar and click **Run AI-Powered Simulation**.")
else:
    st.subheader("4) AI Prediction Insights")
    ai_forecast = monte_state["ai_forecast"]
    metrics = monte_state["metrics"]
    paths = monte_state["paths"]
    bands = monte_state["bands"]
    term_rets = monte_state["term_rets"]
    conv = monte_state["conv"]
    years = float(monte_state["years"])
    elapsed = float(monte_state["elapsed"])

    a1, a2, a3 = st.columns(3)
    a1.metric("AI Annual Return Estimate", f"{ai_forecast['mu_annual']:.2%}")
    a2.metric("AI Annual Volatility Estimate", f"{ai_forecast['sigma_annual']:.2%}")
    a3.metric("Forecast Source", str(ai_forecast["mode_label"]))

    st.caption(
        f"Debug | price rows: {ai_forecast['n_hist_rows']} | feature rows: {ai_forecast['n_feature_rows']} | "
        f"scikit-learn available: {ai_forecast['sklearn_available']}"
    )

    st.subheader("5) Portfolio Risk Analytics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR (95%)", f"{metrics['VaR_95']:.2%}")
    c2.metric("CVaR (95%)", f"{metrics['CVaR_95']:.2%}")
    c3.metric("Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.2f}")
    c4.metric("Maximum Drawdown", f"{metrics['Max_Drawdown']:.2%}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Expected Terminal Return", f"{metrics['Expected_Return']:.2%}")
    r2.metric("Annualized Volatility", f"{metrics['Annualized_Volatility']:.2%}")
    r3.metric("Simulation Runtime", f"{elapsed:.3f}s")

    t_axis = np.linspace(0, years, paths.shape[0])
    sample_idx = np.random.default_rng(42).choice(paths.shape[1], size=min(300, paths.shape[1]), replace=False)

    st.subheader("6) Monte Carlo Simulation Results")

    fig_paths = go.Figure()
    for i in sample_idx:
        fig_paths.add_trace(go.Scattergl(x=t_axis, y=paths[:, i], mode="lines", line=dict(width=1), opacity=0.15, showlegend=False))
    fig_paths.add_trace(go.Scatter(x=t_axis, y=bands[50], mode="lines", line=dict(color="#1f77b4", width=3), name="Median"))
    fig_paths.add_trace(go.Scatter(x=t_axis, y=bands[95], mode="lines", line=dict(color="#2ca02c", width=2, dash="dash"), name="95th pct"))
    fig_paths.add_trace(go.Scatter(x=t_axis, y=bands[5], mode="lines", line=dict(color="#d62728", width=2, dash="dash"), name="5th pct"))
    fig_paths.update_layout(title="Simulated Portfolio Growth Paths with Percentile Bands", xaxis_title="Years", yaxis_title="Portfolio Value ($)", template="plotly_white")

    fig_terminal = go.Figure(data=[go.Histogram(x=paths[-1], nbinsx=70, marker_color="#1f77b4", opacity=0.85)])
    fig_terminal.update_layout(title="Distribution of Final Portfolio Values", xaxis_title="Final Value ($)", yaxis_title="Frequency", template="plotly_white")

    fig_returns = go.Figure(data=[go.Histogram(x=term_rets, nbinsx=70, marker_color="#9467bd", opacity=0.85)])
    fig_returns.update_layout(title="Histogram of Simulated Terminal Returns", xaxis_title="Terminal Return", yaxis_title="Frequency", template="plotly_white")

    fig_risk = go.Figure(data=[go.Box(y=term_rets, boxpoints="outliers", marker_color="#ff7f0e", name="Risk Distribution")])
    fig_risk.update_layout(title="Risk Distribution (Terminal Returns)", yaxis_title="Return", template="plotly_white")

    fig_conv = go.Figure(data=[go.Scatter(y=conv, mode="lines", line=dict(color="#2ca02c", width=2))])
    fig_conv.update_layout(title="Convergence of Mean Final Portfolio Value", xaxis_title="Number of Simulations", yaxis_title="Running Mean Final Value", template="plotly_white")

    st.plotly_chart(fig_paths, width="stretch")
    p1, p2 = st.columns(2)
    p1.plotly_chart(fig_terminal, width="stretch")
    p2.plotly_chart(fig_returns, width="stretch")

    p3, p4 = st.columns(2)
    p3.plotly_chart(fig_risk, width="stretch")
    p4.plotly_chart(fig_conv, width="stretch")

    st.subheader("7) Data Snapshot")
    st.dataframe(monte_state["prices"].tail(10), width="stretch")
