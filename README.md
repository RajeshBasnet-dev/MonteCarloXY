# MonteCarloXY — AI-Powered Portfolio Risk & Simulation Dashboard

## Overview
MonteCarloXY is a Streamlit-based quantitative finance project that combines:
- real market data ingestion (`yfinance`),
- vectorized Monte Carlo portfolio simulation,
- machine learning-assisted parameter forecasting,
- institutional-style risk analytics and Plotly visualization.

This project is designed for a machine learning / data science portfolio and keeps Monte Carlo simulation as the core engine while upgrading it into a professional dashboard.

## Core Methodology

### 1) Market Data Pipeline
1. Users input one or more tickers (comma-separated).
2. Historical adjusted close prices are fetched from Yahoo Finance.
3. Daily returns are computed and aligned across assets.
4. Portfolio returns are derived from user-specified allocations.

### 2) AI Forecasting Layer
A Random Forest model is trained on historical portfolio features:
- rolling mean returns (5, 21 days),
- rolling volatility (5, 21 days),
- momentum (10-day compounded return).

The models predict:
- next-period expected return,
- next-period volatility.

If data is insufficient or ML dependencies are unavailable, the system falls back to historical estimates.

### 3) Monte Carlo Simulation Engine
Simulation is implemented with NumPy vectorization for speed and scalability:
- historical bootstrapping samples real return distributions,
- thousands of paths are simulated efficiently,
- AI-predicted drift/volatility recalibrate sampled returns,
- outputs include full portfolio paths and terminal return distributions.

### 4) Risk Analytics
The dashboard computes:
- **Value at Risk (VaR 95%)**,
- **Conditional VaR (CVaR 95%)**,
- **Expected return**,
- **Volatility (terminal + annualized)**,
- **Sharpe ratio** (risk-adjusted return),
- **Maximum drawdown**.

## Dashboard Sections
1. **Stock Selection** – choose tickers and historical lookback.
2. **Simulation Parameters** – initial capital, horizon, simulations, risk-free rate.
3. **Portfolio Allocation** – per-asset user-defined weights.
4. **AI Prediction Insights** – model forecasts and source indicator.
5. **Risk Analytics** – key risk metrics panel.
6. **Simulation Results** – path chart, confidence bands, distribution charts, convergence.
7. **Data Snapshot** – latest fetched market data.

## Visualizations (Plotly)
- Simulated growth paths with 5th/50th/95th percentile bands
- Distribution of final portfolio values
- Histogram of simulated terminal returns
- Risk distribution box plot
- Convergence plot of running mean final value

## Project Structure
```text
MonteCarloXY/
├── app.py               # Streamlit dashboard UI and orchestration
├── data_fetcher.py      # Data loading, return computation, ML feature engineering
├── monte_engine.py      # Vectorized portfolio simulation engine
├── risk_metrics.py      # Risk metric calculations
├── tests/
│   └── test_monte.py    # Core unit tests
├── requirements.txt
└── README.md
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App
```bash
streamlit run app.py
```

## Run Tests
```bash
pytest -q
```

## Notes
- This project is for educational/research use and does not constitute financial advice.
- Historical performance and simulation outputs are not guarantees of future returns.

## ML Reliability and Fallback Behavior
- The app validates data sufficiency before ML training (minimum historical rows + minimum engineered feature rows).
- Rolling features are generated first and `dropna()` is applied only after full feature/target construction to avoid unnecessary data loss.
- If scikit-learn is missing or data is insufficient, the app automatically switches to historical volatility and labels the prediction source in the UI.
- Debug telemetry is exposed (historical rows, feature rows, scikit-learn availability, selected prediction mode).
