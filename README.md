# MonteCarloXY v2.0

## Project Overview
MonteCarloXY v2.0 is a portfolio-ready quantitative risk analysis project that upgrades a beginner Monte Carlo model into a modular, professional data science application. It combines vectorized simulation, real market data, machine learning-based volatility forecasting, and an interactive Streamlit dashboard.

## Key Features
- **Vectorized Monte Carlo GBM engine** supporting up to 100,000 paths.
- **Real stock data integration** using `yfinance`.
- **Machine learning volatility forecast** via `RandomForestRegressor`.
- **Risk modeling metrics**: VaR (95%), CVaR, Sharpe Ratio, Maximum Drawdown.
- **Interactive Plotly charts** for paths, return distribution, and convergence.
- **Clean modular architecture** for data, simulation, risk, UI, and tests.

## Project Structure
```text
MonteCarloXY/
├── app.py
├── monte_engine.py
├── data_fetcher.py
├── risk_metrics.py
├── tests/
│   └── test_monte.py
├── requirements.txt
└── README.md
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App
```bash
streamlit run app.py
```

## Dashboard Outputs
- Monte Carlo price path chart
- Terminal return distribution histogram
- Convergence plot (running mean terminal price)
- Risk panel with VaR 95%, CVaR, Sharpe Ratio, and Max Drawdown

## Performance Notes
- On this implementation, **100,000 paths** are supported by vectorized NumPy operations.
- Typical runtime for simulation core is targeted to be **under ~3 seconds** depending on hardware.
- ML-based volatility forecasting uses rolling market features and a Random Forest model.

## Tests
```bash
pytest -q
```

## Example Screenshot
Run the dashboard and capture a screenshot after simulation:

`streamlit run app.py`

(See screenshot included in PR artifacts.)

## Disclaimer
This project is for educational and research purposes only and not financial advice.
