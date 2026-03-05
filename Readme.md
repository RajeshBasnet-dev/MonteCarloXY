
# MonteCarloX – Multi-Asset Monte Carlo Portfolio Risk Simulator

MonteCarloX is a quantitative finance dashboard that simulates correlated multi-asset portfolio growth using Monte Carlo methods and Geometric Brownian Motion (GBM).

The application models asset correlation via Cholesky decomposition and computes portfolio-level tail risk metrics including Value at Risk (VaR) and Conditional Value at Risk (CVaR).

---

## 🚀 Features

- Multi-asset Monte Carlo simulation (2–5 assets)
- Correlated GBM modeling using Cholesky factorization
- Fully vectorized NumPy architecture (no simulation loops)
- Portfolio construction with user-defined weights
- Loss-based VaR (95%) and CVaR (Expected Shortfall)
- Interactive risk dashboard built with Streamlit
- Correlation heatmap visualization
- High-performance Plotly visualizations

---

## 📊 Mathematical Framework

### Geometric Brownian Motion

For each asset:

Sₜ = S₀ exp((μ − ½σ²)t + σWₜ)

Where:
- μ = Expected return
- σ = Volatility
- Wₜ = Brownian motion

---

### Correlated Shocks

Correlation is introduced using Cholesky decomposition:

Σ = LLᵀ

Correlated shocks:

Z_correlated = LZ

---

### Portfolio Construction

Vₜ = Σ wᵢ Sₜ⁽ⁱ⁾

Where:
- wᵢ = Asset weights
- Σ wᵢ = 1

---

### Risk Metrics

Loss definition:

L = V₀ − Vₜ

- VaR (95%): 95th percentile of loss distribution
- CVaR (95%): Mean loss beyond VaR threshold
- Probability of Loss
- Mean & Median Ending Wealth

---

## 🛠 Installation

### 1. Clone Repository

```bash
git clone https://github.com/RajeshBasnet-dev/MonteCarloXY
cd MonteCarloXY
````

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
streamlit run app.py
```

---

## 📈 Technical Highlights

* Vectorized 3D simulation arrays
* Efficient matrix algebra using NumPy
* Positive semi-definite correlation validation
* Loss-based tail risk modeling
* Scalable visualization using Plotly ScatterGL

---

## 🎯 Use Cases

* Portfolio stress testing
* Diversification analysis
* Risk education
* Quantitative finance experimentation
* Tail-risk analysis

---

## ⚠ Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice.



---

