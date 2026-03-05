"""
This Streamlit application simulates correlated Geometric Brownian Motion (GBM)
asset paths, constructs portfolio paths from user-defined weights, and computes
portfolio risk metrics including VaR and CVaR from the loss distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def validate_correlation_matrix(corr: np.ndarray, tol: float = 1e-8) -> tuple[bool, str]:
    """Validate correlation matrix properties: square, symmetric, diag=1, PSD."""
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        return False, "Correlation matrix must be square."

    if not np.allclose(corr, corr.T, atol=tol):
        return False, "Correlation matrix must be symmetric."

    if not np.allclose(np.diag(corr), 1.0, atol=tol):
        return False, "Correlation matrix diagonal entries must all be 1."

    eigenvalues = np.linalg.eigvalsh(corr)
    if np.min(eigenvalues) < -tol:
        return False, "Correlation matrix must be positive semi-definite."

    if np.any(corr < -1.0 - tol) or np.any(corr > 1.0 + tol):
        return False, "Correlation values must be in [-1, 1]."

    return True, "Valid correlation matrix."


def simulate_correlated_gbm(
    initial_prices: np.ndarray,
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    years: float,
    n_simulations: int,
    steps_per_year: int = 252,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate correlated multi-asset GBM paths (fully vectorized).

    Returns:
        ndarray with shape (n_steps + 1, n_simulations, n_assets)
    """
    n_assets = initial_prices.shape[0]
    n_steps = max(int(years * steps_per_year), 1)
    dt = years / n_steps

    # Small diagonal bump allows numerically stable Cholesky for near-singular PSD matrices.
    stable_corr = correlation_matrix + 1e-12 * np.eye(n_assets)
    cholesky_l = np.linalg.cholesky(stable_corr)

    rng = np.random.default_rng(seed)
    independent_shocks = rng.normal(0.0, 1.0, size=(n_steps, n_simulations, n_assets))
    correlated_shocks = independent_shocks @ cholesky_l.T

    drift = (expected_returns - 0.5 * volatilities**2) * dt
    diffusion = volatilities * np.sqrt(dt) * correlated_shocks

    log_returns = drift[None, None, :] + diffusion
    cumulative_log_returns = np.cumsum(log_returns, axis=0)

    simulated_prices = initial_prices[None, None, :] * np.exp(cumulative_log_returns)
    initial_layer = np.broadcast_to(initial_prices, (1, n_simulations, n_assets))
    return np.concatenate([initial_layer, simulated_prices], axis=0)


def compute_portfolio_paths(asset_paths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute portfolio value paths from asset paths and weights."""
    return np.einsum("tna,a->tn", asset_paths, weights)


def calculate_risk_metrics(portfolio_paths: np.ndarray) -> dict[str, float | np.ndarray]:
    """Calculate portfolio metrics with VaR/CVaR based on loss distribution.

    Loss definition:
        L = V0 - VT
    """
    ending_values = portfolio_paths[-1, :]
    initial_values = portfolio_paths[0, :]
    losses = initial_values - ending_values

    var_95 = float(np.percentile(losses, 95))
    # Defensive tail handling: if the selected tail is empty, fallback to VaR.
    # This guarantees CVaR remains numeric and the UI can always render safely.
    tail_losses = losses[losses >= var_95]
    cvar_95 = float(np.mean(tail_losses)) if tail_losses.size > 0 else var_95

    return {
        "ending_values": ending_values,
        "losses": losses,
        "mean_ending_wealth": float(np.mean(ending_values)),
        "median_ending_wealth": float(np.median(ending_values)),
        "probability_of_loss": float(np.mean(losses > 0) * 100),
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def plot_portfolio_paths(
    portfolio_paths: np.ndarray,
    years: float,
    max_display_paths: int = 300,
    seed: int | None = None,
) -> go.Figure:
    """Plot portfolio paths; sample subset for high simulation counts."""
    n_steps, n_simulations = portfolio_paths.shape
    time_axis = np.linspace(0, years, n_steps)

    if n_simulations > 2000:
        rng = np.random.default_rng(seed)
        selected = rng.choice(n_simulations, size=min(max_display_paths, n_simulations), replace=False)
        display_paths = portfolio_paths[:, selected]
        title_suffix = f" (showing {display_paths.shape[1]} of {n_simulations} paths)"
    else:
        display_paths = portfolio_paths
        title_suffix = ""

    fig = go.Figure()
    for idx in range(display_paths.shape[1]):
        fig.add_trace(
            go.Scattergl(
                x=time_axis,
                y=display_paths[:, idx],
                mode="lines",
                line=dict(width=1),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Monte Carlo Portfolio Paths{title_suffix}",
        xaxis_title="Time (Years)",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_loss_distribution(losses: np.ndarray, var_95: float, cvar_95: float) -> go.Figure:
    """Plot loss distribution with VaR and CVaR thresholds."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=losses,
            nbinsx=70,
            name="Loss Distribution",
            marker=dict(color="#2E86DE"),
            opacity=0.8,
        )
    )

    fig.add_vline(
        x=var_95,
        line=dict(color="#E74C3C", width=2, dash="dash"),
        annotation_text="VaR (95%)",
        annotation_position="top right",
    )
    fig.add_vline(
        x=cvar_95,
        line=dict(color="#8E44AD", width=2, dash="dot"),
        annotation_text="CVaR (95%)",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Portfolio Loss Distribution",
        xaxis_title="Loss ($)",
        yaxis_title="Frequency",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_correlation_heatmap(correlation_matrix: np.ndarray, asset_labels: list[str]) -> go.Figure:
    """Plot asset correlation heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=asset_labels,
            y=asset_labels,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="Asset Correlation Matrix",
        xaxis_title="Assets",
        yaxis_title="Assets",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def format_currency(value: float) -> str:
    """Format numeric value as USD currency string."""
    return f"${value:,.2f}"


def _coerce_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame to numeric values while coercing invalid entries to NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def _invalid_cell_labels(invalid_mask: pd.DataFrame, raw_df: pd.DataFrame) -> list[str]:
    """Return human-readable labels for cells marked invalid in the mask."""
    invalid_cells: list[str] = []
    for row_idx, col_idx in np.argwhere(invalid_mask.to_numpy()):
        row_label = raw_df.index[row_idx]
        col_label = raw_df.columns[col_idx]
        invalid_value = raw_df.iat[row_idx, col_idx]
        invalid_cells.append(f"row '{row_label}', column '{col_label}' (value: {invalid_value!r})")
    return invalid_cells


def _validate_numeric_table(raw_df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, list[str]]:
    """Coerce table values to numeric and return validation errors with cell-level detail.

    Validation rules:
    - Non-numeric values are coerced to NaN via ``pd.to_numeric(..., errors='coerce')``.
    - Non-finite numeric values (e.g., inf) are also rejected.
    """
    numeric_df = _coerce_numeric_dataframe(raw_df)

    became_nan = numeric_df.isna() & ~raw_df.isna()
    non_finite = ~np.isfinite(numeric_df.to_numpy(dtype=float)) & ~numeric_df.isna().to_numpy()
    non_finite_mask = pd.DataFrame(non_finite, index=raw_df.index, columns=raw_df.columns)

    invalid_mask = became_nan | non_finite_mask
    if not invalid_mask.to_numpy().any():
        return numeric_df, []

    invalid_cells = _invalid_cell_labels(invalid_mask, raw_df)
    preview = ", ".join(invalid_cells[:4])
    suffix = "" if len(invalid_cells) <= 4 else ", ..."
    return numeric_df, [
        f"Invalid numeric entry detected in {table_name}. Please correct: {preview}{suffix}."
    ]


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="QuantEdge – Portfolio Risk Simulator", layout="wide")

    st.title("QuantEdge – Portfolio Risk Simulator")
    st.markdown(
        "Multi-asset correlated Monte Carlo simulation using GBM, "
        "Cholesky dependency modeling, and VaR/CVaR tail-risk analytics."
    )

    with st.sidebar:
        st.header("Portfolio & Simulation Inputs")

        n_assets = st.slider("Number of Assets", min_value=2, max_value=5, value=3, step=1)
        years = st.number_input("Time Horizon (Years)", min_value=0.25, max_value=50.0, value=10.0, step=0.25)
        n_simulations = st.slider("Number of Simulations", min_value=500, max_value=10_000, value=3_000, step=500)
        use_seed = st.checkbox("Use Random Seed", value=False)
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1, disabled=not use_seed)
        actual_seed = int(seed) if use_seed else None

        st.subheader("Per-Asset Parameters")
        asset_labels = [f"Asset {i + 1}" for i in range(n_assets)]

        default_prices = np.full(n_assets, 10_000.0)
        default_returns = np.full(n_assets, 8.0)
        default_vols = np.full(n_assets, 15.0)
        default_weights = np.full(n_assets, round(1 / n_assets, 4))

        asset_df = pd.DataFrame(
            {
                "Asset": asset_labels,
                "Initial Price ($)": default_prices,
                "Expected Return (%)": default_returns,
                "Volatility (%)": default_vols,
                "Weight": default_weights,
            }
        )

        edited_asset_df = st.data_editor(
            asset_df,
            use_container_width=True,
            hide_index=True,
            key="asset_table",
            disabled=["Asset"],
            num_rows="fixed",
        )

        st.subheader("Correlation Matrix")
        default_corr = np.full((n_assets, n_assets), 0.25)
        np.fill_diagonal(default_corr, 1.0)
        corr_df = pd.DataFrame(default_corr, columns=asset_labels, index=asset_labels)

        edited_corr_df = st.data_editor(
            corr_df,
            use_container_width=True,
            key="corr_table",
            num_rows="fixed",
        )

    asset_numeric_cols = ["Initial Price ($)", "Expected Return (%)", "Volatility (%)", "Weight"]
    asset_numeric_df, asset_errors = _validate_numeric_table(
        edited_asset_df[asset_numeric_cols], "Per-Asset Parameters table"
    )
    corr_numeric_df, corr_errors = _validate_numeric_table(edited_corr_df, "Correlation Matrix table")

    if asset_errors:
        st.error(asset_errors[0])
        st.stop()

    if corr_errors:
        st.error(corr_errors[0])
        st.stop()

    initial_prices = asset_numeric_df["Initial Price ($)"].to_numpy(dtype=float)
    expected_returns = asset_numeric_df["Expected Return (%)"].to_numpy(dtype=float) / 100.0
    volatilities = asset_numeric_df["Volatility (%)"].to_numpy(dtype=float) / 100.0
    weights = asset_numeric_df["Weight"].to_numpy(dtype=float)
    correlation_matrix = corr_numeric_df.to_numpy(dtype=float)

    weights_sum = float(weights.sum())
    if not np.isclose(weights_sum, 1.0, atol=1e-6):
        st.warning(f"Portfolio weights currently sum to {weights_sum:.6f}. Weights must sum to 1.00.")

    weights_valid = np.isclose(weights_sum, 1.0, atol=1e-6) and np.all(weights >= 0)
    if not weights_valid:
        st.stop()

    corr_valid, corr_message = validate_correlation_matrix(correlation_matrix)
    if not corr_valid:
        st.error(corr_message)
        st.stop()

    asset_paths = simulate_correlated_gbm(
        initial_prices=initial_prices,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        years=years,
        n_simulations=n_simulations,
        seed=actual_seed,
    )

    portfolio_paths = compute_portfolio_paths(asset_paths, weights)
    metrics = calculate_risk_metrics(portfolio_paths)

    st.subheader("Portfolio Risk Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean Ending Wealth", format_currency(metrics["mean_ending_wealth"]))
    col2.metric("Median Ending Wealth", format_currency(metrics["median_ending_wealth"]))
    col3.metric("Probability of Loss", f"{metrics['probability_of_loss']:.2f}%")
    col4.metric(
        "VaR (95%)",
        format_currency(metrics["var_95"]),
        help="Value at Risk (95%) is the loss threshold exceeded only 5% of the time.",
    )
    col5.metric(
        "CVaR (95%)",
        format_currency(metrics["cvar_95"]),
        help="CVaR (95%) is the average loss in the worst 5% tail beyond VaR.",
    )

    st.markdown("---")

    top_left, top_right = st.columns(2)
    with top_left:
        st.plotly_chart(
            plot_portfolio_paths(portfolio_paths, years, seed=actual_seed),
            use_container_width=True,
        )

    with top_right:
        st.plotly_chart(
            plot_loss_distribution(metrics["losses"], metrics["var_95"], metrics["cvar_95"]),
            use_container_width=True,
        )

    st.plotly_chart(
        plot_correlation_heatmap(correlation_matrix, asset_labels),
        use_container_width=True,
    )

    st.caption(
        "Volatility reflects uncertainty in asset returns. "
        "Higher volatility implies wider swings in outcomes."
    )


if __name__ == "__main__":
    main()