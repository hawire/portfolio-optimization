import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT

START_DATE = "2015-01-01"
END_DATE = "2026-06-30"
TICKERS = ["TSLA", "BND", "SPY"]
RISK_FREE_RATE = 0.0


def download_price_data():
    price_frames = {}
    for ticker in TICKERS:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
        price_frames[ticker] = data[price_col].dropna()
    prices = pd.concat(price_frames, axis=1).dropna()
    prices.columns = TICKERS
    return prices


def get_tsla_forecast_return(prices):
    split_date = pd.Timestamp("2024-12-31")
    tsla_prices = prices["TSLA"].copy()
    train_series = tsla_prices[tsla_prices.index <= split_date].to_numpy()

    model = ARIMA(train_series, order=(1, 1, 1))
    fitted_model = model.fit()
    forecast = fitted_model.get_forecast(steps=252).predicted_mean
    forecast_array = np.asarray(forecast)
    forecast_returns = np.diff(np.log(forecast_array))
    annualized_return = np.mean(forecast_returns) * 252
    return annualized_return


def annualized_expected_returns(returns_df, tsla_forecast_return):
    mean_daily_returns = returns_df.mean()
    annualized = mean_daily_returns * 252
    annualized["TSLA"] = tsla_forecast_return
    return annualized


def portfolio_return(weights, expected_returns):
    return float(np.dot(weights, expected_returns))


def portfolio_volatility(weights, cov_matrix):
    variance = float(weights @ cov_matrix @ weights)
    return float(np.sqrt(max(variance, 1e-12)))


def neg_sharpe(weights, expected_returns, cov_matrix):
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return -(ret - RISK_FREE_RATE) / vol if vol > 0 else -np.inf


def optimize_weights(expected_returns, cov_matrix, target_return=None):
    n_assets = len(expected_returns)
    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return})

    initial_guess = np.full(n_assets, 1.0 / n_assets)

    def objective(w):
        return portfolio_volatility(w, cov_matrix) ** 2

    result = minimize(objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def simulate_frontier(expected_returns, cov_matrix):
    n_assets = len(expected_returns)
    min_ret = float(expected_returns.min())
    max_ret = float(expected_returns.max())
    target_returns = np.linspace(min_ret, max_ret, 50)

    frontier = []
    for target in target_returns:
        try:
            weights = optimize_weights(expected_returns, cov_matrix, target_return=target)
            ret = portfolio_return(weights, expected_returns)
            vol = portfolio_volatility(weights, cov_matrix)
            frontier.append((ret, vol, weights))
        except Exception:
            continue
    return frontier


def main():
    prices = download_price_data()
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252

    tsla_forecast_return = get_tsla_forecast_return(prices)
    expected_returns = annualized_expected_returns(returns, tsla_forecast_return)

    # Min volatility and max Sharpe portfolios
    min_vol_weights = optimize_weights(expected_returns, cov_matrix)
    min_vol_return = portfolio_return(min_vol_weights, expected_returns)
    min_vol_vol = portfolio_volatility(min_vol_weights, cov_matrix)

    n_assets = len(expected_returns)
    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    initial_guess = np.full(n_assets, 1.0 / n_assets)
    max_sharpe_result = minimize(
        lambda w: neg_sharpe(w, expected_returns, cov_matrix),
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    max_sharpe_weights = max_sharpe_result.x
    max_sharpe_return = portfolio_return(max_sharpe_weights, expected_returns)
    max_sharpe_vol = portfolio_volatility(max_sharpe_weights, cov_matrix)
    max_sharpe_ratio = (max_sharpe_return - RISK_FREE_RATE) / max_sharpe_vol if max_sharpe_vol > 0 else 0.0

    frontier = simulate_frontier(expected_returns, cov_matrix)
    frontier_returns = [point[0] for point in frontier]
    frontier_vols = [point[1] for point in frontier]

    # Monte Carlo portfolios for context
    rng = np.random.default_rng(42)
    n_portfolios = 4000
    random_returns = []
    random_vols = []
    random_weights_list = []
    for _ in range(n_portfolios):
        weights = rng.dirichlet(np.ones(n_assets), size=1)[0]
        random_returns.append(portfolio_return(weights, expected_returns))
        random_vols.append(portfolio_volatility(weights, cov_matrix))
        random_weights_list.append(weights)

    # Plot efficient frontier
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(random_vols, random_returns, s=10, color="lightgray", alpha=0.3, label="Random Portfolios")
    axes[0].plot(frontier_vols, frontier_returns, color="royalblue", linewidth=2, label="Efficient Frontier")
    axes[0].scatter(min_vol_vol, min_vol_return, marker="*", s=250, color="red", label="Min Volatility")
    axes[0].scatter(max_sharpe_vol, max_sharpe_return, marker="D", s=220, color="green", label="Max Sharpe")
    axes[0].set_title("Efficient Frontier for TSLA/BND/SPY")
    axes[0].set_xlabel("Volatility (Annualized)")
    axes[0].set_ylabel("Expected Annual Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Covariance heatmap
    cov_heatmap = cov_matrix.copy()
    sns.heatmap(cov_heatmap, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title("Covariance Matrix Heatmap")
    axes[1].set_xlabel("Asset")
    axes[1].set_ylabel("Asset")
    plt.tight_layout()
    frontier_path = OUTPUT_PATH / "efficient_frontier.png"
    heatmap_path = OUTPUT_PATH / "covariance_heatmap.png"
    plt.savefig(frontier_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cov_heatmap, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax2)
    ax2.set_title("Covariance Matrix Heatmap")
    ax2.set_xlabel("Asset")
    ax2.set_ylabel("Asset")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    recommended_weights = max_sharpe_weights
    recommended_weights_df = pd.DataFrame(
        {
            "Asset": TICKERS,
            "Weight": recommended_weights,
        }
    )
    recommended_weights_df["Weight"] = recommended_weights_df["Weight"].round(4)

    summary = f"""# Portfolio Optimization Recommendation

Using the forecast-implied expected return for TSLA and historical average daily returns for BND and SPY, the optimized portfolio suggests a risk-adjusted allocation that tilts meaningfully toward the defensive assets while keeping a modest exposure to TSLA.

## Recommended Portfolio (Maximum Sharpe Ratio)
- TSLA: {recommended_weights_df.loc[recommended_weights_df['Asset']=='TSLA','Weight'].iloc[0]:.2%}
- BND: {recommended_weights_df.loc[recommended_weights_df['Asset']=='BND','Weight'].iloc[0]:.2%}
- SPY: {recommended_weights_df.loc[recommended_weights_df['Asset']=='SPY','Weight'].iloc[0]:.2%}

## Portfolio Metrics
- Expected Annual Return: {max_sharpe_return:.2%}
- Expected Volatility: {max_sharpe_vol:.2%}
- Sharpe Ratio: {max_sharpe_ratio:.3f}

## Why This Portfolio
The maximum Sharpe ratio portfolio is recommended because it offers the best balance between return and risk rather than simply maximizing return regardless of volatility. The allocation preserves some exposure to TSLA to benefit from the forecast-driven view, but it reduces concentration risk by placing the majority of capital in BND and SPY, which provide stability and diversification.
"""

    summary_path = OUTPUT_PATH / "portfolio_recommendation.md"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Saved frontier plot to {frontier_path}")
    print(f"Saved covariance heatmap to {heatmap_path}")
    print(f"Saved recommendation summary to {summary_path}")
    print("\nMaximum Sharpe Portfolio:")
    print(recommended_weights_df.to_string(index=False))
    print(f"Expected annual return: {max_sharpe_return:.2%}")
    print(f"Expected volatility: {max_sharpe_vol:.2%}")
    print(f"Sharpe ratio: {max_sharpe_ratio:.3f}")


if __name__ == "__main__":
    main()
