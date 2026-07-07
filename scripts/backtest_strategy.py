import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (12, 7)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2024-01-01"
BACKTEST_START = "2025-01-01"
BACKTEST_END = "2026-06-30"
STRATEGY_WEIGHTS = np.array([0.0, 0.5466, 0.4534])
BENCHMARK_WEIGHTS = np.array([0.0, 0.4, 0.6])


def download_prices():
    price_frames = {}
    for ticker in TICKERS:
        data = yf.download(ticker, start=START_DATE, end=BACKTEST_END, progress=False, auto_adjust=True)
        price_series = data["Close"].dropna() if "Close" in data.columns else data.iloc[:, 0].dropna()
        price_frames[ticker] = price_series
    prices = pd.concat(price_frames, axis=1).dropna()
    prices.columns = TICKERS
    return prices


def simulate_portfolio(returns, weights):
    current_value = 1.0
    current_weights = np.array(weights, dtype=float)
    values = []
    dates = []

    months = pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq="MS")
    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        month_returns = returns.loc[month_start:month_end]
        if month_returns.empty:
            continue
        for day, row in month_returns.iterrows():
            daily_return = float(np.dot(current_weights, row.to_numpy()))
            current_value *= 1.0 + daily_return
            dates.append(day)
            values.append(current_value)
        current_weights = np.array(weights, dtype=float)

    if not values:
        raise ValueError("No portfolio values generated")

    series = pd.Series(values, index=dates)
    return series


def compute_metrics(series):
    daily_returns = series.pct_change().dropna()
    total_return = float(series.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (252.0 / len(daily_returns)) - 1.0) if len(daily_returns) > 0 else 0.0
    sharpe_ratio = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0
    cumulative = np.maximum.accumulate(series)
    drawdown = (cumulative - series) / cumulative
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }


def main():
    prices = download_prices()
    returns = prices.pct_change().dropna()
    backtest_returns = returns.loc[BACKTEST_START:BACKTEST_END]

    strategy_series = simulate_portfolio(backtest_returns, STRATEGY_WEIGHTS)
    benchmark_series = simulate_portfolio(backtest_returns, BENCHMARK_WEIGHTS)

    strategy_metrics = compute_metrics(strategy_series)
    benchmark_metrics = compute_metrics(benchmark_series)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(strategy_series.index, strategy_series, label="Strategy (Task 4 allocation)", linewidth=2)
    ax.plot(benchmark_series.index, benchmark_series, label="Benchmark (60% SPY / 40% BND)", linewidth=2)
    ax.set_title("Strategy vs. Benchmark Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plot_path = OUTPUT_PATH / "strategy_backtest_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics_df = pd.DataFrame(
        {
            "Portfolio": ["Strategy", "Benchmark"],
            "Total Return": [strategy_metrics["Total Return"], benchmark_metrics["Total Return"]],
            "Annualized Return": [strategy_metrics["Annualized Return"], benchmark_metrics["Annualized Return"]],
            "Sharpe Ratio": [strategy_metrics["Sharpe Ratio"], benchmark_metrics["Sharpe Ratio"]],
            "Max Drawdown": [strategy_metrics["Max Drawdown"], benchmark_metrics["Max Drawdown"]],
        }
    )
    metrics_df["Total Return"] = metrics_df["Total Return"].map(lambda x: f"{x:.2%}")
    metrics_df["Annualized Return"] = metrics_df["Annualized Return"].map(lambda x: f"{x:.2%}")
    metrics_df["Sharpe Ratio"] = metrics_df["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
    metrics_df["Max Drawdown"] = metrics_df["Max Drawdown"].map(lambda x: f"{x:.2%}")

    md_lines = [
        "# Strategy Backtest Summary",
        "",
        "The backtest uses the last available period of the dataset from 2025-01-01 through 2026-06-30, with the Task 4 portfolio allocation rebalanced monthly against a static 60% SPY / 40% BND benchmark.",
        "",
        "## Performance Metrics",
        "",
        "| Portfolio | Total Return | Annualized Return | Sharpe Ratio | Max Drawdown |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics_df.iterrows():
        md_lines.append(
            f"| {row['Portfolio']} | {row['Total Return']} | {row['Annualized Return']} | {row['Sharpe Ratio']} | {row['Max Drawdown']} |"
        )

    if strategy_metrics["Total Return"] > benchmark_metrics["Total Return"]:
        conclusion = (
            "The strategy outperformed the benchmark over this backtest window, which suggests the portfolio allocation from Task 4 was directionally useful during this period. "
            "That said, the result should be treated as an initial signal rather than proof of enduring edge because the sample is short and the forecast-based allocation was not adjusted for changing market regimes."
        )
    else:
        conclusion = (
            "The strategy did not beat the benchmark over this backtest window, which suggests the Task 4 allocation was not strong enough to justify the model-driven approach on its own. "
            "This does not invalidate the earlier forecast, but it does indicate that the simple balanced benchmark was more robust in this historical period and that further refinement is needed."
        )

    md_lines.extend([
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "This backtest is useful as a first-pass validation, but it has limitations. The window is relatively short, it ignores transaction costs and taxes, and it relies on historical returns that may not reflect future market conditions. It also uses a single forecast-implied allocation rather than a more dynamic regime-aware process.",
    ])

    summary_path = OUTPUT_PATH / "backtest_results.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved cumulative returns plot to {plot_path}")
    print(f"Saved performance summary to {summary_path}")
    print("\nPerformance Metrics")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
