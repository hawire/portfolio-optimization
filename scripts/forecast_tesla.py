import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "TSLA_processed.csv"
OUTPUT_PATH = PROJECT_ROOT


def load_data():
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        print(f"Loaded processed data from {DATA_PATH}")
    else:
        data = yf.download("TSLA", start="2015-01-01", end="2026-06-30", progress=False)
        print("Downloaded TSLA data from Yahoo Finance")
    return data


def main():
    tsla_data = load_data()
    price_column = "Adj Close" if "Adj Close" in tsla_data.columns else "Close"

    split_date = pd.Timestamp("2024-12-31")
    train_data = tsla_data[tsla_data.index <= split_date]
    test_data = tsla_data[tsla_data.index > split_date]
    train_prices = train_data[[price_column]].copy()
    test_prices = test_data[[price_column]].copy()

    train_series = train_prices.iloc[:, 0].values.flatten()

    best_order = (1, 1, 1)
    best_seasonal_order = None
    model_type = "ARIMA"
    model = ARIMA(train_series, order=best_order)
    fitted_model = model.fit()

    forecast_horizon = 252
    test_steps = len(test_prices)

    test_prediction = fitted_model.get_forecast(steps=test_steps)
    test_forecast = test_prediction.predicted_mean
    test_conf_int = test_prediction.conf_int(alpha=0.05)

    future_prediction = fitted_model.get_forecast(steps=forecast_horizon)
    future_forecast = future_prediction.predicted_mean
    future_conf_int = future_prediction.conf_int(alpha=0.05)

    test_results = pd.DataFrame(
        {
            "Date": test_prices.index,
            "Actual": test_prices.iloc[:, 0].values,
            "Predicted": test_forecast,
            "Lower_CI": test_conf_int[:, 0],
            "Upper_CI": test_conf_int[:, 1],
        }
    )

    future_dates = pd.date_range(
        tsla_data.index[-1] + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq="B",
    )
    forecast_results = pd.DataFrame(
        {
            "Date": future_dates,
            "Forecast": future_forecast,
            "Lower_CI": future_conf_int[:, 0],
            "Upper_CI": future_conf_int[:, 1],
        }
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        tsla_data.index,
        tsla_data[price_column],
        color="black",
        linewidth=2,
        label="Historical Price",
    )
    ax.plot(
        test_results["Date"],
        test_results["Predicted"],
        color="royalblue",
        linewidth=2,
        label="Test Predictions",
    )
    ax.fill_between(
        test_results["Date"],
        test_results["Lower_CI"],
        test_results["Upper_CI"],
        color="royalblue",
        alpha=0.15,
        label="Test 95% CI",
    )
    ax.plot(
        forecast_results["Date"],
        forecast_results["Forecast"],
        color="darkorange",
        linewidth=2.5,
        label="12-Month Forecast",
    )
    ax.fill_between(
        forecast_results["Date"],
        forecast_results["Lower_CI"],
        forecast_results["Upper_CI"],
        color="darkorange",
        alpha=0.2,
        label="Forecast 95% CI",
    )
    ax.axvline(split_date, color="gray", linestyle="--", linewidth=1.5, label="Training/Test Split")
    ax.set_title("Tesla TSLA: Historical Prices, Test Forecasts, and 12-Month Outlook", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plot_path = OUTPUT_PATH / "forecast_visualization.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Trend and uncertainty analysis
    start_price = forecast_results["Forecast"].iloc[0]
    end_price = forecast_results["Forecast"].iloc[-1]
    total_return_pct = (end_price / start_price - 1) * 100
    avg_daily_growth = (end_price / start_price) ** (1 / len(forecast_results)) - 1

    forecast_widths = forecast_results["Upper_CI"] - forecast_results["Lower_CI"]
    width_start = forecast_widths.iloc[0]
    width_end = forecast_widths.iloc[-1]
    width_growth_pct = (width_end / width_start - 1) * 100

    horizon_points = [30, 90, 180, 252]
    horizon_summary = []
    for point in horizon_points:
        width = forecast_widths.iloc[point - 1]
        horizon_summary.append(f"- {point} days: ${width:,.2f} interval width")

    trend_label = "upward" if end_price > start_price else "downward" if end_price < start_price else "stable"
    if trend_label == "upward":
        opportunity_note = "The model suggests a constructive upside path, which may support bullish positioning or accumulation strategies if the trend is confirmed."
    elif trend_label == "downward":
        opportunity_note = "The model points to a weaker path, so downside protection and tighter risk limits may be more appropriate."
    else:
        opportunity_note = "The model is broadly flat, which may favor range-bound strategies and disciplined risk management."

    risk_note = (
        "The 95% confidence bands widen materially over the horizon, so the long-run forecast is much less certain than the near-term path."
    )

    summary = f"""# Tesla TSLA Forecast Analysis

Using the {model_type} model trained on the 2015-2024 training period, the next 12 months were projected from the latest available close price. The forecast begins at ${start_price:,.2f} and ends at ${end_price:,.2f}, implying a {total_return_pct:.1f}% change across the horizon. The model indicates a generally {trend_label} trend, with an average daily growth rate of {avg_daily_growth * 100:.2f}%.

The confidence intervals widen as the horizon extends. The interval width grows by about {width_growth_pct:.1f}% from the first day to the last day of the forecast, which means near-term estimates are more reliable than long-term estimates. The confidence bounds at key horizons are:
{chr(10).join(horizon_summary)}

Market opportunities: the forecast suggests potential upside if Tesla continues to trend higher, especially for investors willing to tolerate short-term volatility. Risks: the confidence interval widens considerably, increasing the chance that unforeseen events or regime shifts will move prices outside the projected range. The wider bands imply that the model should be treated as a directional guide rather than a precise price target.

{opportunity_note}
{risk_note}
"""

    summary_path = OUTPUT_PATH / "forecast_analysis_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Saved forecast plot to {plot_path}")
    print(f"Saved forecast summary to {summary_path}")
    print("\nForecast Analysis Summary:\n")
    print(summary)


if __name__ == "__main__":
    main()
