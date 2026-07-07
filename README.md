# Portfolio Optimization Project - GMF Investments

## Overview
This project applies time series forecasting and modern portfolio theory to optimize investment portfolios for GMF Investments. The analysis covers three key assets: Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY).

## Project Structure
```
portfolio-optimization/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── processed/
├── notebooks/
│   ├── __init__.py
│   ├── README.md
│   └── 01_data_preprocessing_eda.ipynb
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
└── scripts/
    └── __init__.py
```

## Assets Overview
| Asset | Ticker | Description | Risk Profile |
|-------|--------|-------------|--------------|
| Tesla | TSLA | High-growth stock in consumer discretionary | High risk, high potential return |
| Vanguard Total Bond Market ETF | BND | U.S. investment-grade bonds | Low risk, stability and income |
| S&P 500 ETF | SPY | Broad market exposure | Moderate risk, diversification |

## Data Period
- **Start Date:** January 1, 2015
- **End Date:** June 30, 2026

## Tasks

### Task 1: Preprocess and Explore the Data
Load, clean, and understand the data to prepare it for modeling. This includes:
- Extracting historical financial data using YFinance
- Data cleaning and handling missing values
- Exploratory Data Analysis (EDA)
- Stationarity testing (ADF test)
- Risk metrics calculation (VaR, Sharpe Ratio)

### Task 2: Statistical Modeling (ARIMA/SARIMA)
Build and evaluate time series forecasting models.

### Task 3: Deep Learning Modeling (LSTM)
Construct and train LSTM neural networks for forecasting.

### Task 4: Efficient Frontier and Portfolio Optimization
Generate and analyze the efficient frontier.

### Task 5: Backtesting and Performance Evaluation
Validate strategies with historical simulation.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd portfolio-optimization
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

## Key Learning Outcomes

### Skills
- API Usage with YFinance
- Data Wrangling with pandas
- Feature Engineering
- Data Scaling and Normalization
- Statistical Modeling (ARIMA/SARIMA)
- Deep Learning (LSTM)
- Model Evaluation (MAE, RMSE, MAPE)
- Efficient Frontier Visualization
- Modern Portfolio Theory Implementation
- Backtesting

### Knowledge
- Asset class characteristics
- Efficient Market Hypothesis implications
- Stationarity and time series concepts
- Efficient Frontier significance
- Backtesting methodology

### Behaviors
- Critical Evaluation of Models
- Problem Framing and Synthesis
- Data-Driven Decision Making

### Communication
- Professional investment memos
- Clear model comparisons and justifications

## Team
- Kerod
- Mahbubah
- Feven

## Key Dates
- Challenge Introduction: Wednesday, July 01, 2026 - 10:30 AM UTC
- Interim Submission: Sunday, July 05, 2026 - 8:00 PM UTC
- Final Submission: Tuesday, July 07, 2026 - 8:00 PM UTC

## Support
- Slack Channel: #all-week9
- Office Hours: Mon–Fri, 08:00–15:00 UTC

## References
- [YFinance Documentation](https://pypi.org/project/yfinance/)
- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [StatsModels ARIMA](https://www.statsmodels.org/stable/tsa.html)
- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)

## License
This project is part of the GMF Investments financial analysis curriculum.
