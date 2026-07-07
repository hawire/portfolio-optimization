# Strategy Backtest Summary

The backtest uses the last available period of the dataset from 2025-01-01 through 2026-06-30, with the Task 4 portfolio allocation rebalanced monthly against a static 60% SPY / 40% BND benchmark.

## Performance Metrics

| Portfolio | Total Return | Annualized Return | Sharpe Ratio | Max Drawdown |
| --- | ---: | ---: | ---: | ---: |
| Strategy | 17.53% | 11.60% | 1.288 | 8.43% |
| Benchmark | 20.56% | 13.54% | 1.197 | 11.29% |

## Conclusion

The strategy did not beat the benchmark over this backtest window, which suggests the Task 4 allocation was not strong enough to justify the model-driven approach on its own. This does not invalidate the earlier forecast, but it does indicate that the simple balanced benchmark was more robust in this historical period and that further refinement is needed.

This backtest is useful as a first-pass validation, but it has limitations. The window is relatively short, it ignores transaction costs and taxes, and it relies on historical returns that may not reflect future market conditions. It also uses a single forecast-implied allocation rather than a more dynamic regime-aware process.