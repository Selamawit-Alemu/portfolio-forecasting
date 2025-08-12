# Findings Summary

## Overview

This summary highlights the key findings from the financial time series analysis, forecasting, portfolio optimization, and backtesting conducted on three assets: TSLA, BND, and SPY. The analysis was guided by Modern Portfolio Theory (MPT) principles and aimed to build a robust, optimized portfolio grounded in both forecasted and historical data.

---

## 1. Exploratory Data Analysis (EDA)

* **Volatility:**

  * TSLA exhibited the highest volatility with large price swings, reflecting its growth stock nature.
  * BND showed very low volatility consistent with a bond fund’s stability.
  * SPY demonstrated moderate volatility with steady growth aligned with the broader market.

* **Trends and Stationarity:**

  * Augmented Dickey-Fuller tests confirmed that returns are stationary, suitable for modeling.
  * Seasonality and trend components were identified and accounted for in the forecasting models.

* **Risk Metrics:**

  * Value at Risk (95%) was highest for TSLA, indicating greater downside risk.
  * Sharpe Ratios highlighted favorable risk-adjusted returns for TSLA and SPY compared to BND.

---

## 2. Forecasting Models

* Developed and compared multiple models including ARIMA and LSTM for TSLA’s price prediction.
* The best-performing model forecasted an expected annual return of approximately 42.7% for TSLA, providing a forward-looking estimate to use in portfolio construction.

---

## 3. Portfolio Optimization

* **Inputs:**

  * Forecasted return for TSLA from the best model.
  * Historical annualized returns for BND and SPY.
  * Covariance matrix based on historical returns of all three assets.

* **Optimized Portfolios:**

  * **Maximum Sharpe Ratio Portfolio:** Emphasizes higher risk-adjusted returns, allocating \~75.6% to TSLA and \~24.4% to SPY, with an expected annual return of 19.38%, volatility of 5.92%, and Sharpe Ratio of 2.94.
  * **Minimum Volatility Portfolio:** Focuses on risk minimization, allocating majority to BND (\~24.76%) and SPY (\~72.47%) with minimal TSLA (\~2.77%), yielding an expected return of 2.35% and volatility of 1.02%.

* **Recommendation:**

  * The choice between portfolios depends on investor risk tolerance—high-growth seekers should prefer the Max Sharpe portfolio, while conservative investors should opt for the Min Volatility portfolio.

---

## 4. Backtesting Results

* **Period:** August 1, 2024 – July 31, 2025.
* **Benchmark:** Static 60% SPY / 40% BND portfolio.
* **Findings:**

  * The optimized strategy marginally outperformed the benchmark with a total return of 0.27% vs. -0.57%.
  * Sharpe Ratio for the strategy was 0.08, compared to -0.40 for the benchmark.
* **Conclusion:**

  * These preliminary backtest results support the viability of the model-driven portfolio but highlight the need for further testing over multiple periods.

---

## 5. Challenges and Mitigations

| Challenge                                            | Mitigation                                                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Missing/inconsistent historical price data           | Cleaned datasets by removing NaNs, forward/backward fills, and aligned indexes.     |
| Extreme volatility events skewing volatility metrics | Used Z-score filtering and thresholds to detect outliers without distorting trends. |
| Different asset price scales complicating visuals    | Normalized prices to a 0–1 scale for consistent comparison.                         |
| Visual clutter in market phase annotations           | Applied semi-transparent shaded regions with concise labels for clarity.            |

---

## Final Notes

This project successfully integrated data-driven forecasting with portfolio theory principles to produce actionable investment strategies. While the backtest shows promising initial results, continued refinement, extended testing, and incorporation of transaction costs and rebalancing strategies are recommended for real-world deployment.

