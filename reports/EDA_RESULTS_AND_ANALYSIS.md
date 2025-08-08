# Results and Analysis

## 1. Price Trends and Patterns

* **TSLA (Tesla Inc.)**
  TSLA showed remarkable growth from 2015 to 2025, starting with stable low prices under \$20 until late 2019, followed by an explosive surge peaking over \$400 in early 2021. After a volatile correction phase, it rebounded strongly, nearing \$500 by mid-2025. The stock exhibits extreme price fluctuations, highlighting its high-risk, high-reward nature.

* **BND (Vanguard Total Bond Market ETF)**
  BND maintained a stable price range between \$60 and \$80 for most of the period, reflecting its conservative bond fund profile. After 2020, BND experienced a consistent decline, correlating with rising interest rates, before showing signs of stabilization. Its price movements are minimal compared to equity assets.

* **SPY (SPDR S\&P 500 ETF Trust)**
  SPY demonstrated steady, long-term growth from about \$180 to over \$600, consistent with the broad US market’s bullish trend. Minor dips corresponded with market events like the COVID-19 crash in early 2020. The performance indicates a moderate-risk, diversified investment.

## 2. Volatility and Daily Returns

* **TSLA:** Exhibits the highest volatility with frequent large swings in daily returns, including many days with gains or losses exceeding ±10%. Volatility spikes coincide with major market events and company-specific news, underscoring its speculative nature.

* **BND:** Displays very low volatility with daily returns tightly clustered near zero. Outliers are rare and generally coincide with macroeconomic events like early 2020’s market turmoil.

* **SPY:** Shows moderate volatility with daily return fluctuations larger than BND but smaller than TSLA. Significant spikes align with market-wide shocks.

## 3. Stationarity Tests (Augmented Dickey-Fuller)

* All three asset **closing prices** were found to be **non-stationary**, indicating trends and/or seasonality in their price series. This implies the need for differencing before applying ARIMA or other time series models requiring stationarity.

* Conversely, the **daily returns** for TSLA, BND, and SPY were all **stationary**, making returns suitable for volatility modeling and risk metric calculations.

## 4. Rolling Statistics – Short-term Trends and Volatility

* **Rolling Means (30-day):** Smooth the daily price fluctuations to reveal underlying trends. TSLA’s rolling mean reflects its major growth and correction phases, while BND’s rolling mean stays nearly flat, and SPY’s shows steady growth.

* **Rolling Standard Deviations (Volatility):** Quantify short-term fluctuations. TSLA’s volatility is extremely high with large spikes, BND maintains consistently low volatility, and SPY exhibits moderate volatility with occasional spikes during market stress.

## 5. Risk Metrics

| Asset | Value at Risk (VaR) 95% | Sharpe Ratio (Annualized) |
| ----- | ----------------------- | ------------------------- |
| TSLA  | -5.47%                  | 0.78                      |
| BND   | -0.49%                  | 0.36                      |
| SPY   | -1.72%                  | 0.79                      |

* **VaR:** TSLA shows the highest potential daily loss at the 95% confidence level, reflecting its riskiness. BND shows the lowest risk, consistent with bond fund characteristics.

* **Sharpe Ratio:** SPY and TSLA demonstrate favorable risk-adjusted returns. BND’s lower Sharpe ratio reflects its conservative return profile.

---

# Summary

This analysis highlights the differing investment profiles of TSLA, BND, and SPY:

* TSLA is a **high-volatility, high-growth** stock, suitable for investors with higher risk tolerance.
* BND is a **stable, low-risk bond fund**, ideal for risk-averse investors seeking capital preservation.
* SPY provides a **balanced, moderate-risk market exposure**, reflecting overall market performance.