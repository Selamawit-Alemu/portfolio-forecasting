# portfolio-forecasting

## Financial Time Series Analysis & Portfolio Optimization – TSLA, BND, SPY

## Overview

This project performs a comprehensive financial time series analysis, forecasting, and portfolio optimization for three assets:

* **TSLA** – High-growth, high-volatility stock
* **BND** – Low-risk, stable bond ETF
* **SPY** – Moderate-risk, diversified S\&P 500 ETF

The analysis includes:

* Data loading, cleaning, and preprocessing
* Exploratory Data Analysis (EDA) including volatility and risk metrics
* Time series forecasting using ARIMA and LSTM models
* Portfolio optimization based on Modern Portfolio Theory (MPT)
* Strategy backtesting against benchmark portfolios
* Visualization and detailed reporting

This repository is organized for **full reproducibility** and **scalable experimentation**.

---

## Repository Structure

```plaintext
.github/
└── workflows/                # CI workflow files
data/
├── processed/                # Cleaned datasets
└── raw/                      # Raw downloaded datasets
models/
├── arima_model.pkl           # Serialized ARIMA model
├── lstm_model.h5             # LSTM model weights
└── lstm_model.pkl            # Serialized LSTM model
notebooks/
├── 01_eda.ipynb              # Exploratory Data Analysis
├── 02_forecasting_models.ipynb  # Forecasting using ARIMA & LSTM
└── 03_Portfolio_Optimization_and_Backtesting.ipynb  # Portfolio tasks
reports/
├── figures/                  # Generated figures and plots
├── EDA_RESULTS_AND_ANALYSIS.md  # Summary of EDA findings
└── normalized_price_analysis.txt  # Price normalization analysis
src/
├── arima_model.py            # ARIMA forecasting logic
├── data_cleaning.py          # Data cleaning pipeline
├── data_loader.py            # Data fetching utilities
├── data_prep.py              # Data preprocessing helpers
├── lstm_model.py             # LSTM forecasting model implementation
└── portfolio_optimization.py # Portfolio optimization and backtesting logic
.venv/                        # Local Python virtual environment
.gitignore
docker-compose.yml
Dockerfile
README.md
requirements.txt
```

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/Selamawit-Alemu/portfolio-forecasting.git
cd portfolio-forecasting
```

2. Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Workflow

* **Raw Data:** Use `src/data_loader.py` to fetch historical daily prices for TSLA, BND, and SPY, saved under `data/raw/`.
* **Data Cleaning:** Run `src/data_cleaning.py` to clean, fill missing values, and save processed data under `data/processed/`.
* **Data Preparation:** Use `src/data_prep.py` for feature engineering and preprocessing required for modeling.

---

## Modeling & Analysis

* **Exploratory Data Analysis (EDA):** Conducted in `notebooks/01_eda.ipynb` with visualizations and volatility metrics.
* **Forecasting Models:** Implemented ARIMA and LSTM forecasting in `notebooks/02_forecasting_models.ipynb`.
* **Portfolio Optimization and Backtesting:** Performed using Modern Portfolio Theory in `notebooks/03_Portfolio_Optimization_and_Backtesting.ipynb` combining forecasted returns with historical data.

---

## Running the Notebooks

To explore and reproduce the analysis, launch the notebooks:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_forecasting_models.ipynb
jupyter notebook notebooks/03_Portfolio_Optimization_and_Backtesting.ipynb
```

---

## Results Summary

* **Forecasting:** LSTM and ARIMA models produced varying accuracy; the best forecast was used for portfolio optimization.
* **Portfolio Optimization:** Combined forecasted and historical returns to generate Efficient Frontier, identifying max Sharpe and minimum volatility portfolios.
* **Backtesting:** Strategy backtest showed initial promise over benchmark portfolio, with caveats on data limitations and model assumptions.

---

## Version Control & Reproducibility

* All code and analysis tracked with Git commits.
* Continuous Integration via GitHub Actions (`.github/workflows/ci.yml`) runs tests on core scripts.
* Raw and processed data are versioned separately to maintain data provenance.

---

