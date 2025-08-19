# portfolio-forecasting

# **Financial Time Series Analysis – TSLA, BND, SPY**

## **Overview**

Your previous README is a solid start. To refine it and make it fully reflect the completed work and professional standards required by your rubric, we need to update it to highlight your key deliverable: the interactive Streamlit dashboard.

Here is a revised version of your README.md. It's more concise, clearly highlights the project's value, and provides all the necessary information for a professional submission.

portfolio-forecasting

Financial Time Series Analysis & Visualization

Overview

This project delivers a comprehensive financial time series analysis and a professional, user-friendly dashboard. It analyzes the historical performance, volatility, and risk metrics of three financial assets:

* **TSLA** – High-return, high-volatility growth stock
* **BND** – Low-risk, stable bond fund
* **SPY** – Moderate-risk, diversified S\&P 500 ETF
The core deliverable is an interactive Streamlit dashboard that translates complex financial analysis into clear, actionable insights for a non-technical audience.

The analysis includes:

* Data loading, cleaning, and preprocessing
* Exploratory Data Analysis (EDA)
* Seasonality and trend testing (ADF test)
* Volatility measurement
* Risk metrics (Value at Risk, Sharpe Ratio)
* Visualizations and key insights


The repository is structured to ensure **professional organization** and **full reproducibility**.

---

## **Repository Structure**

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
   test/
   ├── Unit Tests for data_cleaning.py              
   └── Unit Tests for portfolio_optimization.py
   .gitignore
   docker-compose.yml
   Dockerfile
   app.py                  #streamlit dashboard
   README.md
   requirements.txt
```

---

## **Setup Instructions**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Selamawit-Alemu/portfolio-forecasting.git
cd portfolio-forecasting
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```
### 4 For Dashboard Access

```bash
streamlit run app.py 
```
---

## **Data Workflow**

1. **Raw Data**
   Run the scripts in `src/data_loader.py` to:

    * Fetches historical financial data for a predefined list of tickers (e.g., TSLA, BND, SPY) using the yfinance library.
    * Saves the retrieved data for each ticker into a separate CSV file.
    * Manages file directories by automatically creating the data/raw/ folder if it doesn't exist.
   Stored in `data/raw/`, containing unmodified historical prices.

2. **Cleaning & Preprocessing**
   Run the cleaning scripts in `src/data_cleaning.py` to:

   * Handle missing values
   * Ensure correct data types
   * Save cleaned data to `data/processed/`

3. **Exploratory Analysis**
   Open `notebooks/01_eda.ipynb` to:

   * Plot price trends and returns
   * Analyze volatility and trends
   * Perform statistical tests (ADF)

3. **Unit test**
   Open `test` to:

   * Get unit test scripts
   
### How to use in Docker:

1. **Build the Docker image** (run this in your project root, where the Dockerfile is):

```bash
docker build -t portfolio-forecasting .
```

2. **Run the container** and map port 8888 so you can access Jupyter Notebook from your browser:

```bash
docker run -p 8888:8888 portfolio-forecasting
```

3. Open your browser to `http://localhost:8888` to start working inside the container.

## **Running the Analysis**

### Using Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Using Scripts

```bash
python src/data_loader.py # to load BND, SPY and TSLA data
python src/data_cleaning.py
```

---

## **Version Control & Reproducibility**

* All code changes are tracked via Git commits.
* GitHub Actions CI (`.github/workflows/ci.yml`) ensures scripts execute without errors.
* Cleaned datasets are versioned in `data/processed/` for reproducibility.
* Raw data is preserved in `data/raw/` to ensure original sources remain unchanged.


## **Results & Insights**

* **BND**: Very low volatility; stable performance.
* **SPY**: Moderate volatility with consistent growth.
* **TSLA**: Extremely high volatility; high growth potential but large drawdowns.
* Risk metrics calculated:

  * **Value at Risk (95%)** for downside risk
  * **Sharpe Ratio (annualized)** for risk-adjusted returns

Full details in `reports/`.

