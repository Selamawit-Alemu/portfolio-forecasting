# portfolio-forecasting

# **Financial Time Series Analysis – TSLA, BND, SPY**

## **Overview**

This project analyzes the historical performance, volatility, and risk metrics of three financial assets:

* **TSLA** – High-return, high-volatility growth stock
* **BND** – Low-risk, stable bond fund
* **SPY** – Moderate-risk, diversified S\&P 500 ETF

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
    ├── .github/workflows/ci.yml        # GitHub Actions CI workflow
    ├── data/
    │   ├── processed/                  # Cleaned datasets for analysis
    │   │   ├── bnd_clean.csv
    │   │   ├── spy_clean.csv
    │   │   └── tsla_clean.csv
    │   └── raw/                        # Original downloaded datasets
    │       ├── BND.csv
    │       ├── SPY.csv
    │       └── TSLA.csv
    ├── notebooks/
    │   └── 01_eda.ipynb                 # Exploratory data analysis
    ├── reports/
    │   ├──figures/                       #figures of plots
    │   ├──EDA_RESULTS_AND_ANALYSIS.MD    #Text-based summary of findings
    │   └── normalized_price_analysis.txt # Text-based summary of normalized price analysis
    ├── src/
    │   ├── data_cleaning.py             # Data cleaning pipeline
    │   └── data_loader.py               # Data loading utilities
    ├── .gitignore
    ├── requirements.txt                 # Python dependencies
    ├── app.py                           # dashboard for non- technical users
    ├── README.md                        # Project documentation
    └── .venv/                           # Virtual environment (local use only)
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

