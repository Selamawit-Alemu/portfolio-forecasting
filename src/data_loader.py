import yfinance as yf
import pandas as pd
import os
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s] - %(message)s"
)

# Constants
TICKERS: List[str] = ['TSLA', 'BND', 'SPY']
START_DATE: str = '2015-07-01'
# End date is one day after the last day you want to include, as yfinance is exclusive.
END_DATE: str = '2025-08-01' 
SAVE_DIR: str = 'data/raw/'
EXPECTED_COLUMNS: List[str] = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

def fetch_and_validate_data(
    ticker: str,
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Fetches historical stock data and performs validation checks.

    Args:
        ticker (str): The stock ticker symbol.
        start (str): The start date for data fetching (YYYY-MM-DD).
        end (str): The end date for data fetching (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame with validated historical data, or an empty
                      DataFrame if fetching fails.
    """
    try:
        logging.info(f"Attempting to fetch data for {ticker} from {start} to {end}...")
        # Set auto_adjust=False to get the Adj Close column
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

        if df.empty:
            logging.warning(f"No data found for ticker '{ticker}' in the specified date range.")
            return pd.DataFrame()

        # Data Validation Checks
        if 'Date' not in df.columns:
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
        
        # Check for expected columns
        if not all(col in df.columns for col in EXPECTED_COLUMNS):
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            logging.warning(f"Missing expected columns for '{ticker}': {missing_cols}. Proceeding with available data.")
        
        # Check for date range consistency
        if df.index.min() > pd.to_datetime(start) or df.index.max() < pd.to_datetime(pd.to_datetime(end) - pd.Timedelta(days=1)):
            logging.warning(
                f"Fetched data for '{ticker}' does not cover the full range. "
                f"Available data: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}."
            )

        logging.info(f"Successfully fetched and validated data for {ticker}.")
        return df

    except Exception as e:
        logging.error(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, ticker: str, save_dir: str):
    """Saves a DataFrame to a CSV file."""
    if df.empty:
        logging.warning(f"Cannot save empty DataFrame for '{ticker}'.")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{ticker}.csv")
    df.to_csv(save_path)
    logging.info(f"Saved {ticker} data to {save_path}")

def main():
    """Main function to orchestrate data fetching and saving for all tickers."""
    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        df = fetch_and_validate_data(ticker, START_DATE, END_DATE)
        if not df.empty:
            save_data(df, ticker, SAVE_DIR)
            all_data[ticker] = df
    
    if not all_data:
        logging.error("No data was successfully fetched for any ticker.")
    else:
        logging.info("Data fetching and saving process completed.")
    
    return all_data

if __name__ == "__main__":
    main()