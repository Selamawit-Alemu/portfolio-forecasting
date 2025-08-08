import yfinance as yf
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
TICKERS = ['TSLA', 'BND', 'SPY']
START_DATE = '2015-07-01'
END_DATE = '2025-07-31'
SAVE_DIR = 'data/raw/'

def fetch_and_save_data(ticker: str, start: str, end: str, save_dir: str):
    """Fetches and saves historical data for a given ticker."""
    try:
        logging.info(f"Fetching data for {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        
        if df.empty:
            logging.warning(f"No data found for {ticker}.")
            return

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{ticker}.csv")
        df.reset_index().to_csv(save_path, index=False)
        logging.info(f"Saved {ticker} data to {save_path}")
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")

def fetch_all_data():
    """Fetches data for all target tickers and saves them."""
    for ticker in TICKERS:
        fetch_and_save_data(ticker, START_DATE, END_DATE, SAVE_DIR)

if __name__ == "__main__":
    fetch_all_data()
