# src/portfolio_optimization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns

def load_data():
    bnd = pd.read_csv('../data/raw/BND.csv', parse_dates=['Date'])
    spy = pd.read_csv('../data/raw/SPY.csv', parse_dates=['Date'])

    # Drop rows where 'Date' is missing
    bnd = bnd.dropna(subset=['Date'])
    spy = spy.dropna(subset=['Date'])

    # Set 'Date' as index
    bnd.set_index('Date', inplace=True)
    spy.set_index('Date', inplace=True)

    # Convert 'Close' to numeric, coercing errors to NaN
    bnd_close = pd.to_numeric(bnd['Close'], errors='coerce')
    spy_close = pd.to_numeric(spy['Close'], errors='coerce')

    # Drop NaNs in Close columns
    bnd_close = bnd_close.dropna()
    spy_close = spy_close.dropna()

    return bnd_close, spy_close

def compute_expected_returns(tsla_forecasted_return, bnd_prices, spy_prices):
    # Annualize historical returns for BND and SPY
    bnd_returns = bnd_prices.pct_change().dropna()
    spy_returns = spy_prices.pct_change().dropna()

    bnd_annual_return = bnd_returns.mean() * 252
    spy_annual_return = spy_returns.mean() * 252

    # Combine expected returns vector
    # TSLA forecast return is already annualized or you may need to annualize it
    mu = pd.Series({
        'TSLA': tsla_forecasted_return,
        'BND': bnd_annual_return,
        'SPY': spy_annual_return
    })

    return mu

def compute_covariance_matrix(tsla_returns, bnd_returns, spy_returns):
    # Combine returns into single DataFrame aligned by date
    df_returns = pd.concat([tsla_returns, bnd_returns, spy_returns], axis=1)
    df_returns.columns = ['TSLA', 'BND', 'SPY']
    df_returns.dropna(inplace=True)

    # Compute sample covariance matrix (annualized)
    cov_matrix = df_returns.cov() * 252

    return cov_matrix

def optimize_portfolio(mu, cov_matrix):
    ef = EfficientFrontier(mu, cov_matrix)
    weights_max_sharpe = ef.max_sharpe()

    ef_min_vol = EfficientFrontier(mu, cov_matrix)
    weights_min_vol = ef_min_vol.min_volatility()
    
    # Ensure both variables are returned
    return weights_max_sharpe, weights_min_vol

def plot_efficient_frontier(mu, cov_matrix, weights_max_sharpe, weights_min_vol):

    try:
        plt.style.use("seaborn-v0_8-deep")
    except OSError:
        plt.style.use("seaborn-darkgrid")
    
    from pypfopt import plotting
    ef = EfficientFrontier(mu, cov_matrix)
    fig, ax = plt.subplots(figsize=(10, 7))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    plt.show()


def summarize_portfolio(weights, mu, cov_matrix):
    ef = EfficientFrontier(mu, cov_matrix)
    ef.set_weights(weights)
    ret, vol, sharpe = ef.portfolio_performance()

    print("Portfolio Weights:")
    for asset, weight in weights.items():
        print(f"  {asset}: {weight:.2%}")

    print(f"Expected annual return: {ret:.2%}")
    print(f"Annual volatility (risk): {vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
