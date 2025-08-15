# src/portfolio_optimization.py

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from typing import Dict, Tuple

def compute_portfolio_metrics(
    tsla_prices: pd.Series,
    bnd_prices: pd.Series,
    spy_prices: pd.Series,
    tsla_forecasted_return: float
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Computes the expected returns vector and covariance matrix for the portfolio.

    This function calculates the historical annualized returns for BND and SPY,
    combines them with the forecasted return for TSLA, and computes the
    annualized covariance matrix.

    Args:
        tsla_prices (pd.Series): Time series of historical TSLA prices.
        bnd_prices (pd.Series): Time series of historical BND prices.
        spy_prices (pd.Series): Time series of historical SPY prices.
        tsla_forecasted_return (float): The forecasted annualized return for TSLA.

    Returns:
        Tuple[pd.Series, pd.DataFrame]: A tuple containing the expected returns
                                        vector (mu) and the covariance matrix.
    """
    # Align all price series by date to ensure they are consistent
    all_prices = pd.concat([tsla_prices, bnd_prices, spy_prices], axis=1, join='inner')
    all_prices.columns = ['TSLA', 'BND', 'SPY']
    
    # Calculate historical returns for all assets
    returns = expected_returns.returns_from_prices(all_prices)
    
    # Calculate expected returns (using forecasted TSLA return)
    mu_bnd = expected_returns.mean_historical_return(all_prices['BND'])
    mu_spy = expected_returns.mean_historical_return(all_prices['SPY'])
    
    # Create the expected returns vector (mu)
    mu = pd.Series({
        'TSLA': tsla_forecasted_return,
        'BND': mu_bnd,
        'SPY': mu_spy
    })
    
    # Compute the annualized sample covariance matrix
    cov_matrix = risk_models.sample_cov(all_prices, frequency=252)

    return mu, cov_matrix

def optimize_portfolio(
    mu: pd.Series,
    cov_matrix: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Optimizes the portfolio to find the Maximum Sharpe and Minimum Volatility weights.

    Args:
        mu (pd.Series): Expected returns vector.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: A tuple containing the
        weights for the Maximum Sharpe portfolio and the Minimum Volatility portfolio.
    """
    ef = EfficientFrontier(mu, cov_matrix, gamma=0)
    weights_max_sharpe = ef.max_sharpe()
    cleaned_weights_max_sharpe = ef.clean_weights()

    ef_min_vol = EfficientFrontier(mu, cov_matrix, gamma=0)
    weights_min_vol = ef_min_vol.min_volatility()
    cleaned_weights_min_vol = ef_min_vol.clean_weights()
    
    return cleaned_weights_max_sharpe, cleaned_weights_min_vol

def get_portfolio_performance(
    weights: Dict[str, float],
    mu: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculates the expected annual return, volatility, and Sharpe Ratio for a portfolio.

    Args:
        weights (Dict[str, float]): The portfolio weights.
        mu (pd.Series): The expected returns vector.
        cov_matrix (pd.DataFrame): The covariance matrix.
        risk_free_rate (float): The risk-free rate for Sharpe Ratio calculation.

    Returns:
        Dict[str, float]: A dictionary of performance metrics.
    """
    ret, vol, sharpe = EfficientFrontier(mu, cov_matrix, risk_free_rate=risk_free_rate).portfolio_performance(verbose=False)
    
    return {
        'expected_return': ret,
        'annual_volatility': vol,
        'sharpe_ratio': sharpe
    }

if __name__ == "__main__":
    # Example usage with dummy data to show the workflow
    print("--- Portfolio Optimization Workflow Example ---")

    # Dummy data setup: Assumes data is already loaded and cleaned
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    tsla_prices = pd.Series(100 + np.cumsum(np.random.randn(1000)), index=dates)
    bnd_prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.1), index=dates)
    spy_prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.5), index=dates)
    
    # Assume TSLA's forecasted return is available from your model
    tsla_forecast = 0.25 # 25% annualized return

    # 1. Compute portfolio metrics
    print("\nComputing expected returns and covariance matrix...")
    mu_computed, cov_matrix_computed = compute_portfolio_metrics(
        tsla_prices, bnd_prices, spy_prices, tsla_forecast
    )
    print("Expected returns (mu):")
    print(mu_computed)
    print("\nCovariance Matrix:")
    print(cov_matrix_computed)

    # 2. Optimize the portfolio
    print("\nOptimizing portfolio for Max Sharpe and Min Volatility...")
    weights_max_sharpe, weights_min_vol = optimize_portfolio(mu_computed, cov_matrix_computed)

    print("\n--- Maximum Sharpe Ratio Portfolio ---")
    print("Optimal Weights:", weights_max_sharpe)
    sharpe_performance = get_portfolio_performance(weights_max_sharpe, mu_computed, cov_matrix_computed)
    print("Performance:", sharpe_performance)
    
    print("\n--- Minimum Volatility Portfolio ---")
    print("Optimal Weights:", weights_min_vol)
    min_vol_performance = get_portfolio_performance(weights_min_vol, mu_computed, cov_matrix_computed)
    print("Performance:", min_vol_performance)

