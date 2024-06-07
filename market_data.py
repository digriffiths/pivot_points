import yfinance as yf
import numpy as np
from pandas import DataFrame
from typing import Optional


def get_market_data(ticker: str, start_date: str, end_date: str, interval: str) -> Optional[DataFrame]:
    """
    Download market data for a given ticker and compute logarithmic returns and cumulative returns.

    Args:
    ticker (str): The stock symbol to download data for.
    start_date (str): The start date for the data download (inclusive).
    end_date (str): The end date for the data download (inclusive).
    interval (str): The interval between data points (e.g., '1d' for daily).

    Returns:
    Optional[DataFrame]: A DataFrame containing the logarithmic returns and cumulative returns, or None if the data download fails.
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by="ticker",
        )
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        return data[["returns", "creturns"]]
    except Exception as e:
        print(f"Failed to download or process data: {e}")
        return None
