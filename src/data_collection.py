import yfinance as yf
import pandas as pd

from src.utils import logger, save_dataframe


# =========================================================
# DOWNLOAD STOCK DATA
# =========================================================

def download_stock_data(
    ticker: str = "AAPL",
    start_date: str = "2015-01-01",
    end_date: str = "2025-01-01",
    save_path: str = "data/stock_data.csv"
):
    """
    Download historical stock data using Yahoo Finance.
    """

    logger.info(f"Downloading stock data for {ticker}...")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date
    )

    if df.empty:
        raise ValueError("Downloaded dataframe is empty.")

    df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    save_dataframe(df, save_path)

    logger.info(
        f"Downloaded {len(df)} rows for {ticker}"
    )

    return df


# =========================================================
# LOAD SAVED DATA
# =========================================================

def load_stock_data(
    file_path: str = "data/stock_data.csv"
):
    """
    Load stock dataset from CSV.
    """

    logger.info(f"Loading dataset from '{file_path}'")

    df = pd.read_csv(file_path)

    logger.info(
        f"Dataset loaded successfully — shape: {df.shape}"
    )

    return df


# =========================================================
# QUICK TEST
# =========================================================

if __name__ == "__main__":

    stock_df = download_stock_data()

    print(stock_df.head())