import yfinance as yf
import pandas as pd

from src.utils import (
    logger,
    save_dataframe
)

# =========================================================
# DOWNLOAD MULTIPLE STOCKS
# =========================================================

def download_stock_data():

    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "JPM",
        "SPY"
    ]

    all_data = []

    for ticker in tickers:

        logger.info(
            f"Downloading data for {ticker}"
        )

        df = yf.download(
            ticker,
            start="2000-01-01",
            end="2025-01-01",
            interval="1d",
            auto_adjust=True
        )

        # -------------------------------------------------
        # FIX MULTI-INDEX COLUMNS
        # -------------------------------------------------

        df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)

        # -------------------------------------------------
        # KEEP ONLY REQUIRED COLUMNS
        # -------------------------------------------------

        df = df[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ]

        df["Ticker"] = ticker

        all_data.append(df)

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    final_df.sort_values(
        by="Date",
        inplace=True
    )

    save_dataframe(
        final_df,
        "data/stock_data.csv"
    )

    logger.info(
        f"Dataset shape: {final_df.shape}"
    )

    return final_df


# =========================================================
# LOAD DATASET
# =========================================================

def load_stock_data():

    logger.info(
        "Loading stock dataset..."
    )

    df = pd.read_csv(
        "data/stock_data.csv",
        low_memory=False
    )

    logger.info(
        f"Dataset loaded — shape: {df.shape}"
    )

    return df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    download_stock_data()