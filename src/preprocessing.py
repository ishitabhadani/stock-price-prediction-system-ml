import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.utils import logger


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

def clean_data(df):

    logger.info("Cleaning dataset...")

    missing_before = df.isnull().sum().sum()

    logger.info(
        f"Missing values before cleaning: {missing_before}"
    )

    df = df.ffill().bfill()

    missing_after = df.isnull().sum().sum()

    logger.info(
        f"Missing values after cleaning: {missing_after}"
    )

    return df


# =========================================================
# CREATE SEQUENCES PER STOCK
# =========================================================

def create_sequences_per_stock(
    df,
    sequence_length=60
):

    logger.info(
        f"Creating sequences per stock "
        f"(window={sequence_length})"
    )

    X = []
    y = []

    tickers = df["Ticker"].unique()

    for ticker in tickers:

        logger.info(
            f"Processing {ticker}"
        )

        stock_df = df[
            df["Ticker"] == ticker
        ].copy()

        stock_df = stock_df.sort_values(
            by="Date"
        )

        # -------------------------------------
        # USE ONLY CLOSE PRICE
        # -------------------------------------

        close_prices = stock_df[
            ["Close"]
        ].values

        # -------------------------------------
        # SCALE PER STOCK
        # -------------------------------------

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(
            close_prices
        )

        # -------------------------------------
        # CREATE SEQUENCES
        # -------------------------------------

        for i in range(
            sequence_length,
            len(scaled_data)
        ):

            X.append(
                scaled_data[
                    i-sequence_length:i
                ]
            )

            y.append(
                scaled_data[i]
            )

    X = np.array(X)
    y = np.array(y)

    logger.info(
        f"Generated {len(X)} sequences."
    )

    return X, y


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

def train_test_split_sequences(
    X,
    y,
    train_ratio=0.8
):

    split_index = int(
        len(X) * train_ratio
    )

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    logger.info(
        f"Train size: {len(X_train)} | "
        f"Test size: {len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# =========================================================
# FULL PIPELINE
# =========================================================

def preprocess_pipeline(
    df,
    sequence_length=60
):

    logger.info("=" * 60)
    logger.info(
        "Starting preprocessing pipeline"
    )
    logger.info("=" * 60)

    df = clean_data(df)

    X, y = create_sequences_per_stock(
        df,
        sequence_length
    )

    X_train, X_test, y_train, y_test = (
        train_test_split_sequences(X, y)
    )

    logger.info(
        "Preprocessing pipeline complete."
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        None
    )