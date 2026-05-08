import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.utils import logger


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

def clean_data(df: pd.DataFrame):
    """
    Handle missing values.
    """

    logger.info("Cleaning dataset...")

    missing_before = df.isnull().sum().sum()

    logger.info(f"Missing values before cleaning: {missing_before}")

    df = df.ffill().bfill()

    missing_after = df.isnull().sum().sum()

    logger.info(f"Missing values after cleaning: {missing_after}")

    return df


# =========================================================
# NORMALIZE DATA
# =========================================================

def normalize_data(
    data: pd.DataFrame,
    column: str = "Close"
):
    """
    Normalize stock prices using MinMaxScaler.
    """

    logger.info("Normalizing stock prices...")

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(
        data[[column]]
    )

    logger.info("Normalization complete.")

    return scaled_data, scaler


# =========================================================
# CREATE SLIDING WINDOWS
# =========================================================

def create_sequences(
    data,
    sequence_length: int = 60
):
    """
    Create time-series sliding windows.
    """

    logger.info(
        f"Creating sequences with window size {sequence_length}"
    )

    X = []
    y = []

    for i in range(sequence_length, len(data)):

        X.append(
            data[i-sequence_length:i]
        )

        y.append(
            data[i]
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
    train_ratio: float = 0.8
):
    """
    Split sequences into train and test sets.
    """

    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    logger.info(
        f"Train size: {len(X_train)} | "
        f"Test size: {len(X_test)}"
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# FULL PREPROCESSING PIPELINE
# =========================================================

def preprocess_pipeline(
    df,
    target_column: str = "Close",
    sequence_length: int = 60
):
    """
    Complete preprocessing pipeline.
    """

    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    df = clean_data(df)

    scaled_data, scaler = normalize_data(
        df,
        target_column
    )

    X, y = create_sequences(
        scaled_data,
        sequence_length
    )

    X_train, X_test, y_train, y_test = (
        train_test_split_sequences(X, y)
    )

    logger.info("Preprocessing pipeline complete.")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )