import numpy as np
import pandas as pd

from src.utils import logger


# =========================================================
# SIMPLE MOVING AVERAGES
# =========================================================

def add_moving_averages(df):

    logger.info("Adding moving averages...")

    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    return df


# =========================================================
# EXPONENTIAL MOVING AVERAGES
# =========================================================

def add_exponential_moving_averages(df):

    logger.info("Adding exponential moving averages...")

    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()

    return df


# =========================================================
# RSI (RELATIVE STRENGTH INDEX)
# =========================================================

def add_rsi(df, window=14):

    logger.info("Adding RSI indicator...")

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# =========================================================
# MACD
# =========================================================

def add_macd(df):

    logger.info("Adding MACD indicator...")

    ema_12 = df["Close"].ewm(span=12).mean()
    ema_26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema_12 - ema_26

    df["MACD_SIGNAL"] = (
        df["MACD"].ewm(span=9).mean()
    )

    return df


# =========================================================
# VOLATILITY
# =========================================================

def add_volatility(df):

    logger.info("Adding volatility features...")

    df["Daily_Return"] = (
        df["Close"].pct_change()
    )

    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    return df


# =========================================================
# BOLLINGER BANDS
# =========================================================

def add_bollinger_bands(df):

    logger.info("Adding Bollinger Bands...")

    sma_20 = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    std_20 = (
        df["Close"]
        .rolling(window=20)
        .std()
    )

    df["BB_UPPER"] = sma_20 + (2 * std_20)
    df["BB_LOWER"] = sma_20 - (2 * std_20)

    return df


# =========================================================
# MOMENTUM
# =========================================================

def add_momentum(df):

    logger.info("Adding momentum indicators...")

    df["Momentum_10"] = (
        df["Close"] - df["Close"].shift(10)
    )

    return df


# =========================================================
# FEATURE ENGINEERING PIPELINE
# =========================================================

def feature_engineering_pipeline(df):

    logger.info("=" * 60)
    logger.info("Starting feature engineering pipeline")
    logger.info("=" * 60)

    df = add_moving_averages(df)

    df = add_exponential_moving_averages(df)

    df = add_rsi(df)

    df = add_macd(df)

    df = add_volatility(df)

    df = add_bollinger_bands(df)

    df = add_momentum(df)

    # Remove NaN rows caused by rolling windows
    df = df.dropna()

    logger.info(
        f"Feature engineering complete — shape: {df.shape}"
    )

    return df