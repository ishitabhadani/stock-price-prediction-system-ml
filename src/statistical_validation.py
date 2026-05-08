import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro

from statsmodels.tsa.stattools import (
    adfuller,
    kpss
)

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

from statsmodels.tsa.seasonal import (
    seasonal_decompose
)

from src.utils import logger


# =========================================================
# SHAPIRO-WILK NORMALITY TEST
# =========================================================

def shapiro_test(df, column="Close"):

    logger.info("Performing Shapiro-Wilk test...")

    stat, p_value = shapiro(df[column])

    logger.info(
        f"Shapiro-Wilk p-value: {p_value:.6f}"
    )

    if p_value > 0.05:
        logger.info("Data appears normally distributed.")
    else:
        logger.info("Data is NOT normally distributed.")

    return p_value


# =========================================================
# ADF TEST
# =========================================================

def adf_test(df, column="Close"):

    logger.info("Performing Augmented Dickey-Fuller test...")

    result = adfuller(df[column])

    logger.info(f"ADF Statistic: {result[0]:.6f}")
    logger.info(f"ADF p-value: {result[1]:.6f}")

    if result[1] < 0.05:
        logger.info("Series is stationary.")
    else:
        logger.info("Series is NOT stationary.")

    return result


# =========================================================
# KPSS TEST
# =========================================================

def kpss_test(df, column="Close"):

    logger.info("Performing KPSS test...")

    result = kpss(
        df[column],
        regression="c",
        nlags="auto"
    )

    logger.info(f"KPSS Statistic: {result[0]:.6f}")
    logger.info(f"KPSS p-value: {result[1]:.6f}")

    if result[1] < 0.05:
        logger.info("Series is NOT stationary.")
    else:
        logger.info("Series is stationary.")

    return result


# =========================================================
# ACF / PACF PLOTS
# =========================================================

def generate_acf_pacf_plots(
    df,
    column="Close"
):

    logger.info("Generating ACF/PACF plots...")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    plot_acf(
        df[column],
        ax=axes[0],
        lags=40
    )

    plot_pacf(
        df[column],
        ax=axes[1],
        lags=40
    )

    axes[0].set_title("ACF Plot")
    axes[1].set_title("PACF Plot")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/acf_pacf.png"
    )

    logger.info(
        "Saved ACF/PACF plot."
    )


# =========================================================
# SEASONAL DECOMPOSITION
# =========================================================

def seasonal_decomposition_analysis(
    df,
    column="Close"
):

    logger.info(
        "Performing seasonal decomposition..."
    )

    decomposition = seasonal_decompose(
        df[column],
        model="additive",
        period=30
    )

    fig = decomposition.plot()

    fig.set_size_inches(12, 8)

    plt.tight_layout()

    plt.savefig(
        "reports/figures/seasonal_decomposition.png"
    )

    logger.info(
        "Saved seasonal decomposition plot."
    )


# =========================================================
# FULL VALIDATION PIPELINE
# =========================================================

def run_statistical_validation(
    df,
    column="Close"
):

    logger.info("=" * 60)
    logger.info("Starting statistical validation")
    logger.info("=" * 60)

    shapiro_test(df, column)

    adf_test(df, column)

    kpss_test(df, column)

    generate_acf_pacf_plots(df, column)

    seasonal_decomposition_analysis(
        df,
        column
    )

    logger.info(
        "Statistical validation complete."
    )