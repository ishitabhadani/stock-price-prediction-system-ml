import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils import logger


# =========================================================
# STYLE
# =========================================================

sns.set_style("darkgrid")


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

def plot_actual_vs_predicted(
    actual,
    predicted,
    model_name
):

    logger.info(
        f"Generating prediction plot for {model_name}"
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        actual,
        label="Actual Prices"
    )

    plt.plot(
        predicted,
        label="Predicted Prices"
    )

    plt.title(
        f"{model_name} Predictions"
    )

    plt.xlabel("Time")

    plt.ylabel("Normalized Price")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"reports/figures/{model_name.lower().replace(' ', '_')}_predictions.png"
    )

    plt.close()

    logger.info(
        f"Saved prediction plot for {model_name}"
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

def plot_model_comparison(
    results_df
):

    logger.info(
        "Generating model comparison plot..."
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="RMSE"
    )

    plt.title(
        "Model RMSE Comparison"
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        "reports/figures/model_comparison.png"
    )

    plt.close()

    logger.info(
        "Saved model comparison plot."
    )


# =========================================================
# RESIDUAL DISTRIBUTION
# =========================================================

def plot_residuals(
    actual,
    predicted,
    model_name
):

    residuals = actual.flatten() - predicted.flatten()

    plt.figure(figsize=(10, 6))

    sns.histplot(
        residuals,
        bins=30,
        kde=True
    )

    plt.title(
        f"{model_name} Residual Distribution"
    )

    plt.xlabel("Residual Error")

    plt.tight_layout()

    plt.savefig(
        f"reports/figures/{model_name.lower().replace(' ', '_')}_residuals.png"
    )

    plt.close()

    logger.info(
        f"Saved residual plot for {model_name}"
    )


# =========================================================
# TRAINING HISTORY
# =========================================================

def plot_training_history(
    history,
    model_name
):

    logger.info(
        f"Generating training history for {model_name}"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    if "val_loss" in history.history:

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

    plt.title(
        f"{model_name} Training History"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"reports/figures/{model_name.lower().replace(' ', '_')}_history.png"
    )

    plt.close()

    logger.info(
        f"Saved history plot for {model_name}"
    )