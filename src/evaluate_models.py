import pandas as pd

from src.utils import logger


# =========================================================
# COMBINE RESULTS
# =========================================================

def combine_model_results(
    ml_results,
    dl_results
):

    logger.info(
        "Combining model evaluation results..."
    )

    combined = pd.concat(
        [ml_results, dl_results],
        ignore_index=True
    )

    combined = combined.sort_values(
        by="RMSE"
    )

    logger.info(
        "Model results combined successfully."
    )

    return combined


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(
    results_df
):

    path = "reports/model_results.csv"

    results_df.to_csv(
        path,
        index=False
    )

    logger.info(
        f"Saved results to '{path}'"
    )