import os
import logging
import random
import numpy as np
import tensorflow as tf


# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

DIRECTORIES = [
    "data",
    "models",
    "reports",
    "reports/figures",
    "screenshots"
]

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# RANDOM SEED FOR REPRODUCIBILITY
# =========================================================

def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    logger.info(f"Random seed set to {seed}")


# =========================================================
# SAVE FILE HELPER
# =========================================================

def save_dataframe(df, path: str):
    """
    Save dataframe as CSV.
    """

    df.to_csv(path, index=False)

    logger.info(f"Saved dataframe to '{path}'")


# =========================================================
# PRETTY SECTION PRINTER
# =========================================================

def print_section(title: str):
    """
    Print formatted section title.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)