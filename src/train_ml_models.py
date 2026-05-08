import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    GridSearchCV
)

from src.utils import logger


# =========================================================
# FLATTEN SEQUENCES FOR ML MODELS
# =========================================================

def flatten_sequences(X):

    return X.reshape(X.shape[0], -1)


# =========================================================
# LINEAR REGRESSION
# =========================================================

def train_linear_regression(
    X_train,
    y_train
):

    logger.info(
        "Training Linear Regression..."
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    logger.info(
        "Linear Regression training complete."
    )

    return model


# =========================================================
# RANDOM FOREST
# =========================================================

def train_random_forest(
    X_train,
    y_train
):

    logger.info(
        "Training Random Forest..."
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train.ravel())

    logger.info(
        "Random Forest training complete."
    )

    return model


# =========================================================
# RANDOM FOREST TUNING
# =========================================================

def tune_random_forest(
    X_train,
    y_train
):

    logger.info(
        "Starting Random Forest tuning..."
    )

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }

    model = RandomForestRegressor(
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train.ravel()
    )

    logger.info(
        f"Best Parameters: "
        f"{grid_search.best_params_}"
    )

    return grid_search.best_estimator_


# =========================================================
# EVALUATION
# =========================================================

def evaluate_regression_model(
    model,
    X_test,
    y_test,
    model_name="Model"
):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    logger.info(
        f"[{model_name}] "
        f"MAE={mae:.6f} | "
        f"RMSE={rmse:.6f} | "
        f"R2={r2:.6f}"
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(
    model,
    filename
):

    path = f"models/{filename}"

    joblib.dump(model, path)

    logger.info(
        f"Saved model to '{path}'"
    )


# =========================================================
# FULL ML PIPELINE
# =========================================================

def train_ml_models_pipeline(
    X_train,
    X_test,
    y_train,
    y_test
):

    logger.info("=" * 60)
    logger.info("Starting ML model training")
    logger.info("=" * 60)

    X_train_flat = flatten_sequences(
        X_train
    )

    X_test_flat = flatten_sequences(
        X_test
    )

    results = []

    # ---------------------------------------------
    # Linear Regression
    # ---------------------------------------------

    lr_model = train_linear_regression(
        X_train_flat,
        y_train
    )

    lr_metrics = evaluate_regression_model(
        lr_model,
        X_test_flat,
        y_test,
        "Linear Regression"
    )

    save_model(
        lr_model,
        "linear_regression.pkl"
    )

    results.append(lr_metrics)

    # ---------------------------------------------
    # Random Forest
    # ---------------------------------------------

    rf_model = tune_random_forest(
        X_train_flat,
        y_train
    )

    rf_metrics = evaluate_regression_model(
        rf_model,
        X_test_flat,
        y_test,
        "Random Forest"
    )

    save_model(
        rf_model,
        "random_forest.pkl"
    )

    results.append(rf_metrics)

    results_df = pd.DataFrame(results)

    logger.info(
        "ML model pipeline complete."
    )

    return results_df