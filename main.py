from src.data_collection import (
    load_stock_data
)

from src.feature_engineering import (
    feature_engineering_pipeline
)

from src.statistical_validation import (
    run_statistical_validation
)

from src.preprocessing import (
    preprocess_pipeline
)

from src.train_ml_models import (
    train_ml_models_pipeline
)

from src.train_dl_models import (
    train_dl_models_pipeline
)

from src.evaluate_models import (
    combine_model_results,
    save_results
)

from src.visualization import (
    plot_model_comparison
)


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # ---------------------------------------------
    # Load Data
    # ---------------------------------------------

    df = load_stock_data()

    # ---------------------------------------------
    # Feature Engineering
    # ---------------------------------------------

    engineered_df = (
        feature_engineering_pipeline(df)
    )

    # ---------------------------------------------
    # Statistical Validation
    # ---------------------------------------------

    run_statistical_validation(
        engineered_df
    )

    # ---------------------------------------------
    # Preprocessing
    # ---------------------------------------------

    X_train, X_test, y_train, y_test, scaler = (
        preprocess_pipeline(df)
    )

    # ---------------------------------------------
    # ML Models
    # ---------------------------------------------

    ml_results = train_ml_models_pipeline(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------------------------------------
    # DL Models
    # ---------------------------------------------

    dl_results = train_dl_models_pipeline(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------------------------------------
    # Combine Results
    # ---------------------------------------------

    combined_results = combine_model_results(
        ml_results,
        dl_results
    )

    save_results(combined_results)

    print("\nFINAL MODEL RESULTS:\n")

    print(combined_results)

    # ---------------------------------------------
    # Visualization
    # ---------------------------------------------

    plot_model_comparison(
        combined_results
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()