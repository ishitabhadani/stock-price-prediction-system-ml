import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_collection import (
    load_stock_data
)

from src.feature_engineering import (
    feature_engineering_pipeline
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
    combine_model_results
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stock Price Prediction System",
    layout="wide"
)

st.title("📈 Stock Price Prediction System")

st.markdown("""
This dashboard demonstrates:
- Traditional Machine Learning models
- Deep Learning architectures
- Financial feature engineering
- Time-series forecasting
""")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = load_stock_data()

    return df


df = load_data()


# =========================================================
# SHOW RAW DATA
# =========================================================

st.header("Raw Stock Dataset")

st.dataframe(df.head())


# =========================================================
# STOCK PRICE VISUALIZATION
# =========================================================

st.header("Stock Closing Prices")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df["Close"])

ax.set_title("Closing Price Trend")

ax.set_xlabel("Time")

ax.set_ylabel("Price")

st.pyplot(fig)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

st.header("Feature Engineering")

engineered_df = (
    feature_engineering_pipeline(df)
)

st.write(
    engineered_df.head()
)


# =========================================================
# PREPROCESSING
# =========================================================

X_train, X_test, y_train, y_test, scaler = (
    preprocess_pipeline(df)
)


# =========================================================
# MODEL TRAINING BUTTON
# =========================================================

if st.button("Train Models"):

    st.subheader(
        "Training Machine Learning Models..."
    )

    ml_results = (
        train_ml_models_pipeline(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    st.dataframe(ml_results)

    st.subheader(
        "Training Deep Learning Models..."
    )

    dl_results = (
        train_dl_models_pipeline(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    st.dataframe(dl_results)

    # -----------------------------------------------------
    # COMBINE RESULTS
    # -----------------------------------------------------

    combined_results = combine_model_results(
        ml_results,
        dl_results
    )

    st.subheader("Final Model Comparison")

    st.dataframe(combined_results)

    # -----------------------------------------------------
    # RMSE COMPARISON PLOT
    # -----------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.bar(
        combined_results["Model"],
        combined_results["RMSE"]
    )

    ax2.set_title("RMSE Comparison")

    ax2.set_ylabel("RMSE")

    plt.xticks(rotation=15)

    st.pyplot(fig2)

    st.success(
        "Training Complete!"
    )