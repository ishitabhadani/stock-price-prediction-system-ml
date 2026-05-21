import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

from src.data_collection import (
    load_stock_data
)

from src.preprocessing import (
    preprocess_pipeline
)

from src.utils import logger


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    layout="wide"
)

st.title("📈 Stock Price Prediction Dashboard")

st.markdown("""
Interactive dashboard for stock forecasting using:
- Machine Learning
- Deep Learning
- Financial Time-Series Analysis
""")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def get_data():

    df = load_stock_data()

    return df


df = get_data()

st.header("Dataset Preview")

st.dataframe(df.head())


# =========================================================
# PREPROCESS DATA
# =========================================================

X_train, X_test, y_train, y_test, scaler = (
    preprocess_pipeline(df)
)


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_trained_models():

    models = {}

    try:

        models["Linear Regression"] = joblib.load(
            "models/linear_regression.pkl"
        )

    except:

        logger.warning(
            "Linear Regression model not found."
        )

    try:

        models["Random Forest"] = joblib.load(
            "models/random_forest.pkl"
        )

    except:

        logger.warning(
            "Random Forest model not found."
        )

    try:

        models["LSTM"] = load_model(
            "models/lstm_model.keras"
        )

    except:

        logger.warning(
            "LSTM model not found."
        )

    try:

        models["CNN"] = load_model(
            "models/cnn_model.keras"
        )

    except:

        logger.warning(
            "CNN model not found."
        )

    # try:

    #     models["Transformer"] = load_model(
    #         "models/transformer_model.keras"
    #     )

    # except:

    #     logger.warning(
    #         "Transformer model not found."
    #     )

    return models


models = load_trained_models()


# =========================================================
# MODEL SELECTION
# =========================================================

st.header("Select Model")

selected_model = st.selectbox(
    "Choose a model",
    list(models.keys())
)

model = models[selected_model]


# =========================================================
# PREDICTIONS
# =========================================================

st.header("Predictions")

# ---------------------------------------------------------
# ML MODELS
# ---------------------------------------------------------

if selected_model in [
    "Linear Regression",
    "Random Forest"
]:

    X_test_ml = X_test.reshape(
        X_test.shape[0],
        -1
    )

    predictions = model.predict(
        X_test_ml
    )

# ---------------------------------------------------------
# DL MODELS
# ---------------------------------------------------------

else:

    predictions = model.predict(
        X_test
    ).flatten()


# =========================================================
# VISUALIZATION
# =========================================================

fig, ax = plt.subplots(
    figsize=(7, 3.5)
)

ax.plot(
    y_test[:300],
    label="Actual Prices"
)

ax.plot(
    predictions[:300],
    label="Predicted Prices"
)

ax.set_title(
    f"{selected_model} Predictions"
)

ax.set_xlabel("Time")

ax.set_ylabel("Normalized Price")

ax.legend()

st.pyplot(
    fig,
    use_container_width=False
)


# =========================================================
# METRICS
# =========================================================
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)
st.header("Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{mae:.4f}"
)

col2.metric(
    "RMSE",
    f"{rmse:.4f}"
)

col3.metric(
    "R² Score",
    f"{r2:.4f}"
)
results_df = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "LSTM",
        "CNN"
    ],

    "R² Score": [
        0.9985,
        0.9984,
        0.9969,
        0.9981
    ],

    "RMSE": [
        0.0112,
        0.0124,
        0.0130,
        0.0101
    ]
})
st.subheader("Model Comparison")

st.dataframe(
    results_df,
    width="stretch"
)
best_model = results_df.loc[
    results_df["R² Score"].idxmax()
]

st.success(
    f"Best Performing Model: "
    f"{best_model['Model']} "
    f"(R² = {best_model['R² Score']})"
)