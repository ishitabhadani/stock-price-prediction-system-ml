
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    Conv1D,
    MaxPooling1D,
    Flatten,
    InputLayer
)

from tensorflow.keras.optimizers.legacy import Adam

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.utils import logger


# =========================================================
# EVALUATION
# =========================================================

def evaluate_dl_model(
    model,
    X_test,
    y_test,
    model_name
):

    predictions = model.predict(X_test)

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
# LSTM MODEL
# =========================================================

def build_lstm_model(input_shape):

    model = Sequential([

        InputLayer(input_shape=input_shape),

        LSTM(
            64,
            return_sequences=True
        ),

        Dropout(0.2),

        LSTM(32),

        Dropout(0.2),

        Dense(1)
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    return model


# =========================================================
# CNN MODEL
# =========================================================

def build_cnn_model(input_shape):

    model = Sequential([

        InputLayer(input_shape=input_shape),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),

        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(50, activation="relu"),

        Dense(1)
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    return model


# =========================================================
# TRANSFORMER MODEL
# =========================================================

# def build_transformer_model(
#     input_shape
# ):

#     inputs = Input(shape=input_shape)

#     # -----------------------------------------
#     # Multi-head attention
#     # -----------------------------------------

#     attention = MultiHeadAttention(
#         num_heads=4,
#         key_dim=32
#     )(inputs, inputs)

#     attention = Dropout(0.1)(
#         attention
#     )

#     attention = LayerNormalization(
#         epsilon=1e-6
#     )(attention + inputs)

#     # -----------------------------------------
#     # Feed Forward
#     # -----------------------------------------

#     ff = Dense(
#         64,
#         activation="relu"
#     )(attention)

#     ff = Dense(
#         input_shape[-1]
#     )(ff)

#     ff = LayerNormalization(
#         epsilon=1e-6
#     )(ff + attention)

#     # -----------------------------------------
#     # Pooling
#     # -----------------------------------------

#     x = GlobalAveragePooling1D()(ff)

#     x = Dense(
#         64,
#         activation="relu"
#     )(x)

#     x = Dropout(0.2)(x)

#     outputs = Dense(1)(x)

#     model = Model(
#         inputs,
#         outputs
#     )

#     model.compile(
#         optimizer=Adam(
#             learning_rate=0.0001
#         ),
#         loss="mse"
#     )

#     return model


# =========================================================
# TRAIN MODEL
# =========================================================

def train_dl_model(
    model,
    X_train,
    y_train,
    epochs=10,
    batch_size=32
):

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    return history


# =========================================================
# SAVE MODEL
# =========================================================

def save_dl_model(
    model,
    filename
):

    path = f"models/{filename}"

    model.save(path)

    logger.info(
        f"Saved DL model to '{path}'"
    )


# =========================================================
# FULL DL PIPELINE
# =========================================================

def train_dl_models_pipeline(
    X_train,
    X_test,
    y_train,
    y_test
):

    logger.info("=" * 60)
    logger.info("Starting deep learning training")
    logger.info("=" * 60)

    results = []

    input_shape = (
        X_train.shape[1],
        X_train.shape[2]
    )

    # -----------------------------------------------------
    # LSTM
    # -----------------------------------------------------

    lstm_model = build_lstm_model(
        input_shape
    )

    train_dl_model(
        lstm_model,
        X_train,
        y_train
    )

    lstm_metrics = evaluate_dl_model(
        lstm_model,
        X_test,
        y_test,
        "LSTM"
    )

    save_dl_model(
        lstm_model,
        "lstm_model.keras"
    )

    results.append(lstm_metrics)

    # -----------------------------------------------------
    # CNN
    # -----------------------------------------------------

    cnn_model = build_cnn_model(
        input_shape
    )

    train_dl_model(
        cnn_model,
        X_train,
        y_train
    )

    cnn_metrics = evaluate_dl_model(
        cnn_model,
        X_test,
        y_test,
        "CNN"
    )

    save_dl_model(
        cnn_model,
        "cnn_model.keras"
    )

    results.append(cnn_metrics)

    # -----------------------------------------------------
    # TRANSFORMER
    # -----------------------------------------------------

    # transformer_model = build_transformer_model(
    #     input_shape
    # )

    # train_dl_model(
    #     transformer_model,
    #     X_train,
    #     y_train
    # )

    # transformer_metrics = evaluate_dl_model(
    #     transformer_model,
    #     X_test,
    #     y_test,
    #     "Transformer"
    # )

    # save_dl_model(
    #     transformer_model,
    #     "transformer_model.keras"
    # )

    # results.append(transformer_metrics)

    results_df = pd.DataFrame(results)

    logger.info(
        "Deep learning pipeline complete."
    )

    return results_df