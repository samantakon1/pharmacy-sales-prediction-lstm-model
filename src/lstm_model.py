
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorflow.keras import layers, models


def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def mae_rmse(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse