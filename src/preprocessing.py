from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def enforce_continuous_index(df, freq: str):
    df = df.copy().set_index("ds").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_idx)
    df.index.name = "ds"
    df["y"] = df["y"].fillna(0.0)
    return df.reset_index()

def time_split(df, train_end, val_end):
    df = df.copy().sort_values("ds")
    train = df[df["ds"] <= train_end]
    val = df[(df["ds"] > train_end) & (df["ds"] <= val_end)]
    test = df[df["ds"] > val_end]
    return train, val, test

def scale_series(train, val, test):
    scaler = MinMaxScaler()
    train_y = train[["y"]].values
    val_y   = val[["y"]].values
    test_y  = test[["y"]].values

    scaler.fit(train_y)  # fit only on train

    train_scaled = scaler.transform(train_y).flatten()
    val_scaled   = scaler.transform(val_y).flatten()
    test_scaled  = scaler.transform(test_y).flatten()

    return train_scaled, val_scaled, test_scaled, scaler

def make_windows(series_1d: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(series_1d)):
        X.append(series_1d[i-window_size:i])
        y.append(series_1d[i])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y