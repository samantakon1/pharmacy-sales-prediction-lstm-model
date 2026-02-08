def naive_forecast(series):
    return series.shift(1)


def moving_average_forecast(series, window):
    return series.rolling(window=window).mean().shift(1)