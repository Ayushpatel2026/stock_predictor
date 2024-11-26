'''
    This function takes in data and calculates the features for the model
    It must handle feature engineering for both training and actual production features
'''
import numpy as np
import pandas as pd
import data_analysis as da
from matplotlib import pyplot as plt


def create_features(data):

    # Lag feature: previous day's closing price
    data['lag_close_1'] = data['close'].shift(1)

    # Exponential moving average last 7 days - weighted for more recent data, better than rolling average
    data['ema_7'] = data['close'].ewm(span=7, adjust=False).mean()

    # Volatility - rolling standard deviation
    data['volatility'] = data['close'].rolling(window=7).std()

    # RSI - relative strength index - momentum indicator for past 14 days
    # helps identify overbought or oversold conditions and potential price reversals
    delta = data['close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = -np.where(delta < 0, delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Drop rows with missing values (introduced by lagging/rolling operations)
    data = data.dropna()

    return data

data = da.load_and_clean_data('data/all_stocks_5yr.csv')
feature_data = create_features(data)
print("WE ARE IN FEATURE ENGINEERING")
# plot the features compared to closing price and see if there is any correlation
da.plot_correlation_heatmap(feature_data, ["close", "lag_close_1", "ema_7", "volatility", "rsi"])