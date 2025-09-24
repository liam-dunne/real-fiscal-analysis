import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch


# Get data
# Measure predictor variables (SMA, LMA, RSI, Variability, ROC)
# Input these into sigmoid function to get a prediction
# If prediction > 0.5, going up so buy, else sell
# Apply negative log loss function to measure accuracy of prediction
# Use gradient descent to optimize weights of short and long averages

def getData(ex, period, interval):
    data = yf.download(ex, period=period, interval=interval)
    return data["Close"]

def negativeLogLoss(predictions, values):
    return -(values*torch.log(predictions) + (1-values)*(torch.log(1-predictions))).mean()


# Get data
ex = "GBPUSD=X"
period = "1y"
interval = "1h"
data = getData(ex, period, interval)

# Create weights and bias
weights = torch.zeros(24, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

# Set learning rate and epochs
learning_rate = 0.01
epochs = 1000

# Use rolling function to get SMA, LMA and Variability
# Short average over past 5 days, long average over past 20 days
sma = data.rolling(window=5).mean()
lma = data.rolling(window=20).mean()
volatility = data.rolling(window=20).std()

returns = data.pct_change() # Increase from previous close
returns = returns.shift(-1) # Shift returns so that the target is the next period, not the current period
returns = (returns > 0).astype(int) # 1 if change is positive, 0 if negative
