import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch



# Get data
# Measure predictor variables (SMA, LMA, RSI, Variability, ROC)
# Input these into sigmoid function to get a prediction
# If prediction > 0.5, going up so buy, else sell
# Apply negative log loss function to measure accuracy of prediction
# Use gradient descent to optimize weights of short and long averages

def getData(ex, start, end):
    return yf.download(ex, start=start, end=end)
    

def negativeLogLoss(predictions, values):
    # Use - to turn into a minimization problem
    # Mean to get average loss over all samples
    return -(values*torch.log(predictions) + (1-values)*(torch.log(1-predictions))).mean()


# Get data
ex = "GBPUSD=X"
start = "2022-01-01"
end = "2025-12-31"
data = getData(ex, start, end)

# Create weights and bias
weights = torch.zeros(4, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

# Set learning rate and epochs
learning_rate = 0.1
epochs = 1000

# Use rolling function to get SMA, LMA and volatility (predictor variables)
# Short average over past 5 days, long average and volatility over past 20 days
# Store in dataframe as new columns to allow non-value rows to be dropped
data["SMA"] = data["Close"].rolling(window=5).mean()
data["LMA"] = data["Close"].rolling(window=20).mean()
data["Volatility"] = data["Close"].rolling(window=20).std()

data["Return"] = data["Close"].pct_change() # Increase from previous close
data["Target"] = (data["Return"].shift(-1) > 0).astype(int) # 1 if change is positive, 0 if negative

data.dropna(inplace=True)

X = torch.tensor(data[["Return", "SMA", "LMA", "Volatility"]].values, dtype=torch.float32)
y = torch.tensor(data["Target"].values, dtype=torch.float32)

# Standardize features
X = (X - X.mean(dim=0)) / X.std(dim=0)

# Train-Test 80-20 split
split = int(0.8 * len(data))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(data[["Return", "SMA", "LMA", "Volatility", "Target"]].corr())


# Training loop
for epoch in range(epochs):
    # Calculate wX + b
    linear_combination = torch.matmul(X_train, weights) + bias

    # Make prediction using sigmoid function
    prediction = torch.sigmoid(linear_combination)


    # Calculate loss
    loss = negativeLogLoss(prediction, y_train)

    # Compute gradients of weights and bias (backpropagation)
    loss.backward()

    # Update weights and bias using gradient descent
    # Wrap in torch.no_grad() since we don't need to track gradients, we just want to update the values
    # Using the gradients we already have
    with torch.no_grad():
        # Take a step in the direction of the negative gradient at the rate of learning_rate
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad

        # Reset gradients to zero
        weights.grad.zero_()
        bias.grad.zero_()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
        print(prediction[-10:])
