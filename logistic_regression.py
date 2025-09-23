import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch


# Take previous day
# Sigmoid function on (bias + w * rate) gives probability 0-1
# If output > 0.5 then money up, < 0.5 money down
# Apply negative log loss function to values
# Choose weights which minimise loss function


def negativeLogLoss(predictions, values):
    return -(values*torch.log(predictions) + (1-values)*(torch.log(1-predictions))).sum()
     

def optimiseWeights(loss, weights, bias, learningRate):
    # Uses gradient optimisation with pytorch
    loss.backward()

    with torch.no_grad():
        weights -= weights.grad*learningRate
        bias -= bias.grad*learningRate
    
    weights.grad.zero_()
    bias.grad.zero_()

    return weights, bias

def predict(close_data):
    i = 24
    predictions = []
    values = []

    # Taking past 24 readings
    # Predicting reading in 1hr
    # Check next hours reading to see if its higher
    while i < len(closeData)-1:
        pastDayRates = torch.tensor(closeData[i-24:i], dtype=torch.float)
        p = torch.sigmoid(bias + weights.dot(pastDayRates))

        # Add our predicted class probability (p going up)
        predictions.append(p)

        # Add actual class
        if closeData[i+1] >= closeData[i]:
            values.append(torch.tensor(1))
        else:
            values.append(torch.tensor(0))

        i += 1
    return torch.stack(predictions), torch.tensor(values, dtype=torch.float)



forexData = yf.download("GBPUSD=X", period="1y", interval="1h")

closeData = forexData.iloc[:, 0]

weights = torch.zeros(24, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

learningRate = 0.1
epochs = 100


for j in range(epochs):
    predictions, values = predict(closeData)
    loss = negativeLogLoss(predictions, values)
    print(loss)
    weights, bias = optimiseWeights(loss, weights, bias, learningRate)

print(loss.item())


