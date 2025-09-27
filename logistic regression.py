import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch



# Get data
# Measure predictor variables (SMA, LMA, RSI, Volatility, MACD)
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
weights = torch.zeros(3, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

# Set learning rate and epochs
learning_rate = 0.03
epochs = 1000


# Use rolling function to get SMA, LMA and volatility (predictor variables)
# Short average over past 5 days, long average and volatility over past 20 days
#data["SMA"] = data["Close"].rolling(window=5).mean()
#data["LMA"] = data["Close"].rolling(window=20).mean()
data["Volatility"] = data["Close"].rolling(window=20).std()

data["Return"] = data["Close"].pct_change() # Increase from previous close
data["Target"] = (data["Close"].pct_change(periods=14).shift(-1) > 0).astype(int) # 1 if change is positive, 0 if negative

# Calculate 14 day RSI
# Use np.clip to restrict values to be >=0 or <=0
gain = np.clip(data["Return"], min=0) # Take only positive returns
loss = -np.clip(data["Return"], max=0) # Take only negative returns, - to convert to positive losses
avg_gain = gain.rolling(window=14).mean() 
avg_loss = loss.rolling(window=14).mean()
data["RSI"] = 100 - (100/(1+(avg_gain/avg_loss)))

# Calculate EMAs and MACD
# We set adjust to false to use the standard recursive formula
# Use EMAs since they are more reactive than SMAs, and forex is a fast-moving market
ema_12 = data["Close"].ewm(span=12, adjust=False).mean() # 12-day EMA
ema_26 = data["Close"].ewm(span=26, adjust=False).mean() # 26-day EMA
data["MACD"] = ema_12 - ema_26

data.dropna(inplace=True)

X = torch.tensor(data[["MACD", "Volatility", "RSI"]].values, dtype=torch.float32)
y = torch.tensor(data["Target"].values, dtype=torch.float32)

# Standardize features
X = (X - X.mean(dim=0)) / X.std(dim=0)

# Train-Test 80-20 split
split = int(0.8 * len(data))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(data[["Return", "Volatility", "RSI", "Target"]].corr())

# Create array to track loss
losses = []

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

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    losses.append(loss.item())

# Evaluate on test set
# Use torch.no_grad() since we are just predicting, not training
with torch.no_grad():
    linear_combination = torch.matmul(X_test, weights) + bias
    prediction = torch.sigmoid(linear_combination)
    pred_classes = (prediction >= 0.5).float() # Convert to float since y_test is float
    accuracy = (pred_classes == y_test).float().mean() # Element wise boolean comparison, convert to 0s and 1s, then take mean for accuracy

print(accuracy.item())
print(pred_classes)

# Calculate values for confusion matrix
TP = ((pred_classes == 1) & (y_test == 1)).sum().item() # Use item since classes are tensors (of float type)
FP = ((pred_classes == 1) & (y_test == 0)).sum().item()
TN = ((pred_classes == 0) & (y_test == 0)).sum().item()
FN = ((pred_classes == 0) & (y_test == 1)).sum().item()
# Create confusion matrix
cm = torch.tensor([[TN, FP],
                    [FN, TP]])

# Create figure
fig, ax = plt.subplots(1, 2)
fig.tight_layout() # Improve spacing between figures
fig.set_figwidth(10) # Increase width of plots

# Graph losses over epochs
ax[0].plot(np.arange(1, epochs+1, 1), losses, label="loss")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("epochs")

# Show confusion matrix
ax[1].imshow(cm)
ax[1].set_ylabel("Actual")
ax[1].set_xlabel("Predicted")
ax[1].set_xticks([0, 1])
ax[1].set_yticks([0, 1])
# Add labels to confusion matrix
for i in range(len(cm)):
    for j in range(len(cm)):
        ax[1].text(i, j, cm[i][j].item(), ha="center", va="center", color="black", fontsize="15")

plt.show()