# For data manipulation
import numpy as np
import pandas as pd

# To fetch financial data
import yfinance as yf

# For visualisation
import matplotlib.pyplot as plt


# Set the ticker as 'EURUSD=X'
forex_data = yf.download('GBPUSD=X', period='1y', interval='1d')

# Set the index to a datetime object
forex_data.index = pd.to_datetime(forex_data.index)

# Display the last five rows
print(forex_data.tail())
print(forex_data.columns)
forex_data = forex_data.drop(columns=["Volume"])
print(forex_data.tail())


plt.plot(forex_data.index, forex_data)
plt.legend([i[0] for i in forex_data.columns])
plt.show()