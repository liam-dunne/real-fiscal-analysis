# For data manipulation
import numpy as np
import pandas as pd

# To fetch financial data
import yfinance as yf

# For visualisation
import matplotlib.pyplot as plt


# Set the ticker as 'EURUSD=X'
forex_data = yf.download('GBPUSD=X', period='2y', interval='1d')

# Set the index to a datetime object
forex_data.index = pd.to_datetime(forex_data.index)

# Display the last five rows
print(forex_data.head())
print(forex_data.columns)
forex_data = forex_data.drop(columns=["Volume"])


plt.plot(forex_data.index, forex_data["Close"], label="Close")
# plt.show()

shortPeriod = 5
longPeriod = 100


# Creates a list of moving averages at each time interval starting at the periods' index
shortMovingAverage = [sum(forex_data.iloc[i:i+shortPeriod,0])/shortPeriod for i in range(len(forex_data.index)-shortPeriod)]
longMovingAverage = [sum(forex_data.iloc[i:i+longPeriod,0])/longPeriod for i in range(len(forex_data.index)-longPeriod)]

plt.plot(forex_data.index[shortPeriod:], shortMovingAverage, label="SMA")
plt.plot(forex_data.index[longPeriod:], longMovingAverage, label="LMA")
plt.legend()
plt.xticks(rotation=45)
# plt.show()


gbp = 1
usd = 0
print(gbp * forex_data.iloc[longPeriod, 0])

# initial conditions ie call api and calc avg once
shortMovingAverage = shortMovingAverage[longPeriod - shortPeriod:]
if shortMovingAverage[0] > longMovingAverage[0]:
    shortPos = 1
    
else:
    usd = gbp * forex_data.iloc[longPeriod, 0]
    gbp = 0
    shortPos = -1
# previousPos = shortPos

bal = [[gbp, usd]]
i = 1

while i < len(longMovingAverage):
    # call api
    
    # calc averages

    # # check if we wanna buy / sell
    # if shortMovingAverage[i] > longMovingAverage[i]:
    #     shortPos = 1
    # else:
    #     shortPos = -1
    
    # # short moving below long ie selling usd into gbp
    # if shortPos > previousPos:
    #     gbp = usd / forex_data.iloc[i + longPeriod, 0]
    #     usd = 0
    # # short moving above long ie buying usd with gbp
    # elif shortPos < previousPos:
    #     usd = gbp * forex_data.iloc[i + longPeriod, 0]
    #     gbp = 0


    if shortMovingAverage[i] > longMovingAverage[i] and shortPos==-1:
        gbp = usd / forex_data.iloc[i + longPeriod, 0]
        usd = 0
        shortPos = 1
    elif shortMovingAverage[i] < longMovingAverage[i] and shortPos==1:
        usd = gbp * forex_data.iloc[i + longPeriod, 0]
        gbp = 0
        shortPos = -1

    # previousPos = shortPos
    i += 1
    
    
    bal.append([gbp, usd])

    # wait for new call for api

print(bal[-1][0], bal[-1][1] / forex_data.iloc[-1,0])

plt.plot(forex_data.index[longPeriod:], [i[0] for i in bal], label="GBP")
plt.plot(forex_data.index[longPeriod:], [i[1] for i in bal], label="USD")

plt.show()
