# For data manipulation
import numpy as np
import pandas as pd

# To fetch financial data
import yfinance as yf

# For visualisation
import matplotlib.pyplot as plt




def calcMovingAverages(shortPeriod, longPeriod):
    """ 
    Creates a list of moving averages at each time interval starting at the periods' index
    """
    shortPeriod = int(shortPeriod)
    longPeriod = int(longPeriod)
    shortMovingAverage = [sum(forex_data.iloc[i:i+shortPeriod,0])/shortPeriod for i in range(len(forex_data.index)-shortPeriod)]
    longMovingAverage = [sum(forex_data.iloc[i:i+longPeriod,0])/longPeriod for i in range(len(forex_data.index)-longPeriod)]
    return shortMovingAverage, longMovingAverage

def runSim(shortPeriod, longPeriod):
    """
    Runs the trading algorithm over the given time period
    """
    shortPeriod = int(shortPeriod)
    longPeriod = int(longPeriod)

    shortMovingAverage, longMovingAverage = calcMovingAverages(shortPeriod, longPeriod)

    gbp = 1
    usd = 0
    # print(gbp, gbp * forex_data.iloc[longPeriod, 0])


    # initial conditions ie call api and calc avg once
    shortMovingAverage = shortMovingAverage[longPeriod - shortPeriod:]
    if shortMovingAverage[0] > longMovingAverage[0]:
        usd = gbp * forex_data.iloc[longPeriod, 0]
        gbp = 0
        shortPos = 1
    else:
        # usd = gbp / forex_data.iloc[longPeriod, 0]
        # gbp = 0
        shortPos = -1

    bal = [[gbp, usd]]
    i = 1

    # print(len(shortMovingAverage), len(longMovingAverage))
    # print(shortPeriod, longPeriod)
    while i < len(longMovingAverage):
        # call api
        
        # calc averages

        # BUY USD AT A HIGH, ie WHEN SHORT GOES ABOVE LONG
        # SELL USD AT A LOW, ie WHEN SHORT GOES BELOW LONG
        if shortMovingAverage[i] > longMovingAverage[i] and shortPos==-1:
            usd = gbp * forex_data.iloc[i + longPeriod, 0]
            gbp = 0

            # gbp = usd * forex_data.iloc[i + longPeriod, 0]
            # usd = 0
            
            shortPos = 1
        elif shortMovingAverage[i] < longMovingAverage[i] and shortPos==1:
            gbp = usd / forex_data.iloc[i + longPeriod, 0]
            usd = 0

            # usd = gbp / forex_data.iloc[i + longPeriod, 0]
            # gbp = 0
            shortPos = -1

        i += 1
        
        bal.append([gbp, usd])

        # wait for new call for api

        # print(bal[-1][0], bal[-1][1])


    # # graphing the short and long moving averages against the closing times
    # # This code should only be used for singular short and long period
    # plt.plot(forex_data.index, forex_data["Close"], label="Close")
    # plt.plot(forex_data.index[longPeriod:], shortMovingAverage, label="SMA")
    # plt.plot(forex_data.index[longPeriod:], longMovingAverage, label="LMA")
    # plt.legend()
    # plt.show()
    # return bal


    # This is the other code for varying short and long periods
    if bal[-1][0] == 0:
        return bal[-1][1] / (forex_data.iloc[longPeriod][0])
    else:
        return bal[-1][0] 
    





# Set the ticker as 'EURUSD=X'
forex_data = yf.download('GBPUSD=X', period='2y', interval='1h')
print(forex_data.head())
# Set the index to a datetime object
forex_data.index = pd.to_datetime(forex_data.index)

# # Display the last five rows; and column names
# print(forex_data.head())
# print(forex_data.columns)
forex_data = forex_data.drop(columns=["Volume"])





# # Running one simulation over a specific short and long period to graph its behaviour
# shortPeriod = 16 
# longPeriod = 2400
# bal = runSim(shortPeriod, longPeriod)

# if bal[-1][0] == 0:
#     print(bal[-1][1] / (forex_data.iloc[longPeriod][0]))
# else:
#     print(bal[-1][0]) 

# plt.plot(forex_data.index[longPeriod:], [i[0] for i in bal], label="GBP")
# plt.plot(forex_data.index[longPeriod:], [i[1] for i in bal], label="USD")
# plt.show()




# Running simulations over varying short and long periods
shortVar = np.linspace(1, 51, 11)
longVar = np.linspace(10 * 24, 110 * 24, 11)
shortLat, longLat = np.meshgrid(shortVar, longVar)
runSimVect = np.vectorize(runSim)
simResults = runSimVect(shortLat, longLat)
plt.imshow(simResults, cmap="hot")
plt.colorbar()
plt.xticks(np.linspace(0,10,11), labels=shortVar, rotation=45)
plt.yticks(np.linspace(0,10,11), labels=longVar)
plt.show()