import yfinance as yf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Need to use values up to current time
def mean_reversion(close_data, bal, bound_mult):
    gbp = 1
    usd = 0
    bal.append([gbp,usd])

    # 0 if within bounds, 1 if out
    safe = 0
    # Loop over data
    i = 0
    while i < len(close_data):
        # If > bound, we want usd
        if close_data[i] > mean + bound_mult*std and safe == 0:
            usd = usd + gbp * close_data[i]
            gbp = 0
            safe = 1
         # If < bound, we want gbp
        elif close_data[i] < mean - bound_mult*std and safe == 0:
            gbp = gbp + usd / close_data[i]
            usd = 0
            safe = 1
        else:
            safe = 0
        
        #print([float(gbp), float(usd)])

        bal.append([gbp, usd])
        i += 1

    return bal

def calc_



forex_data = yf.download("GBPUSD=X", period="1y", interval="1d")


# gains = []
# for i in range(0, 20):
#     bal = mean_reversion(forex_data.iloc[:, 0], [], i/10)
#     gain = bal[-1][0]/bal[0][0] + bal[-1][1]/(bal[0][0]*forex_data.iloc[0,0])
#     gains.append(gain)

# print([float(i) for i in gains])
bal = mean_reversion(forex_data.iloc[:, 0], [], 1.6)
#print(bal[-1]/forex_data.iloc[:, 0][0])

# plt.plot(forex_data.index, forex_data.iloc[:, 0])
# plt.plot(forex_data.index, [i[0] for i in bal[1:]])
# plt.plot(forex_data.index, [i[1] for i in bal[1:]])

# plt.axhline(mean)
# plt.axhline(mean + std*bound_mult)
# plt.axhline(mean - std*bound_mult)


# plt.show()
    

   



