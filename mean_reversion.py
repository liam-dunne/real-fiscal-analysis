import yfinance as yf
import numpy as np
import pandas as pd

def mean_reversion(close_data, bal):
    gbp = 1
    usd = 0

    mean = close_data.mean()
    std = close_data.std()
    bound = 1
    # Loop over data
    i = 0
    while i < len(close_data):
        # If > 1.5std + mean, we want usd
        if close_data[i] > mean + bound*std:
            usd = usd + gbp * close_data[i]
            gbp = 0
         # If < mean - 1.5std, we want gbp
        else:
            gbp = gbp + usd / close_data[i]
            usd = 0
        
        print([float(gbp), float(usd)])

        bal.append([gbp, usd])
        i += 1

    return bal


forex_data = yf.download("GBPUSD=X", period="1y", interval="1h")

bal = mean_reversion(forex_data.iloc[:, 0], [])
print(bal[-1]/forex_data.iloc[:, 0][0])
print()
    

   



