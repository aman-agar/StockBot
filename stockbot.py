# Calculate the small delta that is delta = sign(ci+1 - ci) 
# Calculate the capital delta that is DELTA = sign(deltai+1 - deltai) 
# Get the decision according to DELTA and predict the buy or sell call

# Works on actual previous closing price

# The concept is to calculate the maxima and minima to estimate the trend
# Treating i+2 as previous close price

import pandas as pd
import numpy as np
import os
from datetime import date
from src.utils.invokeClient import getClient
from src.fetch.EDdata import getfinalDF
Tickers = ['MSFT']

getdir = os.getcwd()
clientInvoked = False
if str(date.today()) not in os.listdir(f'{getdir}\\inputs\\SB\\') or 'Tickers[0].parquet' not in os.listdir(f'{getdir}\\inputs\\SB\\{date.today()}\\'):
    print("Saved file not found! Fetching data from API...")
    client = getClient()
    df = getfinalDF(Tickers, client, start_dt = '2023-01-01')
    clientInvoked = True
else:
    df = pd.read_parquet(f'{getdir}\\inputs\\SB\\{date.today()}\\{Tickers}.parquet')

df = df.tail()
df.reset_index(inplace=True)
df.drop(['index'], axis =1, inplace = True)
print(df.head())
# print(df.shape)
if clientInvoked == True:
    if str(date.today()) not in os.listdir(f'{getdir}\\inputs\\SB\\'):
        os.mkdir(f'{getdir}\\inputs\\SB\\{str(date.today())}')
    df.to_parquet(f"{getdir}\\inputs\\SB\\{str(date.today())}\\{Tickers[0]}.parquet")

prices = df[Tickers[0]].to_list()

ci2 = prices[-1] 
ci1 = prices[-2]
ci = prices[-3]

deltai1 = np.sign(ci2 - ci1)
deltai = np.sign(ci1 - ci) 
DELTA = deltai1 - deltai

def decision(argument):
    switcher = {
        -2: "-----BUY-----",
        0: "-----HOLD-----",
        2: "-----SELL-----",
    }
    return switcher.get(argument, "NOT VALID")

print (decision(DELTA))

# Calculation for given prices- 
#            Date    AAPL
# 2023-06-16  184.92
# 2023-06-20  185.01
# 2023-06-21  183.96
# 2023-06-22  187.00
# 2023-06-23  186.68


# ci+2 = 186.68 (Considering 23-06-2023 as previous close )
# ci+1 = 187.00
# delta i+1 = -1
# ci+1 = 187.00
# ci = 183.96
# delta i = 1

# DELTA = -1 - 1 = 0
# Hence recommend HOLD  