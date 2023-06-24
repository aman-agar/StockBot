import numpy as np
import pandas as pd
from datetime import timedelta, date
from utils.invokeClient import getClient

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def getDateRange(start_dt, end_dt) -> list:
  dates = []
  for dt in daterange(start_dt, end_dt):
      dates.append(dt.strftime("%Y-%m-%d"))
  return dates

# Extracting close price from the Dataframe
def getClosePrice(df) -> list:
  closeprice = []
  for i in range(df.shape[0]):
    closeprice.append(df['close'][i])
  return closeprice

# Get dates and close price into a dataframe
def getDataframe(dates, closeprice, ticker) -> pd.DataFrame:
  new_df=pd.DataFrame()
  new_df['Date'] = dates
  print(new_df.shape)
  print(len(closeprice))

  new_df[ticker] = closeprice

  return new_df

def getfinalDF(Tickers, client, dates):
    i = 0
    finalDF = pd.DataFrame()
    
    for ticker in Tickers:
        tempDF = client.get_dataframe(ticker,
                            startDate='2017-01-01',
                            endDate='2023-06-18'
                            )
        closeprice = []
        print(i)
        closeprice = getClosePrice(tempDF)
        if i == 0:
            finalDF = getDataframe(dates, closeprice, ticker)
            print(len(dates))
            print(finalDF.head())
        else: 
            finalDF[ticker] = closeprice
        i+=1
    return finalDF
    

class MultiStockData():
    '''
    Create an object of the class passing start date, end date and Tickers as parameters
    and call the object to get the dataframe with closing prices of all the stocks in tickers
    '''
    def __init__(self, start_dt, end_dt, Tickers):
        super(MultiStockData, self).__init__()
        self.client = getClient()
        self.dates = getDateRange(start_dt, end_dt)
        self.finalDF = getfinalDF(Tickers, self.client, self.dates)
        self.Tickers = Tickers

    def __call__(self) -> pd.DataFrame:
        finalDF = getfinalDF(self.Tickers, self.client, self.dates)
        return finalDF
  

