import sys
sys.path.insert(0,'src/')
import pandas as pd
from datetime import timedelta
from utils.invokeClient import getClient
from datetime import date

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def getDateRange(start_dt, end_dt) -> list:
  dates = []
  for dt in daterange(start_dt, end_dt):
      dates.append(dt.strftime("%Y-%m-%d"))
  return dates

# Extracting close price from the Dataframe
def getClosePrice(df):
    dates = []
    closeprice = []
    for index, row in df['close'].items():
        closeprice.append(row)
        dates.append(index.to_pydatetime().strftime('%Y-%m-%d'))
    return dates, closeprice

# Insert dates and close price into a dataframe
def getDataframe(dates, closeprice, ticker) -> pd.DataFrame:
  new_df=pd.DataFrame()
  new_df['Date'] = dates
#   print(new_df.shape)
#   print(len(closeprice))

  new_df[ticker] = closeprice

  return new_df


def getfinalDF(Tickers, client):
    i = 0
    finalDF = pd.DataFrame()
    
    for ticker in Tickers:
        tempDF = client.get_dataframe(ticker,
                            startDate='2017-01-01',
                            endDate=str(date.today())
                            )
        closeprice = []
        # print(i)
        
        if i == 0:
            # dates = getDates(tempDF)
            dates, closeprice = getClosePrice(tempDF)
            finalDF = getDataframe(dates, closeprice, ticker)
            
        else: 
            _, closeprice = getClosePrice(tempDF)
            finalDF[ticker] = closeprice
        i+=1
    return finalDF
    

class MultiStockData:
    '''
    Create an object of the class passing Tickers as parameters
    and call the object to get the dataframe with closing prices of all the stocks in tickers
    '''
    def __init__(self, Tickers):
        super(MultiStockData, self).__init__()
        self.client = getClient()
        self.finalDF = getfinalDF(Tickers, self.client)
        self.Tickers = Tickers

    def __call__(self) -> pd.DataFrame:
        finalDF = self.finalDF
        return finalDF
  

