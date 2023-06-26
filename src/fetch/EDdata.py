import sys
sys.path.insert(0,'src/')
import pandas as pd
from datetime import timedelta
from utils.invokeClient import getClient
from datetime import date

# Extracting close price from the Dataframe
def getClosePrice(df):
    dates = []
    closeprice = []
    for index, row in df['close'].items():
        closeprice.append(row)
        dates.append(index.to_pydatetime().strftime('%Y-%m-%d'))
    return dates, closeprice


def getDataframe(dates, closeprice, ticker) -> pd.DataFrame:
  new_df=pd.DataFrame()
  new_df['Date'] = dates
#   print(new_df.shape)
#   print(len(closeprice))

  new_df[ticker] = closeprice

  return new_df

def getfinalDF(Tickers, client, start_dt='2017-01-01'):
    i = 0
    finalDF = pd.DataFrame()
    
    for ticker in Tickers:
        tempDF = client.get_dataframe(ticker,
                            startDate=start_dt,
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
    

class EncoderDecoderData:
    '''
    Create an object of the class passing Tickers as parameter
    and call the object to get the dataframe with closing prices of the stock
    !CAUTION!
    EncoderDecoder Model is built for only one stock at a time. 
    '''
    def __init__(self, Tickers, start_dt = '2017-01-01'):
        super(EncoderDecoderData, self).__init__()
        self.client = getClient()
        self.finalDF = getfinalDF(Tickers, self.client, start_dt)
        self.Tickers = Tickers

    def __call__(self) -> pd.DataFrame:
        finalDF = self.finalDF
        return finalDF
  
