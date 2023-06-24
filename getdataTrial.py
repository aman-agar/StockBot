from utils.msdata import MultiStockData
from datetime import date

start_dt = date(2017, 1, 1)
end_dt = date(2023, 6, 18)
Tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'INTU']
data = MultiStockData(start_dt, end_dt, Tickers)
df = data()
print(df.head())