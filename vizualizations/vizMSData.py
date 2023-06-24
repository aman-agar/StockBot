import matplotlib.pyplot as plt
from utils.msdata import MultiStockData
from datetime import date

plt.style.use('fivethirtyeight')
start_dt = date(2017, 1, 1)
end_dt = date(2023, 6, 18)
Tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'INTU']
data = MultiStockData(start_dt, end_dt, Tickers)
df = data()

fig = plt.figure(figsize = (14,8))
plt.plot(x = 'Date', y = df[Tickers[0]])
plt.plot(df[Tickers[1]])
plt.plot(df[Tickers[2]])
plt.plot(df[Tickers[3]])
plt.plot(df[Tickers[4]])
plt.savefig("Multiple Stocks.png")
plt.show()