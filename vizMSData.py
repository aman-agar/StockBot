import matplotlib.pyplot as plt
from src.fetch.MSdata import MultiStockData
from datetime import date

plt.style.use('seaborn')
# start_dt = date(2017, 1, 1)
# end_dt = date(2023, 6, 18)

# Define the Tickers 
Tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'INTU']

# Create an object of MultiStockData class
data = MultiStockData(Tickers)
# Call the object to get the daatframe
df = data()
print(df.head())

fig = plt.figure(figsize = (14,8))
fig.tight_layout()

# Plot the graph
for ticker in Tickers:

    plt.plot(df['Date'], df[ticker])
plt.legend(Tickers)

plt.savefig("Multiple Stocks.png")
# plt.show()