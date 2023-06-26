# Encoder Decoder Model
1. fetch/EDdata -> Get dataframe. 
2. evaluation/evalED -> Get future prices
3. models/encoder_decoder -> Model Architecture
4. training/train_test_ED -> Train Encoder-Decoder Model


# Multi Stock Model
1. fetch/MSdata -> Get dataframe for the tickers. Takes Tickers and return dataframe
2. vizMSData -> Get the plot of multiple stocks data
3. models/multistock -> The architecture for the Multistock model
4. training/trainMS -> Call relevant methods from all the above files and start the training process
5. evaluation/evalMS -> Get future prices

# StockBot
stockbot.py -> Set the ticker and get the BUY/HOLD/SELL call based on previous day's close price

# For ALL
1. invokeClient -> Get the tiingo client object used to access the data
2. normalize -> Normalize the dataframe and create a time series seq for LSTM. It has two classes for normalizing and sequentializing

--NOTE--
StockBot is not based on predicted values but on actual closing prices. It uses Maxima and Minima techniques to find upcoming dips

# src directory
fetch -> fetch data from tiingo
model -> model architecture
preprocess -> normalize and convert to sequence for LSTM
training -> Train the model
utils -> script for invoking client object 
evaluation -> run to get future prices

# Note
After running the models it was found that Multi-Stock model is not very useful as it is difficult to get predictions for a couple of days. The approach used to get predictions for multiple days is by adding the predicted value to the input of the model. But in case of Multi-Stock model, the model outputs only one value instead of multiple values for multiple tickers and hence the input shape changes from [num_days, num_tickers] to [num_days, 1]. 
Multi-Stock model is by far yet a theoretical concept which can be used on train and test data but difficult to implement in real scenario. 

ED on the other hand proved to be very useful in predicting the prices. 

StockBot also seems promising as it was tested manually for a given number of days and it gave accurate recommendations for 7/10 calls
