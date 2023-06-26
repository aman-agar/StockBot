import sys
sys.path.insert(0,'src/')
from fetch.EDdata import EncoderDecoderData
from preprocess.normalize import Normalize, Sequentialize
from keras.models import load_model
from datetime import date, timedelta
import datetime
import numpy as np
import pandas as pd
import os

# Define Ticker (ONLY ONE)
Tickers = ['AAPL']
getdir = os.getcwd()

if str(date.today()) not in os.listdir(f'{getdir}\\inputs\\ED\\'):
  raise Exception(f"The model was not run on {date.today()}")


df = pd.read_parquet(f'{getdir}\\inputs\\ED\\{date.today()}\\{Tickers}.parquet')
print(df.head())

# Normalize Data
normalOb = Normalize(df)
scaler, normalizedData = normalOb.normalization()
decoderInputs, normalizedData = Normalize.EDdecoder(normalizedData)
print(f"Normalized shape : {normalizedData.shape}")
normalizedData = normalizedData[normalizedData.shape[0]-102:]
decoderInputs = decoderInputs[decoderInputs.shape[0]-102:]
print(normalizedData.shape)

seq = Sequentialize()
X, Y = seq.preprocessEval(normalizedData, 101) 
X_decoder, Y_decoder = seq.preprocessEval(decoderInputs, 101)
print("X : ", X.shape)
print("Y : ", Y.shape)
print(X)
print(Y)
print("X-Decoder: ", X_decoder.shape)
print("X-Decoder: ", X_decoder)


# Load the model
model = load_model(f"{getdir}\\models\\ED\\{str(date.today())}\\{Tickers[0]}-Encoder-Decoder{date.today()}.h5", compile = False)

i=0
temp_input = normalizedData
temp_input = temp_input.tolist()
decoder_temp_input = decoderInputs
decoder_temp_input = decoder_temp_input.tolist()

output = []
# Predict for 30 days
while(i<3):
  y_hat = model.predict([X, X_decoder])
  print("Y_hat: ", y_hat.shape)
  out = Normalize.inverseMinMax(scaler.min_, scaler.scale_, y_hat)
  output.append(out)
  print(f"{i+1}: {out}")
  temp_input.append(y_hat[0].tolist())
  temp_input = temp_input[1:]
  closeTestprice = np.array(temp_input)
  
  decoder_temp_input.append(y_hat[0].tolist())
  decoder_temp_input = decoder_temp_input[1:]
  closeTestpriceDecoder = np.array(decoder_temp_input)

  X, Y = seq.preprocessEval(closeTestprice, 101)
  X_decoder, Y_decoder = seq.preprocessEval(closeTestpriceDecoder, 101)
  i+=1

predictions = []

for value in output:
  predictions.append(value[0][0])

preddf = pd.DataFrame()
preddf['Predictions'] = predictions
preddf.to_csv(f"{getdir}\\log\\ED\\{Tickers}-{date.today()}.csv")