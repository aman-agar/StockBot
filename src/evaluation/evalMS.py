import sys
sys.path.insert(0,'src/')
from preprocess.normalize import Normalize, Sequentialize
from keras.models import load_model
from datetime import date, timedelta
import datetime
import numpy as np
import pandas as pd
import os

Tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'INTU']
getdir = os.getcwd()

if str(date.today()) not in os.listdir(f'{getdir}\\inputs\\MS\\'):
  raise Exception(f"The model was not run on {date.today()}")


df = pd.read_parquet(f'{getdir}\\inputs\\MS\\{date.today()}\\{Tickers}.parquet')
print(df.head())

# Normalize Data
normalOb = Normalize(df)
scaler, normalizedData = normalOb.normalization()
normalizedData = normalizedData[normalizedData.shape[0]-102:]
print(normalizedData.shape)


seq = Sequentialize()
X, Y = seq.preprocessEval(normalizedData, 101)
print("X : ", X.shape)
print("Y : ", Y.shape)

# Load the model
model = load_model(f"{getdir}\\models\\MS\\{str(date.today())}\\{Tickers[0]}-MS{date.today()}.h5", compile = False)
y_hat = model.predict(X)
out = Normalize.inverseMinMax(scaler.min_[0], scaler.scale_[0], y_hat)
print(y_hat)