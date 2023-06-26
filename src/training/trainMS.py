import os
import sys
sys.path.insert(0,'src/')
from fetch.MSdata import MultiStockData
from preprocess.normalize import Normalize
from preprocess.normalize import Sequentialize
from datetime import date
from models.multistock import MultiStockModel
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

# Define Tickers
# First Ticker will be 
Tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'INTU']

# Get the data
data = MultiStockData(Tickers)
df = data()
getdir = os.getcwd()
print(getdir)
if str(date.today()) not in os.listdir(f'{getdir}\\inputs\\MS\\'):
    os.mkdir(f'{getdir}\\inputs\\MS\\{str(date.today())}')

df.to_parquet(f"{getdir}\\inputs\\MS\\{str(date.today())}\\{Tickers}.parquet")
print(df.head())

# Normalize Data
normalOb = Normalize(df)
scaler, normalizedData = normalOb.normalization()
print(normalizedData.shape)

# Sequentialize Data
seq = Sequentialize()
X_train, y_train, X_test, y_test = seq.preprocess(normalizedData, 101, 0.987, ms = True)
print(X_train.shape)
print(y_test)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Create Model object
MS = MultiStockModel(X_train)
inputs = Input(shape = (X_train.shape[1],X_train.shape[2]))
outputs = MS(inputs)
model = Model(inputs=inputs, outputs=outputs)
opt = Adam(learning_rate=1e-3)
model.compile(loss='mean_squared_error', optimizer=opt)
print(model.summary())

BATCH_SIZE = 120

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=0.1
)

model.evaluate(X_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f"{getdir}\\output\\vizualizations\\{str(date.today())}{Tickers[0]}-LossMS.png")
plt.show()

if str(date.today()) not in os.listdir(f'{getdir}\\models\\MS'):
    os.mkdir(f'{getdir}\\models\\MS\\{str(date.today())}')
model.save(f"{getdir}\\models\\MS\\{str(date.today())}\\{Tickers[0]}-MS{date.today()}.h5")

# Testing on test data
y_hat = model.predict([X_test])
print(y_hat.shape)
y_test_inverse = Normalize.inverseMinMax(scaler.min_[0], scaler.scale_[0], y_test)
y_hat_inverse = Normalize.inverseMinMax(scaler.min_[0], scaler.scale_[0],  y_hat)

plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')

plt.title(f'{Tickers[0]} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='best')
plt.savefig(f"{getdir}\\output\\vizualizations\\{str(date.today())}{Tickers[0]}-MSTest.png")
plt.show()


def Accuracy(y_hat_inverse, y_test_inverse):
    diff = abs(y_hat_inverse - y_test_inverse)
    accuracy = 100 - (100 * (diff/y_test_inverse))
    accuracy = sum(accuracy)/len(accuracy)
    return accuracy

print("Accuracy of the Model: ",Accuracy(y_hat_inverse, y_test_inverse))