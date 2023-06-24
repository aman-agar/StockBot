from utils.MSdata import MultiStockData
from utils.normalize import Normalize
from utils.normalize import Sequentialize
from datetime import date
from models.multistock import MultiStockModel
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

# Define Tickers
Tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'INTU']

# Get the data
data = MultiStockData(Tickers)
df = data()
print(df.head())

# Normalize Data
normalOb = Normalize(df)
scaler, normalizedData = normalOb.normalization()
print(normalizedData.shape)

# Sequentialize Data
seq = Sequentialize()
X_train, y_train, X_test, y_test = seq.preprocess(normalizedData, 101, 0.987)
print(X_train.shape)
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
plt.show()

model.save("models/MultiStockModel.h5")
