import os
import sys
sys.path.insert(0,'src/')
from fetch.EDdata import EncoderDecoderData
from preprocess.normalize import Normalize
from preprocess.normalize import Sequentialize
from models.encoder_decoder import EncoderDecoder
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from datetime import date

# Define Ticker (ONLY ONE)
Tickers = ['AAPL']

# Get the data
data = EncoderDecoderData(Tickers)
df = data()
getdir = os.getcwd()
print(getdir)
if str(date.today()) not in os.listdir(f'{getdir}\\inputs\\ED\\'):
    os.mkdir(f'{getdir}\\inputs\\ED\\{str(date.today())}')

df.to_parquet(f"{getdir}\\inputs\\ED\\{str(date.today())}\\{Tickers}.parquet")
print(df.head())

# Normalize Data
normalOb = Normalize(df)
scaler, normalizedData = normalOb.normalization()

decoderInputs, normalizedData = Normalize.EDdecoder(normalizedData)
print("Encoder Input Shape",normalizedData.shape)
print("Decoder Input Shape",decoderInputs.shape)

# Sequentialize Data
seq = Sequentialize()
X_train, y_train, X_test, y_test = seq.preprocess(normalizedData, 101, 0.987, ms = False)
X_train_Decoder, y_train_Decoder, X_test_Decoder, y_test_Decoder = seq.preprocess(decoderInputs, 101, 0.987)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train_Decoder.shape)
print(y_train_Decoder.shape)

# Create Model object
ED = EncoderDecoder(X_train)
encoder_inputs = Input(shape = (X_train.shape[1],X_train.shape[2]))
decoder_inputs = Input(shape = (X_train_Decoder.shape[1], X_train_Decoder.shape[2]))
print(encoder_inputs.shape)
print(decoder_inputs.shape)
outputs = ED(encoder_inputs, decoder_inputs)
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
opt = Adam(learning_rate=1e-3)
model.compile(loss='mean_squared_error', optimizer=opt)
print(model.summary())

BATCH_SIZE = 120

history = model.fit(
    [X_train, X_train_Decoder], y_train,
    validation_data=([X_test, X_test_Decoder], y_test),
    epochs=100,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=0.1
)

model.evaluate([X_test, X_test_Decoder], y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f"{getdir}\\output\\vizualizations\\{str(date.today())}{Tickers[0]}-LossED.png")
plt.show()
if str(date.today()) not in os.listdir(f'{getdir}\\models\\ED\\'):
    os.mkdir(f'{getdir}\\models\\ED\\{str(date.today())}')
model.save(f"{getdir}\\models\\ED\\{str(date.today())}\\{Tickers[0]}-Encoder-Decoder{date.today()}.h5")

# Testing on test data
y_hat = model.predict([X_test, X_test_Decoder])

print(y_hat.shape)
y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)

plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')

plt.title(f'{Tickers[0]} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='best')
plt.savefig(f"{getdir}\\output\\vizualizations\\{str(date.today())}{Tickers[0]}-EDTest.png")
plt.show()


def Accuracy(y_hat_inverse, y_test_inverse):
    diff = abs(y_hat_inverse - y_test_inverse)
    accuracy = 100 - (100 * (diff/y_test_inverse))
    accuracy = sum(accuracy)/len(accuracy)
    return accuracy

print("Accuracy of the Model: ",Accuracy(y_hat_inverse, y_test_inverse))