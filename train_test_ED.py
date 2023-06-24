from models.encoder_decoder import EncoderDecoder
import tensorflow as tf
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tiingo import TiingoClient
from datetime import timedelta, date
from tensorflow.keras.layers import Input


plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Tiingo API Setup and Data Collection
config = {}
config['session'] = True
config['api_key'] = "44c3cbbc4115125fe0f7012c67b29d10f0be5a00"
df = pd.DataFrame()

try:
    client = TiingoClient(config)

    df = client.get_dataframe('BTCUSD',
                            startDate='2017-01-01',
                            endDate='2023-06-18'
                            )
except:
    print("Error while fetching data from Tiingo")

dates=[]
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
start_dt = date(2017, 1, 1)
end_dt = date(2023, 6, 18)
for dt in daterange(start_dt, end_dt):
    dates.append(dt.strftime("%Y-%m-%d"))


closeprice=[]
for i in range(df.shape[0]):
    closeprice.append(df['close'][i])
print(len(closeprice))


new_df=pd.DataFrame()
new_df['Date']=dates
new_df['Close']=closeprice
df=new_df

ax = df.plot(x='Date', y='Close');
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")


# Data Preprocessing
scaler = MinMaxScaler()
close_price = df.Close.values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)
print("Shape of scaled Data: ", scaled_close.shape)


SEQ_LEN = 100
def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.985)


# Creating object of ED model and compiling
def train():
    ED = EncoderDecoder()
    encoder_inputs = tf.keras.layers.Input(shape = X_train)

#--------------------------------------Decoder-Inuput-Setup---------------------------------------------------------------------------------------
    decoder_inputs = tf.keras.layers.Input(shape = )
    outputs = ED(encoder_inputs, decoder_inputs)
    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    opt=tf.keras.optimizers.Adam(learning_rate=1e-3)
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

    print("------------------Training Completed-------------------")

    model.evaluate(X_test, y_test)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    y_hat = model.predict(X_test)
    print(y_hat.shape)
    y_test_inverse = scaler.inverse_transform(y_test)
    y_hat_inverse = scaler.inverse_transform(y_hat)
 
    plt.plot(y_test_inverse, label="Actual Price", color='green')
    plt.plot(y_hat_inverse, label="Predicted Price", color='red')
 
    plt.title('Bitcoin price prediction')
    plt.xlabel('Time [days]')
    plt.ylabel('Price')
    plt.legend(loc='best')
 
    plt.show()
    print("--------------Training and Validation Complete----------")
    
    def Accuracy(y_hat_inverse, y_test_inverse):
        diff = abs(y_hat_inverse - y_test_inverse)
        accuracy = 100 - (100 * (diff/y_test_inverse))
        accuracy = sum(accuracy)/len(accuracy)
        return accuracy
    
    print("Accuracy of the Model: ", Accuracy(y_hat_inverse, y_test_inverse))


