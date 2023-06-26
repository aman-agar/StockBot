from keras.layers import LSTM, Input, Dense, Flatten, Dropout
from keras.models import Model

class MultiStockModel(Model):
  def __init__(self, X_train):
    super(MultiStockModel, self).__init__()

    self.DROPOUT_RATE = 0.2
    self.lstm1 = LSTM(22, return_sequences = True, name = 'LSTM-Layer-1', input_shape = (X_train.shape[1], X_train.shape[2]))
    self.lstm2 = LSTM(22, return_sequences = True, name = 'LSTM-Layer-2')
    self.lstm3 = LSTM(22, return_sequences = True, name = 'LSTM-Layer-3')
    self.dense = Dense(1, activation = 'linear', name = 'Dense')
    self.dropout = Dropout(self.DROPOUT_RATE)
    self.flatten = Flatten()

  def __call__(self, input_tensor):
    
    x = self.lstm1(input_tensor)
    x = self.lstm2(x)
    x = self.lstm3(x)
    
    x = self.dropout(x)
    x = self.flatten(x)
    x = self.dense(x)

    return x