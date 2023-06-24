from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class Normalize:
    '''
    Normalize a dataframe using MinMaxScaler
    '''
    def __init__(self, df):
        super(Normalize, self).__init__()
        self.df = df
        self.scaler = MinMaxScaler()
    
    def normalization(self):
        # Drop Dates, normalize using MinMaxScaler
        self.df.drop(['Date'], axis = 1, inplace = True)
        # print(self.df.head())
        x = self.df.values # returns a numpy array
        x_scaled = self.scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return self.scaler, x_scaled
    
class Sequentialize():
    '''
    Create a sequence for time series data'''
    def __init__(self) -> None:
        super(Sequentialize, self).__init__()
        # self.seq_len = SEQ_LENGTH
        self.train_split = 0.987

    def to_sequences(self, data, seq_len):
        d = []
        for index in range(len(data) - seq_len):
            d.append(data[index: index + seq_len])
        # print(np.array(d).shape)
        return np.array(d)
    
    def preprocess(self, data_raw, seq_len, train_split):
        data = self.to_sequences(data_raw, seq_len)
        num_train = int(self.train_split * data.shape[0])
        X_train = data[:num_train, :-1, :]
        y_train = data[:num_train, -1, :1]

        X_test = data[num_train:, :-1, :]
        y_test = data[num_train:, -1, :1]
        return X_train, y_train, X_test, y_test

