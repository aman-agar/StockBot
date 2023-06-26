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
    
    @staticmethod
    def EDdecoder(x_scaled): 
        '''
        Decoder Inputs for Encoder-Decoder Model'''
        decoderInputs = x_scaled[::-1]
        print(decoderInputs.shape)
        decoderInputs = decoderInputs[:-1]
        decoderInputs = decoderInputs[::-1]
        x_scaled = x_scaled[:-1]
        print(decoderInputs.shape)
        return decoderInputs, x_scaled
    
    @staticmethod
    def inverseMinMax(min_, scale_, X):
        X -= min_
        X /= scale_
        return X

    
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
    
    def preprocess(self, data_raw, seq_len, train_split, ms=False):
        '''
        Preprocess the data from training
        '''
        data = self.to_sequences(data_raw, seq_len)
        y_train = []
        y_test = []
        num_train = int(self.train_split * data.shape[0])
        X_train = data[:num_train, :-1, :]
        X_test = data[num_train:, :-1, :]
        if ms == True:
            y_train = data[:num_train, -1, :1]
            y_test = data[num_train:, -1, :1]
        else:
            y_train = data[:num_train, -1, :]
            y_test = data[num_train:, -1, :]
        
        return X_train, y_train, X_test, y_test
    
    def preprocessEval(self, data_raw, seq_len, ms=False):
        '''
        PreProcess the data for evaluation
        '''
        data = self.to_sequences(data_raw, seq_len)
        print(data.shape)
        X = data[:, :-1, :]
        if ms == True:
            Y = data[:, -1, :1]
        else:
            Y = data[:, -1, :]
        
        return X, Y


