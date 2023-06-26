from keras.layers import LSTM, Dense, Dropout, Flatten, Input
from keras.models import Model


class EncoderDecoder(Model):
  def __init__(self, X_train):
    super(EncoderDecoder, self).__init__()

    self.encoder = LSTM(22, return_sequences = True, return_state = True, name = 'Encoder', input_shape = (X_train.shape[1], X_train.shape[2]))
    self.decoder = LSTM(22, return_sequences = True, return_state = True, name = 'Decoder')
    self.dense = Dense(1, activation = 'linear', name = 'Dense')
    self.dropout = Dropout(0.2)
    self.flatten = Flatten()

  def __call__(self, encoder_inputs, decoder_input):
    encoderOutput, state_h, state_c = self.encoder(encoder_inputs)
    encoderState = [state_h, state_c]
    decoderOutput, _, _ = self.decoder(decoder_input, initial_state = encoderState)
    x = self.dropout(decoderOutput)
    x = self.flatten(x)
    denseOutput = self.dense(x)

    return denseOutput





