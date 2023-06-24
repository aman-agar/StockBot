from tensorflow.keras.layers import LSTM, Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

class EncoderDecoder(Model):
  def __init__(self):
    super(EncoderDecoder, self).__init__()

    self.encoder = LSTM(20, return_sequence = True, name = 'Encoder')
    self.decoder = LSTM(20, return_sequence = True, return_state = True, name = 'Decoder')
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



