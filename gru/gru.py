from keras.models import Sequential, Model
from keras.layers import Input, GRU, LSTM, Dense, RepeatVector
import numpy as np

# GRU Encoder-Decoder (Many to one)

class GruEncoderDecoder:
    def __init__(self, timesteps, hidden_units):
        self.timesteps = timesteps
        self.hidden_units = hidden_units
        self.num_features = 1
        encoder_inputs = Input(shape=(self.timesteps, self.num_features))
        encoder = GRU(self.hidden_units, return_state=True)
        encoder_outputs, state_h = encoder(encoder_inputs)
        decoder_inputs = Input(shape=(None, self.num_features))
        decoder_gru = GRU(self.hidden_units, return_sequences=True)
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
        decoder_dense = Dense(1, activation='sigmoid')
        decoder_outputs = decoder_dense(decoder_outputs)
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(loss='mse', optimizer='adam')

    def train(self, encoder_input, decoder_input, decoder_output):
        self.model.fit([encoder_input, decoder_input], decoder_output, verbose=2)
    
    def predict(self, encoder_input, decoder_input):
        return self.model.predict([encoder_input, decoder_input])

    def test(self, encoder_input, decoder_input, decoder_output):
        score = self.model.evaluate([encoder_input, decoder_input], decoder_output, verbose=1)
        return (self.model.metrics_names, score)

# np.random.seed(123)

# encoder_input = np.random.rand(2, 3, 1)
# decoder_input = encoder_input[:,-1,:].reshape(-1, 1, 1)
# decoder_output = np.random.rand(2, 1, 1)

# print(encoder_input)
# print()
# print(decoder_input)
# print()
# print(decoder_output)

# model = GruEncoderDecoder(3, 64)

# model.train(encoder_input, decoder_input, decoder_output)

# print(model.predict(encoder_input, decoder_input))