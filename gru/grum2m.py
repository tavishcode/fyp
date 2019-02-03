from keras.models import Sequential, Model
from keras.layers import Input, GRU, LSTM, Dense, RepeatVector
import numpy as np

# GRU Encoder-Decoder (Many to Many)

class GruEncoderDecoder:
    def __init__(self, timesteps, hidden_units, pred_steps):
        self.timesteps = timesteps
        self.hidden_units = hidden_units
        self.num_features = 1
        self.pred_steps = pred_steps
        encoder_inputs = Input(shape=(self.timesteps, self.num_features))
        self.encoder = GRU(self.hidden_units, return_state=True)
        encoder_outputs, encoder_h = self.encoder(encoder_inputs)
        decoder_inputs = Input(shape=(None, self.num_features))
        self.decoder = GRU(self.hidden_units, return_state=True, return_sequences=True)
        decoder_outputs, _ = self.decoder(decoder_inputs, initial_state=encoder_h)
        self.decoder_dense = Dense(1, activation='sigmoid')
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # training architecture
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(loss='mse', optimizer='adam')

        # inference architecture
        self.inference_encoder = Model(encoder_inputs, encoder_h)
        decoder_input_h = Input(shape=(self.hidden_units,))
        decoder_outputs, decoder_output_h = self.decoder(decoder_inputs, initial_state=decoder_input_h)
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.inference_decoder = Model([decoder_inputs, decoder_input_h], [decoder_outputs, decoder_output_h])

    def predict(self, encoder_input):
        decoder_input_h = self.inference_encoder.predict(encoder_input)
        target_seq = encoder_input[:, -1, :].reshape(-1, 1, 1)
        num_samples = encoder_input.shape[0]
        decoded_seq = np.zeros((num_samples, self.pred_steps, 1))
        for i in range(self.pred_steps):
            decoder_output, decoder_input_h = self.inference_decoder.predict([target_seq, decoder_input_h])
            for j in range(num_samples): 
                decoded_seq[j, i, 0] = decoder_output[j, 0, 0]
            target_seq = decoder_output
        return decoded_seq

    def train(self, encoder_input, decoder_input, decoder_output):
        self.model.fit([encoder_input, decoder_input], decoder_output, verbose=0)

# np.random.seed(123)

# encoder_input = np.random.rand(2, 3, 1)
# decoder_input = encoder_input[:,-1,:].reshape(-1, 1, 1)
# decoder_output = np.random.rand(2, 1, 1)

# # print(encoder_input)
# # print()
# # print(decoder_input)
# # print()
# # print(decoder_output)

# model = GruEncoderDecoder(3, 64, 1)

# model.train(encoder_input, decoder_input, decoder_output)

# print(model.predict(encoder_input))