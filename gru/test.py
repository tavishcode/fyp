from keras.models import Sequential, Model
from keras.layers import Input, GRU, LSTM, Dense, RepeatVector
import numpy as np

np.random.seed(123)

encoder_input = np.random.rand(2, 3, 1)
decoder_input = encoder_input[:,-1,:].reshape(-1, 1, 1)
decoder_output = np.random.rand(2, 1, 1)

timesteps = 3
hidden_units = 64
num_features = 1
pred_steps = 1
encoder_inputs = Input(shape=(timesteps, num_features))
encoder = GRU(hidden_units, return_state=True)
encoder_outputs, encoder_h = encoder(encoder_inputs)
decoder_inputs = Input(shape=(None, num_features))
decoder = GRU(hidden_units, return_state=True, return_sequences=True)
decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_h)
decoder_dense = Dense(1, activation='sigmoid')
decoder_outputs = decoder_dense(decoder_outputs)

# training architecture
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(loss='mse', optimizer='adam')


# inference architecture
inference_encoder = Model(encoder_inputs, encoder_h)
decoder_input_h = Input(shape=(hidden_units,))
decoder_outputs, decoder_output_h = decoder(decoder_inputs, initial_state=decoder_input_h)
decoder_outputs = decoder_dense(decoder_outputs)
inference_decoder = Model([decoder_inputs, decoder_input_h], [decoder_outputs, decoder_output_h])


model.fit([encoder_input, decoder_input], decoder_output, verbose=2)



input_seq = encoder_input

decoder_input_h = inference_encoder.predict(input_seq)
target_seq = input_seq[:, -1, :].reshape(-1, 1, 1)
num_samples = input_seq.shape[0]
decoded_seq = np.zeros((num_samples, pred_steps, 1))
for i in range(pred_steps):
    decoder_output, decoder_input_h = inference_decoder.predict([target_seq, decoder_input_h])
    for j in range(num_samples): 
        decoded_seq[j, i, 0] = decoder_output[j, 0, 0]
    target_seq = decoder_output

print(decoded_seq)