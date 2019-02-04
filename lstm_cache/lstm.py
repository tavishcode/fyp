from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, RepeatVector
import numpy as np
class LstmTrainer:
    def __init__(self, timesteps, num_content_types):
        self.timesteps = timesteps
        self.num_content_types = num_content_types
        self.features = 1
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.timesteps, self.features)))
        model.add(RepeatVector(1))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer='adam', loss='mse')
        self.model = model 
    
    def train(self, data, labels):
        self.model.fit(data, labels, verbose=0)
    def test(self, data):
        pred = self.model.predict(data)
        return pred