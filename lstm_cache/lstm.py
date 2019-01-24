from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, RepeatVector
import numpy as np
from keras.callbacks import ModelCheckpoint

class LstmTrainer:
    def __init__(self, minibatch_size, timesteps, num_content_types):
        self.name = "lstm"
        self.timesteps = timesteps
        self.NUM_CONTENT_TYPES = num_content_types
        self.features = 1
        self.trained = False
        #Load previous data
        
#TODO parse in CSV
# Save models, run on batches
# Visualize actual vs predicted reqquests
# Optimize HyperParameters, (crossvalidate)
# Play with learning rate
    def reshape(self, data, labels):
        data = np.ravel(data).reshape((self.NUM_CONTENT_TYPES * data.shape[0], self.timesteps, self.features))
        labels = np.ravel(labels).reshape((self.NUM_CONTENT_TYPES*data.shape[0], 1, 1))

        return data, labels
    def train(self, data, labels):
        #reshape data to fit input
        data, labels = self.reshape(data, labels)
        print("Reshaped Data")
        print(data)
        print("Reshaped Labels")
        print(labels)

        checkpoint = ModelCheckpoint("model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.timesteps, self.features)))
        model.add(RepeatVector(1)) # Reshapes output so it is accessible by next layer
        model.add(LSTM(64, return_sequences=True))
        model.add(Dense(1, activation="sigmoid"))

        if self.trained:
            model.load_weights("model.hdf5")

        model.compile(optimizer='adam', loss='mse')
        model.fit(data, labels, verbose=2, callbacks=callbacks_list)

        model.summary()
        self.trained = True
    
    def test(self, data, labels):
        data, labels = self.reshape(data, labels)

        model.load_weights("model.hdf5")

        scores = model.evaluate(data, labels, verbose=0)
	    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        
        





       
if __name__ == "__main__":
    trainer = LSTMTrainer(2, 3, 4)
    data  = np.random.random((2, 4, 3))
    labels = np.random.random(2*4)

    print("DATA")
    print(data)
    print("Labels")
    print(labels)

    trainer.train(data, labels)
    


    