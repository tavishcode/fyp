from pandas import read_csv, DataFrame #To read CSV data for training
from keras.layers import Input, Dense
from keras.models import Model

df = read_csv("pre-train.csv")

encoding_dim = 3 #4 to 3 encoding
my_epochs = 2 #number of passes


# this is the input matrix per node
input_features = Input(shape=(4,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_features)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(4, activation='relu')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_features, decoded)
encoder = Model(input_features, encoded)

# placeholder for the encoded data
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(loss='mean_squared_error', optimizer='sgd')




def run(routers,timestamp):
    for router in routers:
        
