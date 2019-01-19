from pandas import read_csv, DataFrame #To read CSV data for training
from keras.layers import Input, Dense
from keras.models import Model

df = read_csv("pre-train.csv")

encoding_dim = 32