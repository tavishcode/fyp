import sys
sys.path.insert(0, '../static_data_env/')
sys.path.insert(0, '../gru/')
from supervised_cache_env import SupervisedCacheEnv
from gru import GruEncoderDecoder
import csv
import numpy as np


f = open('gru-online-500k.csv', 'w')
w = csv.writer(f)

PATH = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_data_env/req_hist_100_million.csv'
WINDOW = 3
HIDDEN_UNITS = 64
SEED = 123

np.random.seed(SEED)

env = SupervisedCacheEnv(PATH, rows=500000, window=WINDOW)

model = GruEncoderDecoder(WINDOW, HIDDEN_UNITS)

trainX, trainY = env.get_next_data()

hit_ratios = []

while trainX is not None and trainY is not None:
    encoder_input = trainX
    decoder_input = trainX[:, -1].reshape(-1, 1, 1)
    decoder_output = trainY

    preds = model.predict(encoder_input, decoder_input).flatten()
    preds = list(enumerate(preds))
    preds.sort(key = lambda x: x[1], reverse = True)
    cache = preds[:5]
    cache_ixs = [i for i, j in cache]

    hit_ratio = 0
    true_labels = trainY.flatten()
    for i in cache_ixs:
        hit_ratio += true_labels[i]

    hit_ratios.append(hit_ratio)

    model.train(encoder_input, decoder_input, decoder_output)

    trainX, trainY = env.get_next_data()

w.writerow(hit_ratios)