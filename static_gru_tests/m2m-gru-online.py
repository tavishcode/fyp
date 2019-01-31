import sys
sys.path.insert(0, '../static_data_env/')
sys.path.insert(0, '../gru/')
from supervised_cache_env import SupervisedCacheEnv
from grum2m import GruEncoderDecoder
import csv
import numpy as np


f = open('m2m-gru-online-500k.csv', 'w')
w = csv.writer(f)

PATH = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_data_env/req_hist_100_million.csv'
WINDOW = 3
HIDDEN_UNITS = 64
SEED = 123
NUM_PREDS = 3

np.random.seed(SEED)

env = SupervisedCacheEnv(PATH, start=0, end=500000, window=WINDOW, num_preds=NUM_PREDS)

model = GruEncoderDecoder(WINDOW, HIDDEN_UNITS, NUM_PREDS)

trainX, trainY = env.get_next_data()

hit_ratios = []

while trainX is not None and trainY is not None:
    encoder_input = trainX
    decoder_output = trainY
    preds = model.predict(encoder_input).reshape(-1, NUM_PREDS)
    preds = np.mean(preds, axis=1)
    preds = list(enumerate(preds))
    preds.sort(key = lambda x: x[1], reverse = True)
    cache = preds[:5]
    cache_ixs = [i for i, j in cache]
    hit_ratio = 0
    true_labels = trainY.reshape(-1, NUM_PREDS)
    true_labels = np.mean(true_labels, axis=1)
    for i in cache_ixs:
        hit_ratio += true_labels[i]
    hit_ratios.append(hit_ratio)
    decoder_input = np.zeros((encoder_input.shape[0], NUM_PREDS, 1))
    decoder_input[:,0,0] = encoder_input[:,-1,:].flatten()
    if NUM_PREDS > 1:
        decoder_input[:,1:,0] = trainY[:,:-1,0]
    model.train(encoder_input, decoder_input, decoder_output)
    trainX, trainY = env.get_next_data()

w.writerow(hit_ratios)