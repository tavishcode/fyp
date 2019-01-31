import sys
sys.path.insert(0, '../static_data_env/')
sys.path.insert(0, '../gru/')
from supervised_cache_env import SupervisedCacheEnv
from grum2m import GruEncoderDecoder
import csv
import numpy as np


if __name__ == "__main__":

    f = open('m2m-gru-backtest-500k-1mil.csv', 'w')
    w = csv.writer(f)

    PATH = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_data_env/req_hist_100_million.csv'
    WINDOW = 3
    HIDDEN_UNITS = 64
    SEED = 123
    NUM_PREDS = 50000

    np.random.seed(SEED)

    model = GruEncoderDecoder(WINDOW, HIDDEN_UNITS, NUM_PREDS) # num preds here refer to the testing scheme

    # training period
    train_env = SupervisedCacheEnv(PATH, start=0, end=50000, window=WINDOW, num_preds=1)
    trainX, trainY = train_env.get_next_data()
    while trainX is not None and trainY is not None:
        encoder_input = trainX
        decoder_output = trainY
        decoder_input = np.zeros((encoder_input.shape[0], 1, 1))
        decoder_input[:,0,0] = encoder_input[:,-1,:].flatten()
        if NUM_PREDS > 1:
            decoder_input[:,1:,0] = trainY[:,:-1,0]
        model.train(encoder_input, decoder_input, decoder_output)
        trainX, trainY = train_env.get_next_data()

    # testing period
    hit_ratios = []
    test_env = SupervisedCacheEnv(PATH, start=50000, end=100000, window=WINDOW, num_preds=NUM_PREDS)
    testX, testY = test_env.get_next_data()
    if testX is not None and testY is not None:
        encoder_input = testX
        decoder_output = testY
        preds = model.predict(encoder_input).reshape(encoder_input.shape[0], NUM_PREDS).T
        print('finished predictions')
        true_labels = testY.reshape(encoder_input.shape[0], NUM_PREDS).T
        assert(len(preds) == len(true_labels))
        for i in range(preds.shape[0]):
            pred = list(enumerate(preds[i]))
            pred.sort(key = lambda x: x[1], reverse = True)
            cache = pred[:5]
            cache_ixs = [i for i, j in cache]
            hit_ratio = 0
            for j in cache_ixs:
                hit_ratio += true_labels[i][j]
            hit_ratios.append(hit_ratio)

    w.writerow(hit_ratios)
