from keras.layers import Input, Dense
from keras.models import Sequential
from keras import losses
from keras.utils import plot_model
from keras.models import load_model
from math import log10
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../static_data_env/')
from supervised_cache_env import SupervisedCacheEnv
import csv

f = open('dlcpp-online-500k.csv', 'w')
w = csv.writer(f)

class DlcppTrainer:
    def __init__(self, num_of_contents):
        self.NUM_CONTENT_TYPES = num_of_contents
        self.pop_levels = 20
        self.current_ranking = defaultdict(int)
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(self.NUM_CONTENT_TYPES, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      
    def get_entropy_csv(self,req_prob):
        request_entropy_array = []
        for content_probability in req_prob:
            request_entropy_array.append(content_probability * log10(content_probability) if content_probability != 0 else 0)
        request_entropy = (-1) * sum(request_entropy_array)
        return request_entropy

    def extract_features_csv(self, req_prob, reqs_per_row):
        num_requests = reqs_per_row
        content_sum = len(req_prob)
        request_entropy = self.get_entropy_csv(req_prob)
        input_features = []
        i = 0
        for content_type in range(self.NUM_CONTENT_TYPES):
            input_features.append([int(req_prob[i]*reqs_per_row), num_requests, content_sum, request_entropy])
            i +=1
        return np.array(input_features)

    def get_true_labels_csv(self, req_prob, reqs_per_row):
        probs = list(enumerate(req_prob)) # (1,0.9)
        probs.sort(key = lambda x: x[1], reverse = True) # [(4, 0.6), (12, 0.3)
        true_labels = np.zeros((25, 25))
        for i in range(len(true_labels)):
            true_labels[probs[i][0]][i] = 1
        return true_labels

    def plot_metrics(self,loss,accuracy):
        plt.plot(accuracy,np.arange(1,accuracy.len))
        plt.ylabel('accuracy')
        plt.xlabel('batch')
        plt.show()
        plt.plot(loss,np.arange(1,loss.len))
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.show()

PATH = '/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_data_env/req_hist_100_million.csv'

np.random.seed(123)

dlcpp = DlcppTrainer(25)

env = SupervisedCacheEnv(PATH, start=0, end=500000, window=1, num_preds=1)

hit_ratios = []

trainX, trainY = env.get_next_data()

while trainX is not None and trainY is not None:
    x = trainX.reshape(25)
    y = trainY.reshape(25)
    input_features = dlcpp.extract_features_csv(x, 100)
    true_labels = dlcpp.get_true_labels_csv(y, 100)
    preds = dlcpp.model.predict(input_features)
    preds = np.argmax(preds, axis=1)
    preds = list(enumerate(preds))
    preds.sort(key = lambda x: x[1])
    cache = [i for i, j in preds[:5]]
    hit_ratio = sum(y[cache])
    hit_ratios.append(hit_ratio)
    dlcpp.model.fit(input_features, true_labels, verbose=1)
    trainX, trainY = env.get_next_data()

w.writerow(hit_ratios)