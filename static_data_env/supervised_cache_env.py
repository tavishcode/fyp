import pandas as pd
import numpy as np

class SupervisedCacheEnv:
    def __init__(self, path, start, end, window, num_preds):
        df = pd.read_csv(path, sep=',', skiprows=start+1, nrows=end, header=None)
        self.arr = df.T.values.astype('float32')
        self.start = 0
        self.window = window
        self.num_preds = num_preds
        self.end = window

    def get_next_data(self):
        if self.end + self.num_preds > self.arr.shape[1]:
            return None, None
        trainX = self.arr[:, self.start:self.end].reshape(-1, self.window, 1)
        trainY = self.arr[:,self.end:self.end+self.num_preds].reshape(-1, self.num_preds, 1)
        self.start += self.num_preds
        self.end += self.num_preds
        return trainX, trainY
