import pandas as pd
import numpy as np

class SupervisedCacheEnv:

    def __init__(self, path, rows, window):
        df = pd.read_csv(path, sep=',', skiprows=1, nrows = rows, header=None)
        self.arr = df.T.values.astype('float32')
        self.start = 0
        self.window = window
        self.end = window

    def get_next_data(self):
        if self.end >= self.arr.shape[1]:
            return None, None
        trainX = self.arr[:, self.start:self.end].reshape(-1, self.window, 1)
        trainY = self.arr[:,self.end:self.end+1].reshape(-1, 1, 1)
        self.start += 1
        self.end += 1
        return trainX, trainY
