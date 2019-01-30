import csv
import pandas as pd
import random

class TraditionalCacheEnv:
    def __init__(self, timesteps, seed):
        self.df = pd.read_csv('/home/tavish/Desktop/MEGASync/HKUST/FYP/fyp/static_data_env/req_hist_100_million.csv', sep=',', nrows=timesteps)
        self.cols = self.df.columns.tolist()
        random.seed(seed)

    def get_requests(self, ix):
        arr = self.df.iloc[ix].values
        reqs = []
        for col, val in zip(self.cols, arr):
            reqs.extend([col for i in range(int(val * 100))])
        random.shuffle(reqs)
        return reqs
