import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '../lstm_cache')
from lstm import LstmTrainer
sys.path.insert(0, '../dlcpp_cache')
from dlcpp_trainer import DlcppTrainer


class SupervisedPlayground:
    def __init__(self, batchwise=False):
        self.batchwise = batchwise
        self.data = np.array(pd.read_csv("../data/req_hist_100_million.csv"))
        self.samples = len(data)
        self.contenttypes = len(data.columns)
        self.TRAIN_DELTA = 100 #100 rows at a time
        self.trainer = DlcppTrainer
        self.reqs_per_row = self.TRAIN_DELTA*100

    def reshape_to_deepcache(self, timesteps):
        self.data = np.reshape(np.ravel(self.data), [int(self.samples/(timesteps+1)), timesteps+1, self.contenttypes])
        self.data = np.swapaxes(np.swapaxes(self.data, 1,2),0,1)

        X =  np.reshape(np.ravel(self.data), [self.samples*self.contenttypes, timesteps]).transpose()[0:4].transpose()
        X =  np.reshape(np.ravel(self.data), [self.samples*self.contenttypes, timesteps]).transpose()[5].transpose()
        
        return X, y

    def dlcpp_train(self):
        print(self.data[0], self.data[1])
        batches = np.array_split(self.data,self.TRAIN_DELTA,axis=0)
        curr_batch = False
        for batch in batches:
            prev_batch = curr_batch if curr_batch else False
            curr_batch = np.mean(batch,axis=0)
            if curr_batch and prev_batch:
                self.trainer.train_from_csv(prev_batch,curr_batch,self.reqs_per_row)


    def simulate(self, trainer):
        if trainer.name == "lstm":
            X, y = self.reshape_to_deepcache(trainer.timesteps)

        if not self.batchwise:
            trainer.train(X[0: 600000], y[0: 600000])
            trainer.test(X[600000: 1000000], y [600000: 1000000])
        else:
            #TODO: Train b
            pass
        

    d
                
