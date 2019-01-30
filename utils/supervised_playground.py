import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '../lstm_cache')

from lstm import LstmTrainer


class SupervisedPlayground:
    def __init__(self, batchwise=False):
        self.batchwise = batchwise
        self.data = np.array(pd.read_csv("../data/req_hist_100_million.csv"))
        self.samples = len(data)
        self.contenttypes = len(data.columns) 

    def reshape_to_deepcache(self, timesteps):
        self.data = np.reshape(np.ravel(self.data), [int(self.samples/(timesteps+1)), timesteps+1, self.contenttypes])
        self.data = np.swapaxes(np.swapaxes(self.data, 1,2),0,1)

        X =  np.reshape(np.ravel(self.data), [self.samples*self.contenttypes, timesteps]).transpose()[0:4].transpose()
        X =  np.reshape(np.ravel(self.data), [self.samples*self.contenttypes, timesteps]).transpose()[5].transpose()
        
        return X, y

    def reshape_to_dlcpp(self):
        raise NotImplementedError('Need to Implement input for dlcpp')

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
                
