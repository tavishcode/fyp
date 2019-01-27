import pandas as pd
import numpy as np
import sys

sys.path.insert(0, './')
from lstm_cache.lstm import LstmTrainer
from dlcpp_cache.dlcpp_trainer import DlcppTrainer


class SupervisedPlayground:
    def __init__(self, batchwise=False):
        self.batchwise = batchwise
        self.data = np.array(pd.read_csv("./data/req_hist_100_million.csv"))
        self.samples = len(self.data)
        self.contenttypes = len(self.data[0])
        self.TRAIN_DELTA = 10 #10000 rows at a time
        self.trainer = DlcppTrainer(self.contenttypes)
        self.reqs_per_row = self.TRAIN_DELTA*100

    def reshape_to_deepcache(self, timesteps):
        self.data = np.reshape(np.ravel(self.data), [int(self.samples/(timesteps+1)), timesteps+1, self.contenttypes])
        self.data = np.swapaxes(np.swapaxes(self.data, 1,2),0,1)

        X =  np.reshape(np.ravel(self.data), [self.samples*self.contenttypes, timesteps]).transpose()[0:4].transpose()
        X =  np.reshape(np.ravel(self.data), [self.samples*self.contenttypes, timesteps]).transpose()[5].transpose()
        
        return X, y

    def dlcpp_train(self):
        batches = np.array_split(self.data,self.TRAIN_DELTA,axis=0)
        curr_batch = False
        first_time = True
        for batch in batches:
            prev_batch = False if first_time else curr_batch
            curr_batch = np.mean(batch,axis=0)
            if not first_time:
                # print("curr",curr_batch,"prev",prev_batch)
                self.trainer.train_from_csv(prev_batch,curr_batch,self.reqs_per_row)
            first_time = False
        score = self.trainer.evaluate_csv(curr_batch,prev_batch,self.reqs_per_row)
        if score[1] > 0.90:
            print("score",score)
            self.trainer.report()
        else:
            print("Low accuracy",score)
        


    def simulate(self):
        if self.trainer.name == "lstm":
            X, y = self.reshape_to_deepcache(trainer.timesteps)
        elif self.trainer.name == "dlcpp":
            self.dlcpp_train()

        if not self.batchwise:
            trainer.train(X[0: 600000], y[0: 600000])
            trainer.test(X[600000: 1000000], y [600000: 1000000])
        else:
            #TODO: Train b
            pass
        

if __name__ == "__main__":
    train = SupervisedPlayground(batchwise=True)
    train.simulate()
                
