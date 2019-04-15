from collections import OrderedDict
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class ContentStore:
    def __init__(self, size):
        self.size = size
        self.hits = 0
        self.misses = 0

    def add(self, item):
        pass

    def get_helper(self, item):
        pass

    def get(self, item):
        item = self.get_helper(item)
        if item != None:
            self.hits += 1
        else:
            self.misses += 1
        return item

    def has(self, item):
        if item in self.store:
            return self.store[item]
        else:
            return None


class LruContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)
        self.store = OrderedDict()

    def add(self, item):
        victim = None
        if item in self.store:
            return False, victim
        if self.size:
            if(len(self.store) == self.size):
                victim, _ = self.store.popitem(last=False)
            self.store[item] = item
        return True, victim

    def get_helper(self, item):
        try:
            cached_item = self.store.pop(item)
            self.store[item] = cached_item
            return cached_item
        except:
            return None


class PretrainedCNNContentStore(ContentStore):
    def __init__(self, size, online=False):
        super().__init__(size)

        # bootstrapping
        self.bootstrap = LruContentStore(size)
        self.bootstrap_period = 7
        self.bootstrapping = True

        # constants
        self.num_features = 1
        self.pred_length = 7
        self.window_length = 7
        self.num_features = 11
        self.reqs_per_day = 50000

        # counter
        self.timestep = 0
        self.req_counter = 0

        # ml-related
        self.model = load_model(
            '../trained_models/improved_simple_conv_with_portals.h5')
        self.scaler = MinMaxScaler()

        # stats
        self.history = OrderedDict()

        # ranks
        self.ranking = OrderedDict()

        # cache
        self.store = {}

        # data
        portals_arr = np.load('../portals.npy')
        self.portals = {}
        for i in range(portals_arr.shape[0]):
            self.portlas[str(i)] = portals_arr[i]

    # add item to cache if more popular
    def add(self, item):
        victim = None
        if self.bootstrapping:
            return self.bootstrap.add(item)
        else:
            if item in self.store:
                return False, victim
            if self.size:
                if len(self.store) == self.size:
                    min_key, min_rank = self.get_least_popular()
                    timestep = self.req_counter // self.reqs_per_day % self.pred_length
                    # replace if curr item more popular than least popular in cache
                    if min_rank != None and min_rank < self.ranking[item][timestep]:
                        victim = min_key
                        self.store.pop(victim)
                        self.store[item] = item
                        return True, victim
                    return False, victim
                else:
                    self.store[item] = item
                    return True, victim

    def get_least_popular(self):
        min_item = None
        min_rank = None
        for item in self.store.keys():
            timestep = self.req_counter // self.reqs_per_day % self.pred_length
            if min_item == None or self.ranking[item][timestep] < min_rank:
                min_rank = self.ranking[item]
                min_item = item
        return min_item, min_rank

    def update_rankings(self):
        self.timestep += 1
        if not self.bootstrapping:
            # init npy arrays
            agg_data = np.zeros((len(self.history.keys()), self.window_length))
            portal_data = np.zeros((last_week.shape[0], self.window_length, 10))

            # reshape data
            for i, item in enumerate(self.history.keys()):
                agg_data[i] = self.history[item]
                if item in self.portals:
                    portal_data[item] = np.tile(self.portals[item], (self.window_length, 1))
                else:
                    portal_data[item] = np.tile(np.zeros(10), (self.window_length, 1))

            # log and normalize data
            agg_data = np.log1p(agg_data)
            agg_data = self.scaler.fit_transform(agg_data)
            agg_data = agg_data.reshape(-1, 7, 1)

            # add portal features
            agg_data = np.concatenate((agg_data, portal_data), axis=2)

            # make preds
            rankings = predict_sequence(self.history)[
                :, :, 0]  # update rankings

            # map preds to content types
            for i, key in enumerate(self.history.keys()):
                self.ranking[key] = rankings[i]
            
            # reset stats
            for key in self.history.keys():
                self.history[key] = np.concatenate((self.history[key][self.pred_length:], np.zeros(self.pred_length)))

        if self.timestep == self.bootstrap_period:  # end bootstrap ?
            self.bootstrapping = False

    def get_helper(self, item):
        try:
            if item not in self.history:
                self.history[item] = np.zeros(self.window_length)
            if item not in self.ranking:
                self.ranking[item] = np.zeros(self.pred_length)
            timestep = self.req_counter // self.reqs_per_day % self.window_length
            self.history[item][timestep] += 1
            if self.bootstrapping:
                cached_item = self.bootstrap.get(item)
            else:
                cached_item = self.store[item]
            return cached_item
        except:
            return None

    def predict_sequence(input_sequence):
        history_sequence = input_sequence.copy()
        # initialize output (pred_steps time steps)
        pred_sequence = np.zeros(
            (input_sequence.shape[0], self.pred_length, self.num_features))
        for i in range(self.pred_length):
            # record next time step prediction (last time step of model output)
            last_step_pred = self.model.predict(history_sequence)[
                :, -1, :self.num_features]
            pred_sequence[:, i, :self.num_features] = last_step_pred
            # add the next time step prediction to the history sequence
            history_sequence = np.concatenate(
                [history_sequence, last_step_pred.reshape(-1, 1, self.num_features)], axis=1)
        return pred_sequence
