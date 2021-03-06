from collections import OrderedDict
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import heapq


class ContentStore:
    def __init__(self, size):
        self.size = size
        self.hits = 0
        self.misses = 0

    def add(self, item):
        pass

    def get_helper(self, item):
        # function is overriden for custom get implementations
        # in derived classes
        pass

    def update_stats(self, day, item):
        pass

    def get(self, item):
        item = self.get_helper(item)
        if item != None:
            self.hits += 1
        else:
            self.misses += 1
        return item


class LruContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)
        self.store = OrderedDict()

    def add(self, item):
        # returns can cache ?, victim
        victim = None
        if item in self.store or self.size <= 0:
            return False, victim
        if self.size:
            if(len(self.store) == self.size):
                # remove top element (least recent element)
                victim, _ = self.store.popitem(last=False)
            self.store[item] = item
            return True, victim

    def get_helper(self, item):
        try:
            cached_item = self.store.pop(item)
            # re-insert content to maintain lru-order
            self.store[item] = cached_item
            return cached_item
        except:
            return None


class PretrainedCNNContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)

        # bootstrapping
        self.bootstrap = LruContentStore(size)
        self.bootstrap_period = 7
        self.bootstrapping = True

        # constants
        self.pred_length = 7
        self.window_length = 7
        self.num_features = 11
        self.num_portals = 10

        # counter
        self.day = 0
        self.rank_update_day = self.window_length
        self.heap_update_day = self.window_length

        # ml-related
        self.model = load_model('../trained_models/opt_reshaped_simple_conv_with_portals.h5')
        self.scaler = MinMaxScaler()

        # stats
        self.history = OrderedDict()

        # ranks
        self.ranking = OrderedDict()

        # cache
        self.store = OrderedDict()

        # O(1) extraction of item with lowest rank (cache priority) 
        self.heap = []

        # data
        self.portals = np.load('../portals_dict.npy').item()

    def get_portal_key(self, item):
        # extract index from item name ex: from serverA/12 to 12
        if isinstance(item, str):
            start_ix = item.rfind('/') + 1
            return int(item[start_ix:])
        else:
            return item
    
    # add item to cache if more popular
    def add(self, item):
        victim = None
        if self.bootstrapping:
            return self.bootstrap.add(item)
        else:
            if item in self.store or self.size <= 0:
                return False, victim
            if self.size:
                try:
                    rank = self.ranking[item][self.day % self.pred_length]
                except KeyError:
                    print(f'KeyError for item {item}')
                    return False, None
                else:
                    if len(self.store) == self.size:
                        min_rank, min_item = self.get_least_popular()
                        if min_rank < rank:
                            victim = min_item
                            self.store.pop(victim)
                            self.store[item] = item
                            # add (curr_rank, item) into heap and remove lower priority item
                            heapq.heapreplace(self.heap, (rank, item))
                            return True, victim
                        else:
                            return False, victim
                    else:
                        self.store[item] = item
                        # add (curr_rank, item) into heap
                        heapq.heappush(self.heap, (rank, item))
                        return True, victim

    def get_least_popular(self):
        min_tuple = self.heap[0]
        min_rank = min_tuple[0]
        min_item = min_tuple[1]
        return min_rank, min_item

    def update_rankings(self):
        self.bootstrapping = False
        
        # init npy arrays
        agg_data = np.zeros((len(self.history.keys()), self.window_length))
        portal_data = np.zeros((len(self.history.keys()), self.window_length, self.num_portals))
                
        for ix, key in enumerate(self.history.keys()):
            agg_data[ix] = self.history[key]
            portal_key = self.get_portal_key(key)
            if portal_key in self.portals:
                portal_encoding = np.tile(self.portals[portal_key], (self.window_length, 1))
            else:
                portal_encoding = np.tile(np.zeros((self.num_portals)), (self.window_length, 1))
            portal_data[ix] = portal_encoding
             
        # log and normalize data
        agg_data = np.log1p(agg_data)
        agg_data = self.scaler.fit_transform(agg_data)
        
        agg_data = agg_data.reshape(-1, self.window_length, 1)

        # add portal features
        agg_data = np.concatenate((agg_data, portal_data), axis=2)
        
        # make preds
        predictions = self.predict_sequence(agg_data)  # update rankings                
        rankings = predictions[:, :, 0]
        
        # reset old ranking
        self.ranking = OrderedDict()

        # map preds to content types
        for i, key in enumerate(self.history.keys()):
            self.ranking[key] = rankings[i]
            
        # reset stats
        for key in self.history.keys():
            self.history[key] = np.zeros((self.window_length))

    def update_rankings_wrapper(self):
        # if first update, copy over cache from bootstrap
        if self.rank_update_day == self.window_length:
            self.store = self.bootstrap.store
        if self.size > 0:
            self.update_rankings()
        self.rank_update_day += self.pred_length
    
    def update_stats(self, day, item):
        self.day = day


        # reset heap with new rankings - happens daily
        if self.day == self.heap_update_day:
            self.heap = []
            for item in self.store.keys():
                heapq.heappush(self.heap, (self.ranking[item][self.day % self.pred_length], item))
            self.heap_update_day += 1

        if item not in self.history:
            self.history[item] = np.zeros(self.window_length)
        if item not in self.ranking:
            self.ranking[item] = np.zeros(self.pred_length)
        # update history
        self.history[item][self.day % self.window_length] += 1

    def get_helper(self, item):
        try:
            if self.bootstrapping:
                cached_item = self.bootstrap.get(item)
            else:
                cached_item = self.store[item]
            return cached_item
        except:
            return None

    def predict_sequence(self, input_sequence):
        history_sequence = input_sequence.copy()
        # initialize output (pred_steps time steps)
        pred_sequence = np.zeros((input_sequence.shape[0], self.pred_length, self.num_features))
        for i in range(self.pred_length):
            # record next time step prediction (last time step of model output)
            last_step_pred = self.model.predict(history_sequence)[:, -1, :self.num_features]
            pred_sequence[:, i, :self.num_features] = last_step_pred
            # add the next time step prediction to the history sequence
            history_sequence = np.concatenate([history_sequence, last_step_pred.reshape(-1, 1, self.num_features)], axis=1)
        return pred_sequence