from packet import Packet
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import csv
sys.path.insert(0, './ddpg_cache')

f = open('req_hist.csv', 'w')
writer = csv.writer(f)

"""DDPG-RL Imports"""

from ddpg_cache_train import Trainer as ddpg_trainer
from ddpg_cache_buffer import MemoryBuffer

"""---------------"""


from lstm import LSTMTrainer

sys.path.insert(0, '../dlcpp_cache')
from dlcpp_trainer import DlcppTrainer

""" Abstract Class for defining Cache Policies.

    Attributes: 
        size: no of items that can be stored in ContentStore.
"""
class ContentStore:
    def __init__(self, size):
        self.size = size
        self.hits = 0
        self.misses = 0
    
    def add(self, item):        
        """Decides whether to add item to store and what to evict if store is full"""
        raise NotImplementedError('Base Class ContentStore does not implement a cache')

    def get_helper(self, item_name):
        """Returns item with item_name if it is in store, else returns None"""
        raise NotImplementedError('Base Class ContentStore does not implement a cache')

    def update_state(self):
        pass

    def get(self, item_name):
        """Wrapper function for get_helper which includes hit/miss statistic updates"""
        item = self.get_helper(item_name)
        if item != None:
            self.hits += 1
        else:
            self.misses += 1
        return item

"""First in First Out Cache Policy"""
class FifoContentStore(ContentStore):
    def __init__(self, size, num_content_types):
        super().__init__(size)
        self.num_content_types = num_content_types
        self.reset_req_hist()
        
        # making training data
        keys = []
        for key in self.req_hist.keys():
            keys.append(key)
        writer.writerow(keys)
        #

        self.store = OrderedDict()

    def reset_req_hist(self):
        content_types = ['content' + str(i) for i in range(self.num_content_types)]
        counts = [0 for content in range(self.num_content_types)]
        self.req_hist = OrderedDict(zip(content_types, counts))

    def add(self, item):
        if self.size:
            if(len(self.store) == self.size):
                self.store.popitem(last=False)
            self.store[item.name] = item

    def update_state(self):
        # making training data
        values = []
        for value in self.req_hist.values():
            values.append(value)
        writer.writerow(values)
        #

        self.reset_req_hist()

    def get_helper(self, item):
        try:
            self.req_hist[item.name] += 1
            return self.store[item.name]
        except:
            return None

"""Least Recently Used Cache Policy"""
class LruContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)
        self.store = OrderedDict()

    def add(self, item):
        if self.size:
            if(len(self.store) == self.size):
                self.store.popitem(last=False)
            self.store[item.name] = item

    def get_helper(self, item):
        try:
            cached_item = self.store.pop(item.name)
            self.store[item.name] = cached_item
            return cached_item
        except:
            return None

    def get(self, item_name):
        item = self.get_helper(item_name)
        if item != None:
            self.hits += 1
        else:
            self.misses += 1

"""Least Frequently Used Cache Policy"""
class LfuContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)
        self.store = {} # {'name', [item, freq]}
    
    def add(self, item):
        if self.size:
            if(len(self.store) == self.size):
                min_key = None
                min_freq = None
                for key in self.store.keys():
                    if min_freq == None or self.store[key][1] < min_freq:
                        min_freq = self.store[key][1]
                        min_key = key
                self.store.pop(min_key)
            self.store[item.name] = [item, 1]

    def get_helper(self, item):
        try:
            cached_item = self.store[item.name][0]
            self.store[item.name][1] += 1
            return cached_item
        except:
            return None

"""DLCPP Cache Policy"""
class DlcppContentStore(ContentStore):
    def __init__(self, size, num_content_types):
        super().__init__(size)

        self.store = OrderedDict() 
        self.prev_reqs = defaultdict(int)
        self.curr_reqs = defaultdict(int)
        self.NUM_CONTENT_TYPES = num_content_types
        self.num_updates = 0
        self.BOOTSTRAPPING = True
        self.bootstrap = LfuContentStore(size)
        # self.trainer = dlcpp_trainer(self.NUM_CONTENT_TYPES)
        self.pop_rank

    def get(self, item_name):
        """Wrapper function for get_helper which includes hit/miss statistic updates"""
        item = self.get_helper(item_name)
        if item != None and not self.BOOTSTRAPPING:
            self.hits += 1
        elif not self.BOOTSTRAPPING:
            self.misses += 1
        else: # use bootstrap
            item = self.bootstrap.get(item_name)
            self.hits = self.bootstrap.hits
            self.misses = self.bootstrap.misses
        return item
    
    def get_helper(self, item):
        try:
            self.curr_reqs[item.name] += 1
            return self.store[item.name]
        except:
            return None
    
    def update_ranking(self):
        if hasattr(self.action, '__len__'):
            rating_table = []
            for ix in range(len(self.action)):
                rating_table.append(['content'+ str(ix), self.action[ix]])
            rating_table.sort(key = lambda x: x[1], reverse = True)
            self.store = {}
            for ix in range(self.size):
                self.store[rating_table[ix][0]] = Packet(rating_table[ix][0], is_interest=False)
    
    def update_state(self):
        """Called every CACHE_UPDATE_INTERVAL"""
        #TODO get latest popularity rankings
        self.prev_reqs = self.curr_reqs
        self.curr_reqs = defaultdict(int)
        self.get_latest_rankings(prev_reqs,curr_reqs)

    def get_latest_rankings(self,prev_reqs, curr_reqs):
        ranks = self.trainer.updated_popularity(prev_reqs,curr_reqs)


"""DDPG Cache Policy with Bootstrap"""
class DdpgContentStore(ContentStore):
    def __init__(self, size, num_content_types):
        super().__init__(size)

        # Bootstrap first updates
        self.bootstrap = LfuContentStore(size)
        self.bootstrap_period = 100000

        self.NUM_CONTENT_TYPES = num_content_types
        self.curr_reqs = defaultdict(int)
        self.ram = MemoryBuffer(1000000)
        self.trainer = ddpg_trainer(self.NUM_CONTENT_TYPES, self.NUM_CONTENT_TYPES, self.ram)
        self.first_run = True
        self.state = []
        self.action = None
        self.num_updates = 0
        self.model_updates = 0
        self.temp_hits = 0
        self.temp_misses = 0
        self.rewards = []

    def get_helper(self, item):
        try:
            self.curr_reqs[item.name] += 1
            cached_item = self.store[item.name]
            return cached_item
        except:
            return None
    
    def add(self, item):
        if self.num_updates < self.bootstrap_period:
            self.bootstrap.add(item) 
        else:
            pass

    def get(self, item_name):
        """Wrapper function for get_helper which includes hit/miss statistic updates"""
        item = self.get_helper(item_name)
        if item != None:
            if self.num_updates > self.bootstrap_period:
                self.hits += 1
            self.temp_hits += 1
        else:
            if self.num_updates > self.bootstrap_period:
                self.misses += 1
            self.temp_misses += 1
        if self.num_updates < self.bootstrap_period: # use bootsrap
            item = self.bootstrap.get(item_name)
            self.hits = self.bootstrap.hits
            self.misses = self.bootstrap.misses
        return item

    def update_ranking(self):
        if hasattr(self.action, '__len__'):
            rating_table = []
            for ix in range(len(self.action)):
                rating_table.append(['content'+ str(ix), self.action[ix]])
            rating_table.sort(key = lambda x: x[1], reverse = True)
            self.store = {}
            for ix in range(self.size):
                self.store[rating_table[ix][0]] = Packet(rating_table[ix][0], is_interest=False)

    def update_state(self):
        temp_reqs = self.temp_hits + self.temp_misses

        if temp_reqs and not self.first_run:

            reward = self.temp_hits/temp_reqs
            self.rewards.append(reward)
            print('reward ' + str(reward))
            new_state = []
            for c in range(self.NUM_CONTENT_TYPES):
                new_state.append(0)
            for key in self.curr_reqs.keys():
                ix = int(key[-1])
                new_state[ix] = self.curr_reqs[key]/temp_reqs

            # print('prev state', self.state)
            # print('new state', new_state)

            if self.num_updates < self.bootstrap_period:
                # pass
                self.ram.add(np.float32(self.state), self.action, reward, np.float32(new_state))
            
            self.state = new_state
            
            if self.num_updates < self.bootstrap_period:
                # pass
                self.trainer.optimize()
                self.action = self.trainer.get_exploration_action(np.float32(self.state))
                self.model_updates += 1
                if self.model_updates % 10000 == 0:
                    self.trainer.save_models(1)
                    print('saving model after ' + str(self.model_updates) + ' updates and ' + str(self.num_updates) + ' timesteps')
            else:
                self.action = self.trainer.get_exploitation_action(np.float32(self.state))
            # print('next action', self.action)
            self.update_ranking()
        elif temp_reqs and self.first_run:
            self.state = []
            for c in range(self.NUM_CONTENT_TYPES):
                self.state.append(0)
            for key in self.curr_reqs.keys():
                ix = int(key[-1])
                self.state[ix] = self.curr_reqs[key]/temp_reqs

            if self.num_updates < self.bootstrap_period:
                self.action = self.trainer.get_exploration_action(np.float32(self.state))
                # pass
            else:
                self.action = self.trainer.get_exploitation_action(np.float32(self.state))
            # print('first action', self.action)
            self.first_run = False
            self.update_ranking()

        self.temp_hits = 0
        self.temp_misses = 0
        self.curr_reqs = defaultdict(int)
        self.num_updates += 1
        if self.num_updates % 1000 == 0:
            print(self.num_updates)
        if self.num_updates == self.bootstrap_period:
            self.trainer.load_models(1)

"""Lstm Cache Policy"""
class LstmContentStore(ContentStore):
    super().__init__(size, num_content_types)
    self.store = OrderedDict() 
    self.NUM_CONTENT_TYPES = num_content_types
    self.timesteps = 5
    self.REQUESTS_IN_TIMESTEP = 20
    self.MINI_BATCH_SIZE = 4
    self.BOOTSTRAPPING = True
    self.bootstrap = LfuContentStore(size)
    
    self.data = np.zeros([self.MINI_BATCH_SIZE, self.NUM_CONTENT_TYPES, self.timesteps])
    self.labels = np.zeros([self.NUM_CONTENT_TYPES*self.MINBATCH_SIZE])

    self.curr_reqs = 0
    self.cur_timestep = 0
    self.cur_sample = 0
    
    self.temp_hits = 0
    self.temp_misses = 0

    def get_content_index(self, content):
        return int(str.split("content")[1])

    def increment(self):
        self.curr_reqs += 1
        if curr_reqs == self.REQUESTS_IN_TIMESTEP:
            self.curr_reqs = 0
            self.cur_timestep += 1
        if self.cur_timestep == self.timesteps:
            self.cur_timestep = 0
            self.cur_sample += 1
        if self.cur_sample == self.MINI_BATCH_SIZE:
            self.cur_sample = 0
            self.train()

    def record(self, content):
        idx = self.get_content_index(content)
        if self.cur_timestep == self.timesteps:
            self.labels[self.cur_batch*self.NUM_CONTENT_TYPES + idx] += 1

        self.data[self.cur_batch][idx][timestep] += 1
        self.increment()

    def train(self):
        



    def add(self, item): 
        if self.BOOTSTRAPPING:
            self.bootstrap.add(item)

    def get_helper(self, item):
        self.record(item)
        try:
            self.req_hist[item.name] += 1
            return self.store[item.name]
        except:
            return None

    def get(self, item_name):
        """Wrapper function for get_helper which includes hit/miss statistic updates"""
        item = self.get_helper(item_name)
        if item != None and not self.BOOTSTRAPPING:
            self.hits += 1
        elif not self.BOOTSTRAPPING:
            self.misses += 1
        else: # use bootstrap
            item = self.bootstrap.get(item_name)
            self.hits = self.bootstrap.hits
            self.misses = self.bootstrap.misses

        return item




    def populate(self):
        pass

    def record_timestep(self, l, content):
        l[content] += 1

    
