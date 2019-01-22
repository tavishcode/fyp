from packet import Packet
from collections import OrderedDict, defaultdict

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
    def __init__(self, size):
        super().__init__(size)
        self.store = OrderedDict()
        self.req_hist = defaultdict(int) # for testing purposes

    def update_state(self): # for testing purposes
        counts = len(self.req_hist)
        self.req_hist = defaultdict(int)
        return counts

    def add(self, item):
        if self.size:
            if(len(self.store) == self.size):
                self.store.popitem(last=False)
            self.store[item.name] = item

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
    def __init__(self, size):
        super().__init__(size)
        # record req history for last timestep and current timestep
        self.prev_reqs = defaultdict(int)
        self.curr_reqs = defaultdict(int)
    
    def get_helper(self, item):
        try:
            self.curr_reqs[item.name] += 1
        except:
            pass
    
    def update_state(self):
        self.prev_reqs = self.curr_reqs
        self.curr_reqs = defaultdict(int)

"""DDPG Cache Policy - RL"""
class DdpgContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)
        self.curr_reqs = defaultdict(int)
    
    def get_helper(self, item):
        try:
            self.curr_reqs[item.name] += 1
        except:
            pass

        








