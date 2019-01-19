from packet import Packet
from collections import OrderedDict

""" Abstract Class for defining Cache Policies.

    Attributes: 
        size: no of items that can be stored in ContentStore.
"""
class ContentStore:
    def __init__(self, size):
        self.size = size
    
    def add(self, item):
        """Decides whether to add item to store and what to evict if store is full"""
        raise NotImplementedError('Base Class ContentStore does not implement a cache')

    def get(self, item_name):
        """Returns item with item_name if it is in store, else returns None"""
        raise NotImplementedError('Base Class ContentStore does not implement a cache')
        

"""First in First Out Cache Policy"""
class FifoContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)
        self.store = OrderedDict()

    def add(self, item):
        if self.size:
            if(len(self.store) == self.size):
                self.store.popitem(last=False)
            self.store[item.name] = item

    def get(self, item):
        try:
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

    def get(self, item):
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
        self.store = {} # ['name', [item, freq]]
    
    def add(self, item):
        if self.size:
            if(len(self.store) == self.size):
                min_key = None
                min_freq = None
                for key in self.store.keys():
                    if min_freq == None or self.store[key][1] < min_freq:
                        min_key = key
                self.store.pop(min_key)
            self.store[item.name] = [item, 1]

    def get(self, item):
        try:
            cached_item = self.store[item.name][0]
            self.store[item.name][1] += 1
            return cached_item
        except:
            return None

        








