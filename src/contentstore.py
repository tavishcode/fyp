from packet import Packet
from collections import OrderedDict

class ContentStore:
    def __init__(self, size):
        self.size = size

    def add(self, item):
        pass

    def get(self, pkt):
        pass

class FifoContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)
        self.store = OrderedDict()

    def add(self, item):
        if self.size:
            if(len(self.store) == self.size):
                self.store.pop(last=False)
            self.store[item.name] = item

    def get(self, item):
        try:
            return self.store[item.name]
        except:
            return None

class LruContentStore(ContentStore):
    def __init__(self, size):
        super().__init__(size)
        self.store = OrderedDict()

    def add(self, item):
        if self.size:
            if(len(self.store) == self.size):
                self.store.pop(last=False)
            self.store[item.name] = item

    def get(self, item):
        try:
            cached_item = self.store.pop(item.name)
            self.store[item.name] = cached_item
            return cached_item
        except:
            return None

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
                    if min_freq ==  None or self.store[key][1] < min_freq:
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

        








