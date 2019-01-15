from packet import Packet

class ContentStore:
    def __init__(self, size):
        self.size = size
        self.store = []

    def available_space(self):
        available = self.size - len(self.store)
        return available, self.size

    def add_item(self, item):
        if self.size:
            if(len(self.store) < self.size):
                self.store.append(item)
            else:
                self.store.pop(0) # TODO: Fancy way to discard a packet
                self.store.append(item) 

    def empty_store(self):
        self.store = []

    def get(self, pkt):
        for item in self.store:
            if item.name == pkt.name:
                return item
        return None





