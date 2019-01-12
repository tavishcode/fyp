from packet import Packet

class ContentStore:
    def __init__(self, space):
        self.SIZE = space
        self.store = []

    def available_space(self):
        available = self.SIZE - len(self.store)
        return available, self.SIZE

    def add_item(self, item):
        if(len(self.store) < self.SIZE):
            self.store.append(item)
        else:
            self.store.pop() # TODO: Fancy way to discard a packet
            self.store.append(item) 

    def empty_store(self):
        self.store = []

    def get(self, pkt):
        for item in self.store:
            if item.name == pkt.name:
                return item
        return None





