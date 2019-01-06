from packet import Packet

class ContentStore:
    def __init__(self,space):
        self.SIZE = space
        self.store = set()

    def available_space(self):
        available = self.SIZE - len(self.store)
        return available, self.SIZE

    def add_item(self, item):
        if(len(self.store)<self.SIZE):
            self.store.add(item)
        else:
            self.store.pop() # TODO: Fancy way to discard a packet
            self.store.add(item) 

    def empty_store(self):
        self.store = self.store.clear()

    def has(self,pkt: Packet):
        for item in self.store:
            if item.get_name() == pkt.get_name():
                return item
        return None





