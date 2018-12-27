# TODO update with sets
from packet import Packet

class ContentStore:
    def __init__(self,space):
        self.SIZE = space
        self.store = [None] * self.SIZE 

    def available_space(self):
        available = self.space
        for i in range(len(self.store)):
            if self.store[i] is not None:
                available = available-1
        return available, self.SIZE

    def add_item(self, item):
        self.store.remove()
        self.store.append(item)
        return

    def empty_store(self):
        self.store = [None] * self.SIZE
        return

    def has(self,pkt: Packet):
        for item in self.store:
            while item is not None:
                if item.get_name() == pkt.get_name():
                    return item
        return None





