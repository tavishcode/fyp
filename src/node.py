from contentstore import ContentStore
from packet import Packet


class Node:
    def __init__(self, fib, size):
        self.CACHESIZE = size
        self.content_store = ContentStore(size)
        self.FIB = fib
        self.PIT = {}
        
    def forward(self, pkt: Packet, src):
        self.FIB[pkt].receive(pkt, src)
        
    def get_gateway(self):
        return

    def receive(self, pkt: Packet, src):
        pkt.hopcount+=1 
        if pkt.is_interest:
            found = self.content_store.has(pkt)
            if found is not None:
                return found
            else:
                self.PIT.update(pkt=src)
                self.forward(pkt, src)
        else:
            if pkt in self.PIT:
                for item in self.PIT[pkt]:
                    item.receive(pkt, src)
            else:
                return pkt
