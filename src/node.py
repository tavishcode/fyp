from contentstore import ContentStore
from packet import Packet



class Node:
    def __init__(self, fib, size):
        self.CACHESIZE = size
        self.content_store = ContentStore(size)
        self.FIB = fib # a dict "packet name" : node
        self.PIT = {} # a dict with "packet name" : src node
    
    def print_fib(self):
        print "FIB of node"
        for item in self.FIB.items():
            print item

    def print_pit(self):
        print "PIT of node"
        for item in self.PIT.items():
            print item

    def forward(self, pkt: Packet, src):
        self.FIB[pkt].receive(pkt, src)
        
    def get_gateway(self): 
        if (len(set(self.FIB.values())) == 1):
            return set(self.FIB.values())
        return False

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
