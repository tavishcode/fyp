from contentstore import ContentStore
from packet import Packet



class Node:
    def __init__(self, fib, size, name):
        self.CACHESIZE = size
        self.content_store = ContentStore(size)
        self.FIB = fib # a dict "node name" : node
        self.PIT = {} # a dict with "packet name" : src node
        self.name = name
    
    def print_fib(self):
        print("FIB of " + self.name)
        for item in self.FIB.items():
            print(item[0] + ' , ' + item[1].get_name())

    def setFIB(self, fib):
        self.FIB = fib

    def print_pit(self):
        print("PIT of node")
        for item in self.PIT.items():
            print(item)

    def get_name(self):
        return self.name

    def forward(self, pkt: Packet, src):
        dest = self.FIB[pkt.getSourceName()]
        print("Forward from: " + src.get_name() + " to " + dest.get_name())  
        self.print_fib()
        dest.receive(pkt, src)
        
    def get_gateway(self): 
        if (len(set(self.FIB.values())) == 1):
            return list(self.FIB.values())[0]
        return False

    def receive(self, pkt: Packet, src):
        print("Receiving at " + self.name)
        pkt.hop_count+=1 
        if pkt.is_interest():
            found = self.content_store.has(pkt)
            if found is not None:
                src.receive(found, self)
            else:
                self.PIT[pkt.get_name()] = src
                self.forward(pkt, self)
        else:
            if pkt.get_name() in self.PIT:
                self.content_store.add_item(pkt)
                self.PIT[pkt.get_name()].receive(pkt, src)
            else:
                return pkt
    

