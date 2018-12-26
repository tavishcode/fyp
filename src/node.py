from contentstore import ContentStore
from packet import Packet

class Node:
    def __init__(self, fib, size):
        self.name = name
        self.CACHESIZE = size
        self.content_store = ContentStore(size)
        self.fib = fib
        
    def forward(self, pkt):
        # not sure what this is
        

    def get_gateway():
        return


    def recv(self, pkt: Packet, src):
        pkt.hopcount+=1 
        if pkt.is_interest:
            found = self.content_store.has(pkt)
            if found in not None:
                return found
            else:
                #forward interest packet to next node?
        else:
            #TODO to be implemented if PIT
            print("data packet")


    