from contentstore import ContentStore
from packet import Packet

class Router:
    def __init__(self, fib, size, name):
        self.CACHESIZE = size
        self.content_store = ContentStore(size)
        self.FIB = fib # a dict "node name" : node
        self.PIT = {} # a dict with "packet name" : src node
        self.name = name

    def print_fib(self):
        print("FIB of " + self.name)
        for item in self.FIB.items():
            print(item[0] + ' , ' + item[1].name)

    def print_pit(self):
        print("PIT of " + self.name)
        for item in self.PIT.items():
            print(item)

    def receive(self, pkt, src):
        pkt.hop_count += 1 
        if pkt.is_interest:
            print(self.name + ' receives request for ' + pkt.name)
            found = self.content_store.get(pkt)
            if found is not None:
                new_data_pkt = Packet(pkt.name, is_interest=False)
                src.receive(new_data_pkt, self)
            else:
                # print("Forward from: " + src.name + " to " + dest.name)  
                # self.print_fib()
                self.PIT[pkt.name] = src
                self.FIB[pkt.name].receive(pkt, self)
        else:
            print(self.name + ' receives data packet for ' + pkt.name)
            self.content_store.add_item(pkt)
            self.PIT[pkt.name].receive(pkt, src)

class Consumer:
    def __init__(self, name, gateway):
        self.name = name
        self.gateway = gateway
        self.q = []

    def request(self, pkt):
        print(self.name + ' requests ' + pkt.name)
        self.gateway.receive(pkt, self)

    def receive(self, pkt, src):
        pkt.hop_count += 1
        print('Successfully received pkt ' + pkt.name + ' after ' + str(pkt.hop_count*2) + ' hops')

class Producer:
    def __init__(self, name, gateway, content):
        self.name = name
        self.gateway = gateway
        self.content = content

    def receive(self, pkt, src):
        print(self.name + ' receives request for ' + pkt.name)
        new_data_pkt = Packet(pkt.name, is_interest=False)
        src.receive(new_data_pkt, self)
