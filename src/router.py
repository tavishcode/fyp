import sys
sys.path.insert(0, './src')
from contentstore import ContentStore
from packet import Packet

class Router:
    def __init__(self, cache_size, name):
        self.content_store = ContentStore(cache_size)
        self.FIB = {} # a dict with "node name" : node
        self.PIT = {} # a dict with "packet name" : [nodes]
        self.name = name
        self.q = []

    def print_fib(self):
        print("FIB of " + self.name)
        for item in self.FIB.items():
            print(item[0] + ' , ' + item[1].name)

    def print_pit(self):
        print("PIT of " + self.name)
        for item in self.PIT.items():
            print(item)

    def execute(self):
        event = self.q.pop(0)
        if event[1] == 'REC':
            self.receive(event[0], event[2], event[3])

    def receive(self, time, pkt, src):
        if pkt.is_interest:
            pkt.hop_count += 1
            print(self.name + ' receives request for ' + pkt.name)
            found = self.content_store.get(pkt)
            if found != None:
                new_data_pkt = Packet(pkt.name, is_interest=False, hop_count=pkt.hop_count)
                src.q.append([time + 0.1, 'REC', new_data_pkt, self])
            else:
                if pkt.name in self.PIT and len(self.PIT[pkt.name]) > 0:
                    self.PIT[pkt.name].append(src)
                else:
                    self.PIT[pkt.name] = [src]
                    self.FIB[pkt.name].q.append([time + 0.1, 'REC', pkt, self])
        else:
            print(self.name + ' receives data packet for ' + pkt.name)
            self.content_store.add_item(pkt)
            for node in self.PIT[pkt.name]:
                node.q.append([time + 0.1, 'REC', pkt, self])
            self.PIT[pkt.name] = []