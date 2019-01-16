import sys
sys.path.insert(0, './src')
from contentstore import ContentStore
from packet import Packet

class Router:
    def __init__(self, cache_size, name):
        self.content_store = ContentStore(cache_size)
        self.FIB = {} # a dict with "node name" : node
        self.PIT = {} # a dict with "packet name" : [[node, hop_count], ...]
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
        pkt.hop_count += 1
        if pkt.is_interest:
            print(self.name + ' receives request for ' + pkt.name)
            found = self.content_store.get(pkt)
            if found != None:
                print(self.name + ' found ' + pkt.name + ' in cache')
                new_data_pkt = Packet(pkt.name, is_interest=False, hop_count=pkt.hop_count)
                src.q.append([time + 0.1, 'REC', new_data_pkt, self])
            else:
                if pkt.name in self.PIT and len(self.PIT[pkt.name]) > 0:
                    self.PIT[pkt.name].append([src, pkt.hop_count])
                else:
                    self.PIT[pkt.name] = [[src, pkt.hop_count]]
                    self.FIB[pkt.name].q.append([time + 0.1, 'REC', pkt, self])
        else:
            print(self.name + ' receives data packet for ' + pkt.name)
            self.content_store.add_item(pkt)
            for ix, val in enumerate(self.PIT[pkt.name]):
                node, hop_count = val
                if ix == 0:
                    node.q.append([time + 0.1, 'REC', pkt, self])
                else:
                    node.q.append([time + 0.1, 'REC', Packet(pkt.name, is_interest=False, hop_count=hop_count), self])
            self.PIT[pkt.name] = []