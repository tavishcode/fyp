import sys
sys.path.insert(0, './src')
from contentstore import FifoContentStore, LruContentStore, LfuContentStore
from packet import Packet

""" A CCN Router Node

    Attributes: 
        contentstore: A cache 
        name: name to address node
        FIB: Dict with "node name" : node
        PIT: Dict with "packet name" : [[node, hop_count], ...]
        q: A queue of events receieved by a node
"""
class Router:
    def __init__(self, cache_size, name, policy):
        if policy == 'fifo':
            self.contentstore = FifoContentStore(cache_size)
        elif policy == 'lru':
            self.contentstore = LruContentStore(cache_size)
        elif policy == 'lfu':
            self.contentstore = LfuContentStore(cache_size)
        self.FIB = {} 
        self.PIT = {} 
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
        """Calls next event in q"""
        event = self.q.pop(0)
        if event['type'] == 'REC':
            self.receive(event['time'], event['pkt'], event['src'])

    def receive(self, time, pkt, src):
        """ If pkt is interest, retrives pkt from contentstore or adds receive event to q of next hop for pkt.
            Aggregates interest pkts in PIT
            If pkt is data, adds receive event to all nodes in PIT[pk.name]
        """
        pkt.hop_count += 1
        if pkt.is_interest:
            print(self.name + ' receives request for ' + pkt.name)
            found = self.contentstore.get(pkt)
            if found != None:
                print(self.name + ' found ' + pkt.name + ' in cache')
                new_data_pkt = Packet(pkt.name, is_interest=False, hop_count=pkt.hop_count)
                src.q.append({'time': time + 0.1, 'type': 'REC','pkt': new_data_pkt, 'src': self})
                src.q.sort(key=lambda x: x['time'])
            else:
                if pkt.name in self.PIT and len(self.PIT[pkt.name]) > 0:
                    self.PIT[pkt.name].append([src, pkt.hop_count])
                else:
                    self.PIT[pkt.name] = [[src, pkt.hop_count]]
                    self.FIB[pkt.name].q.append({'time': time + 0.1, 'type': 'REC','pkt': pkt,'src': self})
                    self.FIB[pkt.name].q.sort(key=lambda x: x['time'])
        else:
            print(self.name + ' receives data packet for ' + pkt.name)
            self.contentstore.add(pkt)
            for ix, val in enumerate(self.PIT[pkt.name]):
                node, hop_count = val
                if ix == 0:
                    node.q.append({'time': time + 0.1,'type': 'REC', 'pkt': pkt, 'src': self})
                    node.q.sort(key=lambda x: x['time'])
                else:
                    node.q.append({'time': time + 0.1,'type': 'REC','pkt': Packet(pkt.name, is_interest=False, hop_count=hop_count), 'src': self})
                    node.q.sort(key=lambda x: x['time'])
            self.PIT[pkt.name] = []