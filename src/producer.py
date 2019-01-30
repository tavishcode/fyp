import sys
sys.path.insert(0, './src')
from packet import Packet
from eventlist import *

""" A CCN Producer Node

    Attributes: 
        name: name to address node
        gateway: name of router through which all traffic is routed
        content: name of content stored by this producer
        q: A queue of events receieved by a node
"""
class Producer:
    def __init__(self, name, gateway, content, q):
        self.name = name
        self.gateway = gateway
        self.content = content
        self.q = q

    def execute(self):
        """Execute next event in producer q"""
        event = self.q.popfront()
        assert(event.actor_name == self.name)
        if event.func == 'REC':
            self.receive(event.time, event.pkt, event.src)

    def receive(self, time, pkt, src):
        """Adds rcv event for a data packet to gateway"""
        # print(self.name + ' receives request for ' + pkt.name)
        pkt.hop_count += 1
        new_data_pkt = Packet(pkt.name, is_interest=False, hop_count=pkt.hop_count)
        self.q.add(Event(src.name, time+0.1, 'REC', new_data_pkt, self))
