import sys
sys.path.insert(0, './src')
from packet import Packet

""" A CCN Producer Node Node

    Attributes: 
        name: name to address node
        gateway: name of router through which all traffic is routed
        content: name of content stored by this producer
        q: A queue of events receieved by a node
"""
class Producer:
    def __init__(self, name, gateway, content):
        self.name = name
        self.gateway = gateway
        self.content = content
        self.q = []

    def execute(self):
        """Execute next event in producer q"""
        event = self.q.pop(0)
        if event['type'] == 'REC':
            self.receive(event['time'], event['pkt'], event['src'])

    def receive(self, time, pkt, src):
        """Adds rcv event for a data packet to gateway"""
        print(self.name + ' receives request for ' + pkt.name)
        pkt.hop_count += 1
        new_data_pkt = Packet(pkt.name, is_interest=False, hop_count=pkt.hop_count)
        src.q.append({'time': time + 0.1, 'type': 'REC', 'pkt': new_data_pkt,'src': self})
        src.q.sort(key=lambda x: x['time'])
