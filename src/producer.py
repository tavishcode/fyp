import sys
sys.path.insert(0, './src')
from packet import Packet

class Producer:
    def __init__(self, name, gateway, content):
        self.name = name
        self.gateway = gateway
        self.content = content
        self.q = []

    def execute(self):
        event = self.q.pop(0)
        if event[1] == 'REC':
            self.receive(event[0], event[2], event[3])

    def receive(self, time, pkt, src):
        print(self.name + ' receives request for ' + pkt.name)
        pkt.hop_count += 1
        new_data_pkt = Packet(pkt.name, is_interest=False, hop_count=pkt.hop_count)
        src.q.append([time + 0.1, 'REC', new_data_pkt, self])
        src.q.sort(key=lambda x: x[0])
