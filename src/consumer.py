""" A CCN Consumer Node

    Attributes: 
        name: name to address node
        gateway: name of router through which all traffic is routed
        clock: time of last request made by consumer
        q: A queue of events receieved by a node
"""
class Consumer:
    def __init__(self, name, gateway):
        self.name = name
        self.gateway = gateway
        self.clock = 0
        self.q = []

    def execute(self):
        """Call next event in consumer q"""
        event = self.q.pop(0)
        if event['type'] == 'REQ':
            self.request(event['time'], event['src'])
        elif event['type'] == 'REC':
            self.receive(event['time'], event['pkt'], event['src'])

    def request(self, time, pkt):
        """Add receieve event for an interest packet to gateway"""
        print(self.name + ' requests ' + pkt.name)
        self.clock = time
        self.gateway.q.append({'time': time + 0.1, 'type': 'REC', 'pkt': pkt, 'src': self})
        self.gateway.q.sort(key=lambda x: x['time'])

    def receive(self, time, pkt, src):
        """Log information for receievd data packet"""
        pkt.hop_count += 1
        print(self.name + ' receives pkt ' + pkt.name + ' after ' + str(pkt.hop_count) + ' hops')
