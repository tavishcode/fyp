class Consumer:
    def __init__(self, name, gateway):
        self.name = name
        self.gateway = gateway
        self.clock = 0
        self.q = []

    def execute(self):
        event = self.q.pop(0)
        if event[1] == 'REQ':
            self.request(event[0], event[2])
        elif event[1] == 'REC':
            self.receive(event[0], event[2], event[3])

    def request(self, time, pkt):
        print(self.name + ' requests ' + pkt.name)
        self.clock = time
        self.gateway.q.append([time + 0.1, 'REC', pkt, self])

    def receive(self, time, pkt, src):
        print(self.name + ' receives pkt ' + pkt.name + ' after ' + str(pkt.hop_count*2) + ' hops')
