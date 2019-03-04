""" A CCN Consumer Node

    Attributes: 
        name: name to address node
        gateway: name of router through which all traffic is routed
        clock: time of last request made by consumer
        q: a queue of events receieved by a node
        
"""
from .eventlist import *

class Consumer:
  def __init__(self, name, gateway, q):
    self.name = name
    self.gateway = gateway
    self.q = q
    self.time_of_next_request = 0

  def execute(self):
    """Call next event in consumer q"""
    event = self.q.popfront()
    assert(event.actor_name == self.name)
    if event.func == 'REQ':
      self.request(event.time, event.pkt)
    elif event.func == 'REC':
      self.receive(event.time, event.pkt, event.src)

  def request(self, time, pkt):
    """Add receieve event for an interest packet to gateway"""
    # print(self.name + ' requests ' + pkt.name)
    self.q.add(Event(self.gateway.name, time+0.1, 'REC', pkt, self))

  def receive(self, time, pkt, src):
    """Log information for receievd data packet"""
    pkt.hop_count += 1
    # print(self.name + ' receives pkt ' + pkt.name + ' after ' + str(pkt.hop_count) + ' hops')
