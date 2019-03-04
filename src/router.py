from .contentstore import *
from .packet import Packet
from .eventlist import *

""" A CCN Router Node

    Attributes: 
        contentstore: A cache 
        name: name to address node
        FIB: Dict with "node name" : node
        PIT: Dict with "packet name" : [[node, hop_count], ...]
        q: A queue of events receieved by a node

"""


class Router:
  def __init__(self, cache_size, num_content_types, name, policy, q):

    if policy == 'fifo':
      self.contentstore = FifoContentStore(cache_size, num_content_types)
    elif policy == 'lru':
      self.contentstore = LruContentStore(cache_size)
    elif policy == 'lfu':
      self.contentstore = LfuContentStore(cache_size)
    elif policy == 'ddpg':
      self.contentstore = DdpgContentStore(cache_size, num_content_types)
    elif policy == 'gru':
      self.contentstore = GruContentStore(cache_size, num_content_types)
    elif policy == 'lookback':
      self.contentstore = LookbackContentStore(cache_size, num_content_types)
    elif policy == 'dlcpp':
      self.contentstore = DlcppContentStore(cache_size, num_content_types)
    elif policy == 'lstm':
      self.contentstore = LstmContentStore(cache_size, num_content_types)
    elif policy == 'probrl':
      self.contentstore = ProbRlContentStore(cache_size, num_content_types)
    elif policy == 'random':
      self.contentstore = RandomContentStore(cache_size)
    self.FIB = {}
    self.PIT = {}
    self.name = name
    self.q = q
    self.neighbors = []

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
    event = self.q.popfront()
    assert(event.actor_name == self.name)
    if event.func == 'REC':
      self.receive(event.time, event.pkt, event.src)

  def receive(self, time, pkt, src):
    """ If pkt is interest, retrives pkt from contentstore or adds receive event to q of next hop for pkt.
        Aggregates interest pkts in PIT
        If pkt is data, adds receive event to all nodes in PIT[pk.name]
    """
    pkt.hop_count += 1
    if pkt.is_interest:
      # print(self.name + ' receives request for ' + pkt.name)
      found = self.contentstore.get(pkt)
      if found != None:
        # print(self.name + ' found ' + pkt.name + ' in cache')
        new_data_pkt = Packet(pkt.name, is_interest=False,
                              hop_count=pkt.hop_count)
        self.q.add(Event(src.name, time+0.1, 'REC', new_data_pkt, self))
      else:
        if pkt.name in self.PIT and len(self.PIT[pkt.name]) > 0:
          self.PIT[pkt.name].append([src, pkt.hop_count])
        else:
          self.PIT[pkt.name] = [[src, pkt.hop_count]]
          self.q.add(Event(self.FIB[pkt.name].name,
                           time+0.1, 'REC', pkt, self))
    else:
      # print(self.name + ' receives data packet for ' + pkt.name)

      self.contentstore.add(pkt)
      for ix, val in enumerate(self.PIT[pkt.name]):
        node, hop_count = val
        if ix == 0:
          self.q.add(Event(node.name, time+0.1, 'REC', pkt, self))
        else:
          self.q.add(Event(node.name, time+0.1, 'REC', Packet(pkt.name,
                                                              is_interest=False, hop_count=hop_count), self))
      self.PIT[pkt.name] = []

  def set_neighbors(self, neighbors):
    self.neighbors = neighbors
