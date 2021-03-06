from .src.router import Router
from .src.consumer import Consumer
from .src.producer import Producer
from .src.packet import Packet
from .src.eventlist import *
from .graph import Graph
import random
import math
import numpy as np
from matplotlib import pyplot as plt

"""Creates visualization for simulation"""


def visualize(adj_mtx, consumers, producers):
  import matplotlib.pyplot as plt
  import networkx as nx
  gr = nx.Graph()
  for i in range(len(adj_mtx)):
    gr.add_node('r' + str(i))
  for i in range(len(adj_mtx)):
    for j in range(len(adj_mtx[i])):
      if adj_mtx[i][j] == 1:
        gr.add_edge('r' + str(i), 'r' + str(j))
  for c in consumers:
    gr.add_node(c.name)
    gr.add_edge(c.name, c.gateway.name)
  for p in producers:
    gr.add_node(p.name)
    gr.add_edge(p.name, p.gateway.name)
  nx.draw(gr, node_size=500, with_labels=True)
  plt.show()


class Simulator:

  def __init__(self,
               num_consumers,
               num_content_types,
               end_day,
               request_rate,
               cache_update_interval,
               grid_rows, grid_cols,
               cache_ratio,
               policy,
               requests,
               total_reqs,
               rand_seed=123):
    # requests/s for each consumer
    self.REQUEST_RATE = request_rate
    # number of unique content types
    self.NUM_CONTENT_TYPES = num_content_types
    self.CACHE_SIZE = int(cache_ratio * self.NUM_CONTENT_TYPES)
    # interval to call update_state() on caches
    self.CACHE_UPDATE_INTERVAL = cache_update_interval
    self.RAND_SEED = rand_seed
    self.policy = policy
    self.requests = requests
    self.total_reqs = total_reqs
    self.curr_request = 0
    self.num_requests = 0
    self.q = EventList()
    random.seed(self.RAND_SEED)
    np.random.seed(self.RAND_SEED)

    # random number generator responsible for all non third-party randomizations
    self.rng = np.random.RandomState(self.RAND_SEED)
    # global queue of events
    self.prev_cache_update = 0
    self.curr_time = 0
    # continuously increasing time
    self.curr_day = 0
    self.end_day = end_day
    self.consumers = []
    self.producers = []

    # number of total routers in topology
    num_routers = grid_rows * grid_cols

    # populate content
    self.content_types = ["content" + str(i)
                          for i in range(self.NUM_CONTENT_TYPES)]

    # initialize routers
    self.net_core = Graph(
        self.CACHE_SIZE,
        self.NUM_CONTENT_TYPES,
        grid_rows,
        grid_cols,
        self.policy,
        self.q,
        self.rng
    )

    # assign consumers and producers with gateway routers
    for i in range(num_consumers):
      r = self.net_core.get_random_router()
      self.consumers.append(
          Consumer("c" + str(i), r, self.q)
      )

    for i in range(num_content_types):
      r = self.net_core.get_random_router()
      self.producers.append(
          Producer("p" + str(i), r, "content" + str(i), self.q)
      )

    # set FIBs in routers
    self.net_core.set_routes_to_producers(self.producers)

  def get_next_actor(self):
    """Returns next actor (node) to execute event for (event with min value for time)"""
    min_time = None
    actor = None
    # get next event from queue
    actor_name, min_time = self.q.peek()
    if actor_name != None:                                  # if q is not empty
      actor_ix = int(actor_name[1:])
      actor_type = actor_name[0]
      if actor_type == 'c':                               # if consumer
        actor = self.consumers[actor_ix]
        # schedule next consumer request,
        self.set_next_content_request(actor)
        # ensures queue is never empty and simulation
        # will run until end time
      elif actor_type == 'r':                             # if router
        actor = self.net_core.routers[actor_ix]
      else:                                               # if producer
        actor = self.producers[actor_ix]
      # events cannot occur in the past
      assert(self.curr_time <= min_time)
    self.curr_time = min_time
    return actor

  def set_next_content_request(self, consumer):
    self.curr_request += 1
    # DEBUG: for tests using a subset of content types
    weights = self.requests[:, self.curr_day] / \
        (sum(self.requests[:, self.curr_day]))
    content_name = self.rng.choice(
        self.content_types, 1, p=weights)[0]
    self.q.add(Event(consumer.name, consumer.time_of_next_request,
                     'REQ', Packet(content_name), None))
    if self.curr_request >= self.total_reqs:
      self.curr_day += 1
      self.curr_request = 0
    consumer.time_of_next_request += self.rng.exponential(
        1/self.REQUEST_RATE)

  def run(self):
    """Executes events for nodes"""
    for consumer in self.consumers:                 # init simulation with a content request from all consumers
      self.set_next_content_request(consumer)
    actor = self.get_next_actor()
    while self.curr_day < self.end_day:
      if self.curr_day - self.prev_cache_update >= self.CACHE_UPDATE_INTERVAL:
        self.prev_cache_update = self.curr_day
        for ix, router in enumerate(self.net_core.routers):
          router.contentstore.refresh()
          # total = router.contentstore.hits + router.contentstore.misses
          # if total:
          #   print(router.name + ' ' + self.policy +
          #         ' Hit Rate: ' + str(router.contentstore.hits/total))
      actor.execute()
      actor = self.get_next_actor()
    # for ix, router in enumerate(self.net_core.routers):
    #   total = router.contentstore.hits + router.contentstore.misses
    #   if total:
    #     print(router.name + ' ' + self.policy +
    #           ' Hit Rate: ' + str(router.contentstore.hits/total))
