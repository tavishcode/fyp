import sys
sys.path.insert(0, './src')
from router import Router
from consumer import Consumer
from producer import Producer
from graph import Graph
from packet import Packet
import random
import math
import numpy as np

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
    nx.draw(gr, node_size = 500, with_labels = True)
    plt.show()

class Simulator:

    def __init__(self, num_consumers, num_producers, num_requests_per_consumer, grid_rows, grid_cols):
        self.NUM_REQUESTS_PER_CONSUMER = num_requests_per_consumer
        self.ZIPF_S = 1.2
        self.REQUEST_RATE = 1
        self.NUM_CONTENT_TYPES = num_producers
        self.CACHE_SIZE = int(0.1 * self.NUM_CONTENT_TYPES)

        self.consumers = []
        self.producers = []
       
        num_routers = grid_rows * grid_cols

        self.net_core = Graph(self.CACHE_SIZE, grid_rows, grid_cols)

        # assign consumers and producers with gateway routers
        for i in range(num_consumers):
            r = self.net_core.get_random_router()
            self.consumers.append(
                Consumer("c" + str(i), r)
            )

        for i in range(num_producers):
            r = self.net_core.get_random_router()
            self.producers.append(
                Producer("p" + str(i), r, "content" + str(i))
            )

        # set FIBs in routers
        self.net_core.setRoutesToProducers(self.producers)

        # populate content
        self.content_types = ["content" + str(i) for i in range(num_producers)]
        
        #generate probability distribution
        self.zipf_weights = [(1/k**self.ZIPF_S)/ (sum([1/n**self.ZIPF_S for n in range(1, self.NUM_CONTENT_TYPES+1)])) for k in range(1,self.NUM_CONTENT_TYPES+1)]

    def get_next_actor(self):
        arr = self.consumers + self.producers + self.net_core.routers
        min_time = None
        actor = None
        for i in range(len(arr)):
            if len(arr[i].q) > 0 and (min_time == None or arr[i].q[0][0] < min_time):
                min_time = arr[i].q[0][0]
                actor = arr[i]
        return actor

    def set_next_content_requests(self):
        for consumer in self.consumers:
            # append (time, packet) pair into request queue
            consumer.q.append([consumer.clock + np.random.exponential(1/self.REQUEST_RATE), 'REQ', Packet(random.choices(self.content_types, self.zipf_weights)[0])])
    
    def run(self):
        self.set_next_content_requests()
        num_request_wave = 1
        actor = self.get_next_actor()
        while actor != None:
            actor.execute()
            if num_request_wave < self.NUM_REQUESTS_PER_CONSUMER:
                self.set_next_content_requests()
                num_request_wave += 1
            actor = self.get_next_actor()
        # visualize(self.net_core.adj_mtx, self.consumers, self.producers)

if __name__ == "__main__":
    sim = Simulator(num_consumers = 2, num_producers = 1, num_requests_per_consumer = 1, grid_rows = 1, grid_cols = 1)
    sim.run()
