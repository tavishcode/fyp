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
    nx.draw(gr, node_size = 500, with_labels = True)
    plt.show()


""" Creates CCN Simulation for a given network scenaario

    Attributes:
        NUM_REQUESTS_PER_CONSUMER: num pkt requests each consumer will make
        REQUEST_RATE:
        ZIPF_S: Parameter for Zipf Distribution
        NUM_CONTENT_TYPES: num of unique pkt names in network
        CACHE_SIZE: size of router caches
        TIME_STEP: length of timestep interval
        consumers: list of all consumer nodes
        producers: list of all producer nodes
        net_core: Container for interfacing with router nodes
        zipf_weights: list of zipf distribution based probabilities for content_types

"""
class Simulator:

    def __init__(self, num_consumers, num_producers, num_requests_per_consumer, grid_rows, grid_cols):
        self.NUM_REQUESTS_PER_CONSUMER = num_requests_per_consumer
        self.ZIPF_S = 1.2
        self.REQUEST_RATE = 1
        self.NUM_CONTENT_TYPES = num_producers
        self.CACHE_SIZE = 1 #int(0.1 * self.NUM_CONTENT_TYPES)
        self.TIMESTEP = 10
        
        self.prev_time = 0
        self.curr_time = 0
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
        self.net_core.set_routes_to_producers(self.producers)

        # populate content
        self.content_types = ["content" + str(i) for i in range(num_producers)]
        
        #generate probability distribution
        self.zipf_weights = [(1/k**self.ZIPF_S)/ (sum([1/n**self.ZIPF_S for n in range(1, self.NUM_CONTENT_TYPES+1)])) for k in range(1,self.NUM_CONTENT_TYPES+1)]

    def get_next_actor(self):
        """Returns next actor (node) to execute event for (event with min value for time)"""
        arr = self.consumers + self.producers + self.net_core.routers
        min_time = None
        actor = None
        for i in range(len(arr)):
            if len(arr[i].q) > 0 and (min_time == None or arr[i].q[0]['time'] < min_time):
                min_time = arr[i].q[0]['time']
                actor = arr[i]
        self.curr_time = min_time
        return actor

    def set_next_content_requests(self):
        """Append (time, req, packet) pair into each consumerevent queue"""
        for consumer in self.consumers:
            consumer.q.append({
                'time': consumer.clock + np.random.exponential(1/self.REQUEST_RATE),
                'type': 'REQ',
                'pkt': Packet(random.choices(self.content_types, self.zipf_weights)[0])
              })
            consumer.q.sort(key=lambda x: x['time'])
    
    def run(self):
        """Executes events for nodes
        Calls content request waves after each event for NUM_REQUESTS_PER_CONSUMER waves"""
        self.set_next_content_requests()
        num_request_wave = 1
        actor = self.get_next_actor()
        while actor != None:
            if self.curr_time == 0 and self.prev_time == 0:
                print('first use of algorithm (random params)')
            elif self.curr_time - self.prev_time > self.TIMESTEP:
                # train algorithm
                self.prev_time = self.curr_time
                # update router features after a timestep
                for router in self.net_core.routers:
                    router.contentstore.update_state()
            actor.execute()
            if num_request_wave < self.NUM_REQUESTS_PER_CONSUMER:
                self.set_next_content_requests()
                num_request_wave += 1
            actor = self.get_next_actor()
        visualize(self.net_core.adj_mtx, self.consumers, self.producers)

if __name__ == "__main__":
    sim = Simulator(num_consumers = 2, num_producers = 1, num_requests_per_consumer = 1, grid_rows = 1, grid_cols = 2)
    sim.run()
