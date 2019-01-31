import sys
sys.path.insert(0, './src')
from router import Router
from consumer import Consumer
from producer import Producer
from graph import Graph
from packet import Packet
from test import *
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

    def __init__(self, num_consumers, num_producers, end_time, grid_rows, grid_cols, cache_ratio, policy, rand_seed = 123):
        self.ZIPF_S = 1.2
        self.REQUEST_RATE = 1 # 1 req/s
        self.NUM_CONTENT_TYPES = num_producers
        self.CACHE_SIZE = int(cache_ratio * self.NUM_CONTENT_TYPES)
        self.CACHE_UPDATE_INTERVAL = 1e1
        self.ZIPF_UPDATE_INTERVAL = 5e4
        self.RAND_SEED = rand_seed
        
        random.seed(self.RAND_SEED)
        np.random.seed(self.RAND_SEED)

        self.prev_cache_update = 0
        self.prev_zipf_update = 0
        self.curr_time = 0 # continuously increasing time
        self.end_time = end_time
        self.num_request_waves = 0
        self.consumers = []
        self.producers = []
       
        num_routers = grid_rows * grid_cols

        self.net_core = Graph(self.CACHE_SIZE, self.NUM_CONTENT_TYPES, grid_rows, grid_cols, policy)

        self.req_counts = []
        for r in self.net_core.routers:
            self.req_counts.append([])

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
        
        # generate probability distribution
        self.zipf_weights = [(1/k**self.ZIPF_S)/ (sum([1/n**self.ZIPF_S for n in range(1, self.NUM_CONTENT_TYPES+1)])) for k in range(1,self.NUM_CONTENT_TYPES+1)]
        random.shuffle(self.zipf_weights)
        # print(self.content_types)
        # print(self.zipf_weights)

    
    def get_next_actor(self):
        """Returns next actor (node) to execute event for (event with min value for time)"""
        arr = self.consumers + self.producers + self.net_core.routers
        min_time = None
        actor = None
        for i in range(len(arr)):
            if len(arr[i].q) > 0 and (min_time == None or arr[i].q[0]['time'] < min_time):
                min_time = arr[i].q[0]['time']
                actor = arr[i]
        for consumer in self.consumers:
            if actor.name == consumer.name:
                self.set_next_content_request(consumer)
        if min_time!= None:
            assert(self.curr_time <= min_time)
        self.curr_time = min_time
        return actor

    def set_next_content_request(self, consumer):
        """Append (time, req, packet) pair into consumer queue"""
        consumer.q.append({
            'time': consumer.time_of_next_request,
            'type': 'REQ',
            'pkt': Packet(np.random.choice(self.content_types, 1, p=self.zipf_weights)[0])
            })
        consumer.q.sort(key=lambda x: x['time'])
        consumer.time_of_next_request += np.random.exponential(1/self.REQUEST_RATE)

    def run(self):
        """Executes events for nodes
        Calls content request waves after each event for NUM_REQUESTS_PER_CONSUMER waves"""
        for consumer in self.consumers:
            self.set_next_content_request(consumer)
        actor = self.get_next_actor()
        while self.curr_time < self.end_time:
            # print(self.curr_time)
            if self.curr_time == 0 and self.prev_cache_update == 0:
               # first run of algorithm (no prior training)
               pass
            if self.curr_time - self.prev_cache_update > self.CACHE_UPDATE_INTERVAL:
                self.prev_cache_update = self.curr_time
                print(self.curr_time)
                for ix, router in enumerate(self.net_core.routers):
                    req_count = router.contentstore.update_state()
                    self.req_counts[ix].append(req_count)
            if self.curr_time - self.prev_zipf_update > self.ZIPF_UPDATE_INTERVAL:
                self.prev_zipf_update = self.curr_time
                random.shuffle(self.zipf_weights)
                # print(self.content_types)
                # print(self.zipf_weights)
            actor.execute()
            actor = self.get_next_actor()
            
        # visualize(self.net_core.adj_mtx, self.consumers, self.producers)

if __name__ == "__main__":
    RAND_SEED = 123
    
    """ Simulation Sample Scenario """

    sim = Simulator(
        num_consumers=1, 
        num_producers=25, 
        end_time=5e6,
        grid_rows=1, 
        grid_cols=1, 
        cache_ratio=0.1,
        policy='dlcpp', 
        rand_seed=RAND_SEED
    )
    sim.run()

    # for router in sim.net_core.routers:
    #     plt.plot(router.contentstore.rewards)
    #     plt.show()

    """Print requests per timestep"""

    # print(sim.req_counts)

    """Plot distribution of requests"""
    
    # req_freqs = [0 for c in range(sim.NUM_CONTENT_TYPES)]
    # for consumer in sim.consumers:
    #     for i in range(sim.NUM_CONTENT_TYPES):
    #         req_freqs[i] += consumer.gateway.contentstore.req_hist['content'+str(i)]
    #     print(consumer.gateway.contentstore.req_hist)
    # print(req_freqs)
    # plt.bar([i for i in range(sim.NUM_CONTENT_TYPES)], req_freqs)
    # plt.show()

    """ Traditional Cache Experiment"""
    
    # cache_ratios = [0.05]
    # policies = ['ddpg', 'lfu']
    # for policy in policies:
    #     hit_ratios = []
    #     for cache_ratio in cache_ratios:
    #         sim = Simulator(
    #                 num_consumers=1, 
    #                 num_producers=100, 
    #                 num_requests_per_consumer=20000, 
    #                 grid_rows=1, 
    #                 grid_cols=1, 
    #                 cache_ratio=cache_ratio,
    #                 policy=policy, 
    #                 rand_seed=RAND_SEED
    #             )
    #         sim.run()
    #         hits = 0
    #         reqs = 0
    #         for router in sim.net_core.routers:
    #             reqs += router.contentstore.hits + router.contentstore.misses
    #             hits += router.contentstore.hits
    #         hit_ratios.append(hits/reqs)
    #         print(hit_ratios)
    #     print(hit_ratios)
    #     plt.plot(hit_ratios)
    # plt.legend(policies)
    # plt.show()
