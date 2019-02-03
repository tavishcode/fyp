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
from eventlist import *

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

    def __init__(self, 
                num_consumers, 
                num_content_types, 
                end_time, 
                request_rate,
                zipf_s,
                m_q,
                num_cycles,
                zipf_update_interval, 
                cache_update_interval,
                grid_rows, grid_cols, 
                cache_ratio, 
                policy, 
                rand_seed = 123):

        self.ZIPF_S = zipf_s                                            # zipf skew
        self.M_Q = m_q                                                  # mandelbrot q
        self.NUM_CYCLES = num_cycles                                    # number of zipf cycles
        self.REQUEST_RATE = request_rate                                # requests/s for each consumer
        self.NUM_CONTENT_TYPES = num_content_types                      # number of unique content types
        self.CACHE_SIZE = int(cache_ratio * self.NUM_CONTENT_TYPES)     
        self.CACHE_UPDATE_INTERVAL = cache_update_interval              # interval to call update_state() on caches
        self.ZIPF_UPDATE_INTERVAL = zipf_update_interval                # interval to switch zipf distributions
        self.RAND_SEED = rand_seed
        self.policy = policy

        random.seed(self.RAND_SEED)
        np.random.seed(self.RAND_SEED)

        self.rng = np.random.RandomState(self.RAND_SEED)                # random number generator responsible for all non third-party randomizations

        self.history = []                                               # history of requests throughout simulation
        for i in range(self.NUM_CYCLES):
            self.history.append([])

        self.q = EventList()                                            # global queue of events
        self.zipf_cycle = 0                                             # counter to switch zipf distributions
        self.prev_cache_update = 0                                      
        self.prev_zipf_update = 0
        self.curr_time = 0                                              # continuously increasing time
        self.end_time = end_time
        self.consumers = []
        self.producers = []
       
        num_routers = grid_rows * grid_cols                             # number of total routers in topology

        # populate content
        self.content_types = ["content" + str(i) \
            for i in range(self.NUM_CONTENT_TYPES)] 
        
        # generate probability distribution with seasonal cycles
        total = sum([1/(n + self.M_Q)**self.ZIPF_S \
            for n in range(1, self.NUM_CONTENT_TYPES+1)])
        self.zipf_weights = [1/(k + self.M_Q)**self.ZIPF_S/total \
            for k in range(1,self.NUM_CONTENT_TYPES+1)]
        self.zipf_set = [self.rng.permutation(self.zipf_weights) \
            for i in range(self.NUM_CYCLES)]

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
        actor_name, min_time = self.q.peek()                    # get next event from queue
        if actor_name != None:                                  # if q is not empty
            actor_ix = int(actor_name[1:])      
            actor_type = actor_name[0]
            if actor_type == 'c':                               # if consumer
                actor = self.consumers[actor_ix]
                self.set_next_content_request(actor)            # schedule next consumer request,
                                                                # ensures queue is never empty and simulation
                                                                # will run until end time

            elif actor_type == 'r':                             # if router
                actor = self.net_core.routers[actor_ix]
            else:                                               # if producer
                actor = self.producers[actor_ix]
            assert(self.curr_time <= min_time)                  # events cannot occur in the past
        self.curr_time = min_time
        return actor

    def set_next_content_request(self, consumer):
        """Append (time, req, packet) pair into consumer queue"""
        content_name = self.rng.choice(
                                        self.content_types, 
                                        1, 
                                        p=self.zipf_set[self.zipf_cycle % self.NUM_CYCLES]
                                    )[0]
        self.q.add(Event(
                consumer.name,
                consumer.time_of_next_request,
                'REQ',
                Packet(content_name),
                None
        ))
        consumer.time_of_next_request += self.rng.exponential(1/self.REQUEST_RATE)
        self.history[self.zipf_cycle % self.NUM_CYCLES].append(content_name) # update sim history

    def run(self):
        """Executes events for nodes"""
        for consumer in self.consumers:                 # init simulation with a content request from all consumers
            self.set_next_content_request(consumer)
        actor = self.get_next_actor()
        while self.curr_time < self.end_time:
            # print(self.curr_time)
            if self.curr_time - self.prev_cache_update > self.CACHE_UPDATE_INTERVAL:
                self.prev_cache_update = self.curr_time
                for ix, router in enumerate(self.net_core.routers):
                    router.contentstore.update_state()
            if self.curr_time - self.prev_zipf_update > self.ZIPF_UPDATE_INTERVAL:
                self.prev_zipf_update = self.curr_time
                self.zipf_cycle += 1
            actor.execute()
            actor = self.get_next_actor()
            
        # visualize(self.net_core.adj_mtx, self.consumers, self.producers)

if __name__ == "__main__":
    RAND_SEED = 123
    
    """ Simulation Sample Scenario """

    for policy in ['lru','lfu','gru']:
        sim = Simulator(
            num_consumers=1, 
            num_content_types=50000, 
            end_time=50000, 
            request_rate=1,
            zipf_s=0.7,
            m_q=0.7,
            num_cycles=3,
            zipf_update_interval=10000, 
            cache_update_interval=100,
            grid_rows=1, 
            grid_cols=1, 
            cache_ratio=0.01, 
            policy=policy, 
            rand_seed=RAND_SEED
        )

        sim.run()

        """Optimal Hit Rate (Theoretical Maximum)"""
        
        num_requests = sum([len(hist) for hist in sim.history])
        num_hits = 0
        for i, hist in enumerate(sim.history):
            opt_contents = sorted(enumerate(sim.zipf_set[i]), key = lambda x: x[1], reverse=True)
            opt_cache = ['content' + str(ix) for ix, pop in opt_contents[:int(sim.CACHE_SIZE)]]
            for o in opt_cache:
                for h in hist:
                    if h == o:
                        num_hits += 1
        print('Optimal Hit Rate: ', num_hits/num_requests)
        
        """"""

        """ Policy Hit Rates """

        for router in sim.net_core.routers:
            total = router.contentstore.hits + router.contentstore.misses
            if total:
                print(router.name + ' ' + sim.policy + ' Hit Rate: ' + str(router.contentstore.hits/total))     

        """"""   


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
    
    # cache_ratios = [0.2,0.4,0.6]
    # policies = ['gru', 'lfu', 'lru']

    # for policy in policies:
    #     hit_ratios = []
    #     for cache_ratio in cache_ratios:
    #         sim = Simulator(
    #             num_consumers=5, 
    #             num_producers=5, 
    #             end_time=200000, 
    #             request_rate=1,
    #             zipf_s=1.2,
    #             zipf_update_interval=100000, 
    #             cache_update_interval=100,
    #             grid_rows=3, 
    #             grid_cols=3, 
    #             cache_ratio=cache_ratio, 
    #             policy=policy, 
    #             rand_seed = RAND_SEED
    #         )

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
