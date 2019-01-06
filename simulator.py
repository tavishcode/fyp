import sys
sys.path.insert(0, './src')
from node import Node
from graph import Graph
from packet import Packet

import random
import math
from random import choices



class Simulator:
    def __init__(self, num_consumers, num_producers, grid_rows = 3, grid_cols = 3):

        self.NUM_REQUESTS_PER_CONSUMER = 10
        self.NUM_CONTENT_TYPES = 10
        self.CACHE_SIZE = 0.1 * self.NUM_CONTENT_TYPES

        self.consumers = []
        self.producers = []
        self.routers = []
        
       
        num_routers = grid_rows * grid_cols

        net_core = Graph(grid_rows, grid_cols)

        # assign consumers and producers with gateway routers
        for i in range(num_consumers):
            self.consumers.append(
                Node({'' : net_core.getRandomRouter()}, 0, "c" + str(i))
            )
        for i in range(num_producers):
            self.producers.append(
                Node({'' : net_core.getRandomRouter()}, self.NUM_CONTENT_TYPES / num_producers, "p" + str(i))
            )
        # set FIBs in routers
        net_core.setRoutesToProducers(self.producers)
    
        # populate content
        content_types = []
        for i in range(self.NUM_CONTENT_TYPES):
            next_producer = self.producers[i % num_producers]
            content_name = next_producer.get_name() + "_c" + str(i)
            next_producer.content_store.add_item(Packet(content_name, data=i))
            content_types.append(content_name)
    
        
        #generate probability distribution
        ZIPF_S = 1.2
        zipf_weights = [(1/k**ZIPF_S)/ (sum([1/n**ZIPF_S for n in range(1, self.NUM_CONTENT_TYPES+1)])) for k in range(1,self.NUM_CONTENT_TYPES+1)]

        #make content requests 
        for i in range(0, self.NUM_REQUESTS_PER_CONSUMER):
            for consumer in self.consumers:
                print("NEW REQUEST")
                pkt = Packet(choices(content_types, zipf_weights)[0])
                consumer.get_gateway().receive( pkt, consumer)

        

if __name__ == "__main__":
    Simulator(2,2)
